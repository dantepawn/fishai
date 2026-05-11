import os
import sys
import glob
import argparse
import logging
import torch
import cv2
import numpy as np
import imageio
import open3d as o3d
from pathlib import Path

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


def batch_stereo_inference(left_files, right_files, intrinsic_file, ckpt_dir, 
                          output_base_dir, scale=1, hiera=0, z_far=10, 
                          valid_iters=32, get_pc=1, remove_invisible=1, 
                          denoise_cloud=1, denoise_nb_points=30, denoise_radius=0.03,
                          visualize_pc=False):
    """
    Perform stereo inference on multiple image pairs.
    
    Args:
        left_files: List of paths to left images
        right_files: List of paths to right images  
        intrinsic_file: Path to camera intrinsic matrix file
        ckpt_dir: Path to pretrained model checkpoint
        output_base_dir: Base directory to save all results
        scale: Image downscale factor (must be <=1)
        hiera: Use hierarchical inference for high-res images
        z_far: Maximum depth to clip in point cloud
        valid_iters: Number of flow-field updates
        get_pc: Whether to generate point cloud
        remove_invisible: Remove non-overlapping observations
        denoise_cloud: Whether to denoise point cloud
        denoise_nb_points: Points to consider for denoising
        denoise_radius: Radius for outlier removal
        visualize_pc: Whether to visualize point clouds
    """
    
    # Setup logging and model
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    
    # Load model configuration
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    
    # Update config with parameters
    cfg.update({
        'scale': scale,
        'hiera': hiera,
        'z_far': z_far,
        'valid_iters': valid_iters,
        'get_pc': get_pc,
        'remove_invisible': remove_invisible,
        'denoise_cloud': denoise_cloud,
        'denoise_nb_points': denoise_nb_points,
        'denoise_radius': denoise_radius
    })
    
    args = OmegaConf.create(cfg)
    logging.info(f"Using pretrained model from {ckpt_dir}")
    
    # Load model
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    
    # Load camera intrinsics once
    with open(intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
    K[:2] *= scale
    
    # Process each image pair
    for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
        logging.info(f"Processing pair {i+1}/{len(left_files)}: {os.path.basename(left_file)}")
        
        # Create output directory for this pair
        pair_name = f"pair_{i:04d}_{Path(left_file).stem}"
        output_dir = os.path.join(output_base_dir, pair_name)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load and preprocess images
            img0 = imageio.imread(left_file)
            img1 = imageio.imread(right_file)
            
            assert scale <= 1, "scale must be <=1"
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
            H, W = img0.shape[:2]
            img0_ori = img0.copy()
            
            # Convert to tensors
            img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
            img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
            padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
            img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)
            
            # Run inference
            with torch.cuda.amp.autocast(True):
                if not hiera:
                    disp = model.forward(img0_tensor, img1_tensor, iters=valid_iters, test_mode=True)
                else:
                    disp = model.run_hierachical(img0_tensor, img1_tensor, iters=valid_iters, 
                                               test_mode=True, small_ratio=0.5)
            
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W)
            
            # Save disparity visualization
            vis = vis_disparity(disp)
            vis_combined = np.concatenate([img0_ori, vis], axis=1)
            imageio.imwrite(f'{output_dir}/disparity_vis.png', vis_combined)
            
            # Save raw disparity
            np.save(f'{output_dir}/disparity.npy', disp)
            
            # Remove invisible points if requested
            if remove_invisible:
                yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
                us_right = xx - disp
                invalid = us_right < 0
                disp[invalid] = np.inf
            
            # Generate point cloud if requested
            if get_pc:
                depth = K[0,0] * baseline / disp
                np.save(f'{output_dir}/depth_meter.npy', depth)
                
                xyz_map = depth2xyzmap(depth, K)
                pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
                
                # Filter points by depth
                keep_mask = (np.asarray(pcd.points)[:,2] > 0) & (np.asarray(pcd.points)[:,2] <= z_far)
                keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                pcd = pcd.select_by_index(keep_ids)
                
                # Save point cloud
                o3d.io.write_point_cloud(f'{output_dir}/cloud.ply', pcd)
                
                # Denoise if requested
                if denoise_cloud:
                    logging.info(f"Denoising point cloud for pair {i+1}...")
                    cl, ind = pcd.remove_radius_outlier(nb_points=denoise_nb_points, 
                                                       radius=denoise_radius)
                    inlier_cloud = pcd.select_by_index(ind)
                    o3d.io.write_point_cloud(f'{output_dir}/cloud_denoise.ply', inlier_cloud)
                    pcd = inlier_cloud
                
                # Visualize if requested
                if visualize_pc:
                    logging.info(f"Visualizing point cloud for pair {i+1}. Press ESC to continue.")
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name=f"Point Cloud - Pair {i+1}")
                    vis.add_geometry(pcd)
                    vis.get_render_option().point_size = 1.0
                    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
                    vis.run()
                    vis.destroy_window()
            
            logging.info(f"Results for pair {i+1} saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error processing pair {i+1} ({left_file}, {right_file}): {str(e)}")
            continue
    
    logging.info(f"Batch processing complete. All results saved to {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch stereo inference")
    parser.add_argument('--left_dir', type=str, required=True, help='Directory containing left images')
    parser.add_argument('--right_dir', type=str, required=True, help='Directory containing right images')
    parser.add_argument('--left_pattern', type=str, default='*.png', help='Pattern for left images')
    parser.add_argument('--right_pattern', type=str, default='*.png', help='Pattern for right images')
    parser.add_argument('--intrinsic_file', type=str,default=f'{code_dir}/../assets/K.txt', required=True, help='Camera intrinsic matrix file')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for all results')
    parser.add_argument('--scale', type=float, default=1, help='Image downscale factor')
    parser.add_argument('--hiera', type=int, default=0, help='Use hierarchical inference')
    parser.add_argument('--z_far', type=float, default=10, help='Max depth for point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='Number of iterations')
    parser.add_argument('--get_pc', type=int, default=1, help='Generate point cloud')
    parser.add_argument('--remove_invisible', type=int, default=1, help='Remove invisible points')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='Denoise point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='Points for denoising')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='Denoising radius')
    parser.add_argument('--visualize_pc', action='store_true', help='Visualize each point cloud')
    
    args = parser.parse_args()
    
    # Find matching image pairs
    left_files = sorted(glob.glob(os.path.join(args.left_dir, args.left_pattern)))
    right_files = sorted(glob.glob(os.path.join(args.right_dir, args.right_pattern)))
    
    if len(left_files) != len(right_files):
        logging.error(f"Mismatch: {len(left_files)} left images, {len(right_files)} right images")
        return
    
    if len(left_files) == 0:
        logging.error("No images found with the specified patterns")
        return
    
    logging.info(f"Found {len(left_files)} image pairs to process")
    
    # Run batch inference
    batch_stereo_inference(
        left_files=left_files,
        right_files=right_files,
        intrinsic_file=args.intrinsic_file,
        ckpt_dir=args.ckpt_dir,
        output_base_dir=args.output_dir,
        scale=args.scale,
        hiera=args.hiera,
        z_far=args.z_far,
        valid_iters=args.valid_iters,
        get_pc=args.get_pc,
        remove_invisible=args.remove_invisible,
        denoise_cloud=args.denoise_cloud,
        denoise_nb_points=args.denoise_nb_points,
        denoise_radius=args.denoise_radius,
        visualize_pc=args.visualize_pc
    )


if __name__ == "__main__":
    main()
