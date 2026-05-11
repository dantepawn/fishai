#RAFT_stereo file for batch processing of stereo images using RAFT Stereo model. 
# in Github repository
import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_batch_images(left_files, right_files, batch_size):
    """Load a batch of stereo image pairs"""
    left_batch = []
    right_batch = []
    
    for i in range(batch_size):
        if i < len(left_files) and i < len(right_files):
            left_img = load_image(left_files[i]).squeeze(0)  # Remove batch dimension
            right_img = load_image(right_files[i]).squeeze(0)  # Remove batch dimension
            left_batch.append(left_img)
            right_batch.append(right_img)
    
    if left_batch:
        left_batch = torch.stack(left_batch, dim=0)  # Stack into batch
        right_batch = torch.stack(right_batch, dim=0)  # Stack into batch
        return left_batch, right_batch
    return None, None

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        # Process images in batches
        batch_size = args.batch_size
        num_batches = (len(left_images) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(left_images))
            
            batch_left_files = left_images[start_idx:end_idx]
            batch_right_files = right_images[start_idx:end_idx]
            
            # Load batch of images
            batch_left, batch_right = load_batch_images(batch_left_files, batch_right_files, len(batch_left_files))
            
            if batch_left is None:
                continue
                
            # Pad all images in the batch to the same size
            padder = InputPadder(batch_left.shape, divis_by=32)
            batch_left, batch_right = padder.pad(batch_left, batch_right)

            # Run inference on the entire batch
            _, flow_up_batch = model(batch_left, batch_right, iters=args.valid_iters, test_mode=True)
            flow_up_batch = padder.unpad(flow_up_batch)

            # Save results for each image in the batch
            for i, (left_file, right_file) in enumerate(zip(batch_left_files, batch_right_files)):
                flow_up = flow_up_batch[i].squeeze()
                
                # Extract filename for saving
                file_stem = Path(left_file).stem
                if args.save_numpy:
                    np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy())
                plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy(), cmap='jet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()
    demo(args)
