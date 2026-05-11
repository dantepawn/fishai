### ml-depth-pro repo for inference
# script to calulate depth on the AWS VM 


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import depth_pro
from pathlib import Path

def main():
    # Load images iterator
    img_path = Path("/home/sagemaker-user/ml-depth-pro/Depth-Anything-V2/images/").glob("*.jpg")


    # Load model
    model, transform = depth_pro.create_model_and_transforms(device = "cuda")
    model.eval()
    for image_path in img_path:
        # Load image
        depth_map, image, _, f_px = depth_pro.load_rgb(image_path)
        image = transform(image)
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m]
        focallength_px = prediction["focallength_px"] 

        # Save depth map
        depth_cpu = depth.cpu().numpy()
        depth_normalized = (depth_cpu - np.min(depth_cpu)) / (np.max(depth_cpu) - np.min(depth_cpu))
        plt.imshow(depth_normalized, cmap='viridis')
        output_path = "/home/sagemaker-user/Depth-Anything-V2/images/depths/" + image_path.stem + "_depth.jpg"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
        
