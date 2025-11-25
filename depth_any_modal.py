## deployed depth anything v3 on modal 


# depth_any_modal.py
# deploy depth anything v3 on modal

# how to use it 
# to iterate a google drive folder containing images, use the following command:
# modal run depth_any_modal.py::run_depth_anything --folder "your google drive folder url"

# to run with two image urls, use the following command:
# modal run depth_any_modal.py::run_depth_anything --left_image_url "your left image url" --right_image_url "your right image url"

# To upload images to the modal volume, use the following commands in your terminal:
#modal volume put depth_anything_V3 "path_to_file.jpg" data/file.jpg 



import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import modal


image_with_repo = (
    modal.Image.debian_slim().apt_install("git", "libgl1", "libglib2.0-0").uv_pip_install("torch>=2", "torchvision" , "gdown" ,"requests" , "matplotlib").run_commands(
        "git clone https://github.com/ByteDance-Seed/depth-anything-3","cd depth-anything-3 && pip install -e ."
    )
)

volume = modal.Volume.from_name("depth_anything_V3", create_if_missing=True)
volume_path = (  # the path to the volume from within the container
    Path("/root") / "data"
)


app = modal.App("depth_anything_v3", image=image_with_repo, volumes={volume_path: volume})

@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("custom-secret")]
)
def run_depth_anything(left_image_url=None, right_image_url=None, folder=None):
    import numpy as np
    import os
    import io
    import requests
    import matplotlib.pyplot as plt
    import torch
    import gdown
    from depth_anything_3.api import DepthAnything3

    telegram_bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    telegram_bot_id    = os.environ["TELEGRAM_BOT_ID"]

    def to_direct_download(url: str) -> str:
        if url and "drive.google.com" in url:
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
                return f"https://drive.google.com/uc?id={file_id}"
            if "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
                return f"https://drive.google.com/uc?id={file_id}"
        return url

    volume_dirs = [Path("/root/data/data"), Path("/root/data")]
    for candidate in volume_dirs:
        if candidate.exists():
            data_dir = candidate
            break
    else:
        data_dir = Path("/root/data")
        data_dir.mkdir(parents=True, exist_ok=True)

    downloads = {}
    if left_image_url:
        direct_url = to_direct_download(left_image_url)
        downloads["left.jpg"] = direct_url
    if right_image_url:
        direct_url = to_direct_download(right_image_url)
        downloads["right.jpg"] = direct_url

    if folder:
        folder_dir = data_dir / "drive_folder"
        folder_dir.mkdir(exist_ok=True)
        direct_folder = to_direct_download(folder)
        gdown.download_folder(direct_folder, output=str(folder_dir), quiet=False, use_cookies=False)

    image_paths = []
    if folder:
        # google drive folder provided
        image_paths = [str(p) for p in sorted(folder_dir.glob("**/*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    elif downloads:
        # individual image urls provided from google drive
        image_paths = [str(data_dir / name) for name in downloads]
    else:
        # default images in the volume
        image_paths = [
            str(data_dir / "1715786993_photo_l3.jpg"),
            str(data_dir / "1715786993_photo_l2.jpg"),
        ]

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3-large")
    model = model.to(device=device)

    print("loading images from:", image_paths)
    prediction = model.inference(
        image_paths,
        export_dir=str(data_dir),
        export_format="npz",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
    )
    api_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendPhoto"

    fig, ax = plt.subplots()
    ax.axis("off")
    for idx, depth_map in enumerate(prediction.depth):
        if idx > 0:
            break  # only send the first image for demo purposes
        # Display the depth map with proper scaling
        im = ax.imshow(depth_map, cmap="plasma", vmin=depth_map.min(), vmax=depth_map.max())

        # Remove any whitespace around the plot
        plt.subplots_adjust(left=0, right=0.85, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        files = {"photo": ("depth_map.png", buf.getvalue(), "image/png")}
        data = {"chat_id": telegram_bot_id, "caption": f"Depth: Max {depth_map.max():.2f} m, Min {depth_map.min():.2f} m"}
        resp = requests.post(api_url, data=data, files=files, timeout=30)
    

    # Access results
    print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
    print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
    print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
    print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
    return "Depth Anything v3 inference completed."

@app.local_entrypoint()
def main():
    print("the model is running", run_depth_anything.remote())


