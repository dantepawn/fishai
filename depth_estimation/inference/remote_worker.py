import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import modal


custom_image = modal.Image.from_dockerfile("C:/Users/daeil/Documents/Data Science/Fish/Dockerfile")
app = modal.App("remote_worker_example", image=custom_image)




@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))
