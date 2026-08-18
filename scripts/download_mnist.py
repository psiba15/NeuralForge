import os
import gzip
import shutil
import urllib.request

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

os.makedirs("data", exist_ok=True)

for fname in FILES:
    gz_path = os.path.join("data", fname)
    out_path = gz_path[:-3]
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping")
        continue
    print(f"Downloading {fname} ...")
    urllib.request.urlretrieve(BASE_URL + fname, gz_path)
    print(f"Extracting {fname} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"Done: {out_path}")

print("MNIST download complete.")