import os
import requests

BASE_URL = "https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense/resolve/main"

image_dir = "data/Landslide4Sense/images/train"
mask_dir = "data/Landslide4Sense/annotations/train"

os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# We already have image_1 and mask_1.
# Download images 2-10.
for i in range(2, 11):

    image_url = f"{BASE_URL}/images/train/image_{i}.h5"
    mask_url = f"{BASE_URL}/annotations/train/mask_{i}.h5"

    image_path = os.path.join(image_dir, f"image_{i}.h5")
    mask_path = os.path.join(mask_dir, f"mask_{i}.h5")

    print(f"\nDownloading pair {i}/10...")

    if not os.path.exists(image_path):
        r = requests.get(image_url, stream=True)
        print("Image status:", r.status_code)

        if r.status_code == 200:
            with open(image_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("Image saved:", image_path)
        else:
            print("Could not download image", i)

    else:
        print("Image already exists")

    if not os.path.exists(mask_path):
        r = requests.get(mask_url, stream=True)
        print("Mask status:", r.status_code)

        if r.status_code == 200:
            with open(mask_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("Mask saved:", mask_path)
        else:
            print("Could not download mask", i)

    else:
        print("Mask already exists")

print("\nDownload process completed.")
