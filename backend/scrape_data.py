import os
import requests
from duckduckgo_search import DDGS

queries = {
    "glioma": "brain mri glioma",
    "meningioma": "brain mri meningioma",
    "pneumonia": "chest x-ray pneumonia",
}
base_path = "data/new_collected_data"

def download_images():
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with DDGS() as ddgs:
        for lbl, query in queries.items():
            out_dir = os.path.join(base_path, lbl)
            os.makedirs(out_dir, exist_ok=True)
            results = ddgs.images(query, max_results=5)
            for i, r in enumerate(results):
                try:
                    img_data = requests.get(r['image'], timeout=5).content
                    with open(os.path.join(out_dir, f"scraped_{i}.jpg"), 'wb') as f:
                        f.write(img_data)
                    print(f"Downloaded {lbl} image {i}")
                except Exception as e:
                    print(f"Failed {r['image']}: {e}")

if __name__ == "__main__":
    download_images()
