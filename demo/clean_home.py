import os
import shutil

cache_dir = "/home/ndloc_bk/.cache"
if os.path.exists(cache_dir):
    print(f"Cleaning {cache_dir}...")
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        # Keep Microsoft and playwright if they are needed, delete pip, torch, huggingface, whisper, gdown
        if item in ["pip", "torch", "huggingface", "whisper", "gdown", "yt-dlp", "jedi"]:
            print(f"Deleting {item_path}...")
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
    print("Cleanup complete.")
else:
    print(f"Cache dir {cache_dir} not found.")
