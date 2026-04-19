import os
import shutil
import random
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Your two separate input sources
ARTIFACT_INPUT_ROOT = os.path.join(SCRIPT_DIR, "artifactRawPhotos")
CLEAN_INPUT_ROOT = os.path.join(SCRIPT_DIR, "rawPhotos")

# The final destination for the B0 model
OUTPUT_FOLDER_ROOT = os.path.join(SCRIPT_DIR, "splitDatasetArtifacts")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def get_all_images_from_root(root_dir):
    """Crawls through all subfolders in a root directory and returns a list of image paths."""
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

def process_and_split(image_paths, target_class_name):
    """Shuffles, splits, and copies images into their train/val/test folders."""
    random.shuffle(image_paths)
    
    train_end = int(len(image_paths) * TRAIN_RATIO)
    val_end = train_end + int(len(image_paths) * VAL_RATIO)
    
    splits = {
        'train': image_paths[:train_end],
        'val': image_paths[train_end:val_end],
        'test': image_paths[val_end:]
    }
    
    total_copied = 0
    for split_name, paths in splits.items():
        # Create the destination folder (e.g., splitDatasetArtifacts/train/Artifact_Present)
        dest_dir = os.path.join(OUTPUT_FOLDER_ROOT, split_name, target_class_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        for i, src_path in enumerate(paths):
            # Give the file a unique name to prevent overwriting files with the same name from different subfolders
            new_filename = f"{target_class_name}_{i}.jpg"
            dest_path = os.path.join(dest_dir, new_filename)
            
            img_check = cv2.imread(src_path)
            if img_check is not None:
                shutil.copy2(src_path, dest_path)
                total_copied += 1
            else:
                print(f"Skipping corrupted image: {src_path}")
                
    print(f"  -> Copied {total_copied} images to {target_class_name}")

def main():
    print("Gathering Artifact images...")
    artifact_images = get_all_images_from_root(ARTIFACT_INPUT_ROOT)
    print(f"Found {len(artifact_images)} artifact images.")
    
    print("Gathering Clean Sand images...")
    clean_images = get_all_images_from_root(CLEAN_INPUT_ROOT)
    print(f"Found {len(clean_images)} clean images.")
    
    print("\nProcessing splits...")
    process_and_split(artifact_images, "Artifact_Present")
    process_and_split(clean_images, "No_Artifact")
    
    print(f"\nDone! Binary dataset ready at: {OUTPUT_FOLDER_ROOT}")

if __name__ == "__main__":
    main()