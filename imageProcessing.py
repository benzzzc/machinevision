import os
import shutil
import random
import cv2

# Set up file grab, set the input and output files and their corresponding folders
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER_ROOT = os.path.join(SCRIPT_DIR, "rawPhotos")
OUTPUT_FOLDER_ROOT = os.path.join(SCRIPT_DIR, "splitDataset")

# Split Percentages
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def create_split_folders(base_dir, classes):
    """Creates train, val, and test folders with class subfolders."""
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

def main():
    if not os.path.exists(INPUT_FOLDER_ROOT):
        print(f"Input folder not found: {INPUT_FOLDER_ROOT}")
        return

    print(f"Scanning raw images in: {INPUT_FOLDER_ROOT}")
    
    # Get all class folder names (e.g., 'normal_sand_0.00', 'normal_sand_0.25', etc.)
    classes = [d for d in os.listdir(INPUT_FOLDER_ROOT) if os.path.isdir(os.path.join(INPUT_FOLDER_ROOT, d))]
    
    if not classes:
        print("No class folders found! Make sure your raw photos are sorted into folders like 'normal_sand_0.00'.")
        return
        
    create_split_folders(OUTPUT_FOLDER_ROOT, classes)

    total_copied = 0

    for cls in classes:
        class_dir = os.path.join(INPUT_FOLDER_ROOT, cls)
        # Get all valid images
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Shuffle the list to ensure random distribution
        random.shuffle(images)

        # Calculate split indices
        train_end = int(len(images) * TRAIN_RATIO)
        val_end = train_end + int(len(images) * VAL_RATIO)

        # Slice the list into the three groups
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Helper function to copy files and verify they aren't corrupted
        def copy_images(image_list, split_name):
            nonlocal total_copied
            for img_name in image_list:
                src_path = os.path.join(class_dir, img_name)
                dest_path = os.path.join(OUTPUT_FOLDER_ROOT, split_name, cls, img_name)
                
                # Quick check to ensure the file is a readable image before copying
                img_check = cv2.imread(src_path)
                if img_check is not None:
                    shutil.copy2(src_path, dest_path)
                    total_copied += 1
                else:
                    print(f"Skipping corrupted image: {src_path}")

        print(f"Splitting class '{cls}': {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")
        copy_images(train_imgs, 'train')
        copy_images(val_imgs, 'val')
        copy_images(test_imgs, 'test')

    print(f"\nDone! Successfully processed and split {total_copied} images.")
    print(f"Your ready-to-train dataset is located at: {OUTPUT_FOLDER_ROOT}")

if __name__ == "__main__":
    main()