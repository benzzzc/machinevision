# NOTE source realsense_env/bin/activate
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import re # Added for robust filename searching

def capture_color_image():
    print("--- Dataset Setup (EfficientNet-B7 / 600x600) ---")
    custom_tag = input("Enter a custom tag (e.g., dry): ").strip()
    
    while True:
        try:
            sand_input = float(input("Enter sand ratio (0.0 to 1.0): "))
            if 0.0 <= sand_input <= 1.0: 
                # CRITICAL: We force the ratio to 2 decimal places (e.g., "0.25")
                # This ensures the folder name and filename are consistent every time.
                sand_ratio_str = "{:.2f}".format(sand_input)
                break
        except ValueError: 
            print("Invalid input.")

    base_folder = "testPhotos"
    save_folder = os.path.join(base_folder, f"{custom_tag}_sand_{sand_ratio_str}")
    
    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        img_counter = 0
    else:
        # --- ROBUST COUNTER SCANNER ---
        # Instead of a loop, we look at every file in the folder 
        # and find the highest number in the "img_XXX" slot.
        existing_files = os.listdir(save_folder)
        indices = []
        for f in existing_files:
            # This looks for the 3-digit number after 'img_'
            match = re.search(r'img_(\d{3})_', f)
            if match:
                indices.append(int(match.group(1)))
        
        # Start at the highest number + 1, or 0 if the folder is empty
        img_counter = max(indices) + 1 if indices else 0

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline_started = False 
    try:
        pipeline.start(config)
        pipeline_started = True
        print(f"\nTargeting: {save_folder}")
        print(f"Resuming at index: {img_counter:03d}")
        print("SPACE: Capture | Q: Quit")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())

            # Visual guide for the B7 crop area
            preview = color_image.copy()
            cv2.rectangle(preview, (340, 60), (940, 660), (0, 255, 0), 2)
            cv2.imshow("RealSense D455 Collector", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 32: # SPACE
                # Center Crop 600x600 for EfficientNet-B7
                b7_image = color_image[60:660, 340:940]
                
                # Use the standardized string for the filename
                image_name = f"{custom_tag}_img_{img_counter:03d}_{sand_ratio_str}.png"
                cv2.imwrite(os.path.join(save_folder, image_name), b7_image)
                
                print(f"Saved {image_name}")
                img_counter += 1

            elif key == ord('q') or key == 27:
                break
    finally:
        if pipeline_started: 
            pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_color_image()