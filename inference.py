#hopefully this fully intergrates with the realsense camera video feed for the image classifier model
# NOTE snippeted from https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
# NOTE maybe change the model from .keras to ONYX, also from ^^ see if it is worth adding depth to crop out the background, maybe useless as background different colour 
# or change to HSV... 
# should work for all realsense with colour lense
# CONVERT models to .trt

import cv2
import numpy as np
import pyrealsense2 as rs
import tensorflow as tf

# Load the model
MODEL_PATH = 'model.keras' # Change the name of the model
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ['Soil', 'Soily', 'Soilest'] 
IMG_SIZE = 224

yolo_model = YOLO('best.pt')

# Configure colour streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with colour sensor")
    exit(0)

config.enable_stream(rs.stream.color, 600, 600, rs.format.bgr8, 30) # colour only no need for depth atm

pipeline.start(config)
print("Pipeline started. Press 'q' to quit.")

# Frame Grab 
try:
    while True:
        
        # Wait for coherent colour frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Processs the image for tensorflowkeras Efficientnet
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # camera uses BGR, we need RGB as this is what I trained on, if this is a problem change dataset
        resized_image = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE)) # squishes camera feed to image size
        input_tensor = np.expand_dims(resized_image, axis=0) # add a dummy batch dimensions as the model expects it

        # Predict the feed, outputs the class with the greatest probability
        predictions = model.predict(input_tensor, verbose=0) # no more loading bars 
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_label = CLASS_NAMES[predicted_class_idx]

        display_text = f"Context: {predicted_label} ({soil_confidence:.2f})"
        cv2.putText(color_image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Yolo
        results = yolo_model(color_image, verbose=False)
        
        for result in results:
            for box in result.boxes:
                # Get the pixel boundaries and confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                artifact_confidence = float(box.conf[0])
                
                # Only draw bounding box if over 60% confidence
                if artifact_confidence < 0.60:
                    continue
                    
                # Draw a green bounding box around the artifact
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Write "Artifact" and the confidence score right above the box
                cv2.putText(color_image, f"Artifact {artifact_confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show output on monitor
        cv2.imshow('RealSense AI Vision Pipeline', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Shutting down camera kindfully...")
    pipeline.stop()
    cv2.destroyAllWindows()
