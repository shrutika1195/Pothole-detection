import cv2
import numpy as np
from ultralytics import YOLO

# Import our fuzzy logic system from the previous file
# Note: If you saved the previous file as 'fuzzy.py', change this to:
# from fuzzy import setup_fuzzy_system, calculate_severity
from fuzzy_logic_severity import setup_fuzzy_system, calculate_severity

def process_image(image_path, model_path):
    print("Starting Phase 3: Processing Image with Filters...")
    
    # 1. Load the trained YOLOv8 model and the Fuzzy Logic System
    model = YOLO(model_path)
    fuzzy_system = setup_fuzzy_system()
    
    # 2. Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    img_height, img_width = img.shape[:2]
    total_pixels = img_height * img_width
    
    # 3. Run YOLOv8 to find all possible potholes
    # We lower the confidence (conf=0.15) to catch every single mark, even faint ones.
    results = model(img, conf=0.15)
    
    # 4. Process the results
    for result in results:
        # Check if any potholes were found
        if result.masks is None:
            print("No potholes detected in this image.")
            continue
            
        # Get the masks (exact pixel shapes) and bounding boxes
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xywh.cpu().numpy() # [x_center, y_center, width, height]
        
        for i, mask in enumerate(masks):
            # Calculate Area (Percentage of the whole image)
            pothole_pixels = np.count_nonzero(mask)
            area_percentage = (pothole_pixels / total_pixels) * 100
            
            # PERSPECTIVE FIX: Dashcams make far-away objects look flat.
            # We add a bonus to the depth score if it stretches wide across the road.
            w, h = boxes[i][2], boxes[i][3]
            width_stretch = (w / img_width) * 5 
            depth_score = min(((h / w) * 5) + width_stretch, 10) 
            
            # NEW FILTERING LAYER: Ignore false positives before checking severity
            # If the detected shape is extremely tiny (less than 0.1% area) or perfectly flat, we skip it.
            if area_percentage < 0.1 or depth_score < 0.5:
                print(f"Detection {i+1} -> Ignored by Filter (Too small or flat. Area: {area_percentage:.2f}%)")
                continue
            
            # 5. Get the Severity Score from our Fuzzy Logic system
            score, category = calculate_severity(area_percentage, depth_score, fuzzy_system)
            
            print(f"Pothole {i+1} -> Area: {area_percentage:.2f}%, Depth Score: {depth_score:.2f}")
            print(f"Result: {category} ({score:.2f}/100)")
            
            # 6. Draw on the image to show the results
            # Resize mask to match original image size
            mask_resized = cv2.resize(mask, (img_width, img_height))
            
            # Create a red overlay for the pothole exact shape
            color = (0, 0, 255) # Red in BGR
            colored_mask = np.zeros_like(img)
            colored_mask[mask_resized > 0.5] = color
            
            # Blend the original image with the red mask
            img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)
            
            # Add text with the severity score
            x, y = int(boxes[i][0] - w/2), int(boxes[i][1] - h/2)
            cv2.putText(img, f"{category}: {score:.1f}", (x, max(y-10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
    # Save and show the final output
    output_path = "final_output.jpg"
    cv2.imwrite(output_path, img)
    print(f"\nDone! The processed image has been saved as '{output_path}'.")

if __name__ == '__main__':
    # Make sure your paths are correct. 
    # Use the 'best.pt' file created from your Phase 1 training.
    trained_model_path = r"runs\segment\Pothole_Project\YOLOv8_Segmentation_Run\weights\best.pt"
    
    # Put a test image inside your folder and update this path
    # You can grab any image from the 'valid/images' folder in your Kaggle dataset
    test_image_path = r"C:\Users\geram\Downloads\DIP\Pothole_Segmentation_YOLOv8\valid\images\test_image.jpg" # <-- UPDATE THIS
    
    process_image(test_image_path, trained_model_path)