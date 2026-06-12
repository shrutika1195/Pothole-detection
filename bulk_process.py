import os
import cv2
import numpy as np
from ultralytics import YOLO

# Import our fuzzy logic system
from fuzzy_logic_severity import setup_fuzzy_system, calculate_severity

def add_title_to_image(img, title):
    # This adds a white bar at the top of the image with a title
    h, w = img.shape[:2]
    top_border = 50
    # Create a white border
    img_with_border = cv2.copyMakeBorder(img, top_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # Add black text to the center of the border
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(img_with_border, title, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img_with_border

def process_folder(input_folder, output_folder, model_path):
    print(f"Starting Phase 3 Bulk Processing with Metric Tracking...")
    print(f"Reading images from: {input_folder}")
    
    # 1. Load the model and fuzzy system
    model = YOLO(model_path)
    fuzzy_system = setup_fuzzy_system()
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("No images found in the input folder! Please check the path.")
        return

    # METRIC COUNTERS
    severity_counts = {
        'Low Priority': 0,
        'Medium Priority': 0,
        'High Priority': 0
    }
    total_raw_detections = 0
    filtered_out_count = 0

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        
        # 2. Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        total_pixels = img_height * img_width
        
        # Create copies for our three panels
        panel_1_original = img.copy()
        panel_2_raw = img.copy()
        panel_3_filtered = img.copy()
        
        print(f"Processing: {img_name}")
        
        # 3. Run YOLOv8 to find everything (low confidence)
        results = model(img, conf=0.15, verbose=False)
        
        for result in results:
            if result.masks is None:
                continue
                
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xywh.cpu().numpy()
            
            for i, mask in enumerate(masks):
                # Count EVERY raw detection YOLO makes
                total_raw_detections += 1
                
                # Calculate size and shape
                pothole_pixels = np.count_nonzero(mask)
                area_percentage = (pothole_pixels / total_pixels) * 100
                w, h = boxes[i][2], boxes[i][3]
                
                # PERSPECTIVE FIX: Dashcams make far-away objects look flat.
                width_stretch = (w / img_width) * 5 
                depth_score = min(((h / w) * 5) + width_stretch, 10) 
                
                # Resize mask to draw it
                mask_resized = cv2.resize(mask, (img_width, img_height))
                
                # ---- PANEL 2: RAW YOLO (Draw EVERYTHING) ----
                color_raw = (0, 165, 255) # Orange for raw detections
                colored_mask_raw = np.zeros_like(panel_2_raw)
                colored_mask_raw[mask_resized > 0.5] = color_raw
                panel_2_raw = cv2.addWeighted(panel_2_raw, 1, colored_mask_raw, 0.5, 0)
                
                # ---- PANEL 3: FILTERED + FUZZY ----
                # NEW SMARTER FILTER: Kick out tiny spots (area < 0.6) OR extreme "pancakes" (height is less than 20% of width)
                if area_percentage < 0.6 or (h / w) < 0.2:
                    # TRACK THE JUNK: Add a point to our filter counter!
                    filtered_out_count += 1
                    continue 
                
                # Calculate severity
                score, category = calculate_severity(area_percentage, depth_score, fuzzy_system)
                
                # Update our counter for this severity category
                if category in severity_counts:
                    severity_counts[category] += 1
                
                # Draw red mask
                color_filtered = (0, 0, 255) # Red
                colored_mask_filtered = np.zeros_like(panel_3_filtered)
                colored_mask_filtered[mask_resized > 0.5] = color_filtered
                panel_3_filtered = cv2.addWeighted(panel_3_filtered, 1, colored_mask_filtered, 0.5, 0)
                
                # Add score text
                x, y = int(boxes[i][0] - w/2), int(boxes[i][1] - h/2)
                cv2.putText(panel_3_filtered, f"{category}", (x, max(y-25, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(panel_3_filtered, f"Score: {score:.1f}", (x, max(y-5, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 4. Add titles to each panel
        panel_1_original = add_title_to_image(panel_1_original, "1. Original Image")
        panel_2_raw = add_title_to_image(panel_2_raw, "2. Raw YOLO (No Filter)")
        panel_3_filtered = add_title_to_image(panel_3_filtered, "3. Filtered + Fuzzy Logic")
        
        # 5. Combine the three images side-by-side
        collage = np.hstack((panel_1_original, panel_2_raw, panel_3_filtered))
        
        # Save the final collage
        output_path = os.path.join(output_folder, f"collage_{img_name}")
        cv2.imwrite(output_path, collage)

    # Calculate total real potholes graded
    total_graded = severity_counts['Low Priority'] + severity_counts['Medium Priority'] + severity_counts['High Priority']

    # Print out the final story
    print(f"\nAll done! Check the '{output_folder}' folder for your collages.")
    print("-" * 40)
    print("FINAL PIPELINE METRICS (Proof of Filtering):")
    print(f"Total Images Checked: {len(image_files)}")
    print(f"Total Raw Detections by YOLO: {total_raw_detections}")
    print(f"Junk Detections Removed by Filter: {filtered_out_count}")
    print(f"Actual Potholes Graded by Fuzzy Logic: {total_graded}")
    print("-" * 40)
    print("SEVERITY BREAKDOWN:")
    print(f"Low Priority Potholes: {severity_counts['Low Priority']}")
    print(f"Medium Priority Potholes: {severity_counts['Medium Priority']}")
    print(f"High Priority Potholes: {severity_counts['High Priority']}")
    print("-" * 40)

if __name__ == '__main__':
    # Make sure these paths are correct for your computer
    trained_model_path = r"runs\segment\Pothole_Project\YOLOv8_Segmentation_Run\weights\best.pt"
    input_folder_path = r"C:\Users\geram\Downloads\DIP\Pothole_Segmentation_YOLOv8\valid\images" 
    output_folder_path = r"C:\Users\geram\Downloads\DIP\Collage_Results"
    
    process_folder(input_folder_path, output_folder_path, trained_model_path)