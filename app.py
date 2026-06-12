import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Import our fuzzy logic system
from fuzzy_logic_severity import setup_fuzzy_system, calculate_severity

# Set up the web page layout
st.set_page_config(page_title="Pothole Detection App", layout="wide")

# We use "cache_resource" so the model only loads once, making the app much faster
@st.cache_resource
def load_models():
    # Make sure this path points to your trained YOLOv8 model!
    model_path = 'runs/segment/Pothole_Project/YOLOv8_Segmentation_Run/weights/best.pt'
    model = YOLO(model_path)
    fuzzy_system = setup_fuzzy_system()
    return model, fuzzy_system

def add_title_to_image(img, title):
    # This adds a white bar at the top of the image with a title
    h, w = img.shape[:2]
    top_border = 50
    img_with_border = cv2.copyMakeBorder(img, top_border, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(img_with_border, title, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img_with_border

def create_collage(image, model, fuzzy_system):
    # Convert uploaded image to an OpenCV format
    img = np.array(image)
    # Streamlit uses RGB, OpenCV uses BGR, so we convert it
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img_height, img_width = img.shape[:2]
    total_pixels = img_height * img_width
    
    # Create copies for our three panels
    panel_1_original = img.copy()
    panel_2_raw = img.copy()
    panel_3_filtered = img.copy()
    
    # Run YOLOv8 to find everything (low confidence)
    results = model(img, conf=0.15, verbose=False)
    
    for result in results:
        if result.masks is None:
            continue
            
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xywh.cpu().numpy()
        
        for i, mask in enumerate(masks):
            # Calculate size and shape
            pothole_pixels = np.count_nonzero(mask)
            area_percentage = (pothole_pixels / total_pixels) * 100
            w, h = boxes[i][2], boxes[i][3]
            
            # PERSPECTIVE FIX: Dashcams make far-away objects look flat.
            # We add a bonus to the depth score if it stretches wide across the road.
            width_stretch = (w / img_width) * 5 
            depth_score = min(((h / w) * 5) + width_stretch, 10) 
            
            # Resize mask to draw it
            mask_resized = cv2.resize(mask, (img_width, img_height))
            
            # ---- PANEL 2: RAW YOLO ----
            color_raw = (0, 165, 255) # Orange
            colored_mask_raw = np.zeros_like(panel_2_raw)
            colored_mask_raw[mask_resized > 0.5] = color_raw
            panel_2_raw = cv2.addWeighted(panel_2_raw, 1, colored_mask_raw, 0.5, 0)
            
            # ---- PANEL 3: FILTERED + FUZZY ----
            # Apply our smart filters
            if area_percentage < 0.1 or depth_score < 0.5:
                continue
            
            # Calculate severity
            score, category = calculate_severity(area_percentage, depth_score, fuzzy_system)
            
            # Draw red mask
            color_filtered = (0, 0, 255) # Red
            colored_mask_filtered = np.zeros_like(panel_3_filtered)
            colored_mask_filtered[mask_resized > 0.5] = color_filtered
            panel_3_filtered = cv2.addWeighted(panel_3_filtered, 1, colored_mask_filtered, 0.5, 0)
            
            # Add score text
            x, y = int(boxes[i][0] - w/2), int(boxes[i][1] - h/2)
            cv2.putText(panel_3_filtered, f"{category}", (x, max(y-25, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(panel_3_filtered, f"Score: {score:.1f}", (x, max(y-5, 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add titles to each panel
    panel_1_original = add_title_to_image(panel_1_original, "1. Original Image")
    panel_2_raw = add_title_to_image(panel_2_raw, "2. Raw YOLO (No Filter)")
    panel_3_filtered = add_title_to_image(panel_3_filtered, "3. Filtered + Fuzzy Logic")
    
    # Combine the three images side-by-side
    collage = np.hstack((panel_1_original, panel_2_raw, panel_3_filtered))
    
    # Convert back to RGB for Streamlit to display properly
    collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    return collage_rgb

# --- Web App UI starts here ---

st.title("🚗 Real-Time Pothole Detection & Severity Analysis")
st.write("Upload a dashcam image of a road. The app will use our Hybrid YOLOv8 + Fuzzy Logic model to detect potholes and score their danger level.")

# Load our models
with st.spinner("Loading AI Models..."):
    yolo_model, fuzzy_sys = load_models()

# Create a file uploader button
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    st.write("Processing the image through the pipeline...")
    
    # Run our collage function
    with st.spinner("Analyzing road defects..."):
        final_collage = create_collage(image, yolo_model, fuzzy_sys)
        
    st.success("Analysis Complete!")
    
    # Display the final collage on the web page
    st.image(final_collage, caption="Left: Original | Middle: Raw AI Output | Right: Filtered AI with Fuzzy Logic Severity", use_column_width=True)