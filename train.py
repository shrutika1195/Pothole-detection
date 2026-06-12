from ultralytics import YOLO

def main():
    print("Starting Phase 1: YOLOv8 Instance Segmentation Training")
    print("Initializing model...")
    
    # Load the pre-trained YOLOv8 segmentation model
    # We use the 'nano' version (yolov8n-seg.pt) as it trains fast and runs in real-time
    model = YOLO('yolov8n-seg.pt')

    # Before running this, make sure you download the Kaggle dataset:
    # https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset
    # Unzip it and locate the 'data.yaml' file inside the dataset folder.
    
    dataset_yaml_path = r"C:\Users\geram\Downloads\DIP\Pothole_Segmentation_YOLOv8\data.yaml" # <-- UPDATE THIS PATH
    
    print("Applying Digital Image Processing (DIP) techniques and starting training...")
    
    # Train the model with the exact DIP techniques mentioned in your report
    results = model.train(
        data=dataset_yaml_path,
        epochs=50,               # Number of training loops (you can increase this later)
        imgsz=640,               # DIP Technique 1: Image Resizing to 640x640
        mosaic=1.0,              # DIP Technique 2: Mosaic Augmentation (1.0 means 100% probability)
        hsv_h=0.015,             # DIP Technique 3a: HSV Color Space Augmentation (Hue)
        hsv_s=0.7,               # DIP Technique 3b: HSV Color Space Augmentation (Saturation)
        hsv_v=0.4,               # DIP Technique 3c: HSV Color Space Augmentation (Value/Brightness)
        device='cpu',            # Switched to CPU since CUDA is not detected right now
        project='Pothole_Project',
        name='YOLOv8_Segmentation_Run'
    )
    
    print("Training complete! The model weights are saved in the 'Pothole_Project' folder.")

if __name__ == '__main__':
    # Make sure to install the required library first by running:
    # pip install ultralytics
    main()