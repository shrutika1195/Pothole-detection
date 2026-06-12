🚗 Pothole Detection \& Severity Analysis - Setup Guide



Hello! This guide will help you set up and run the pothole detection project on your computer.



This project uses a trained YOLOv8 AI model to find potholes and a Fuzzy Logic system to score how dangerous they are based on their size and depth.



Step 1: Install Python



Make sure you have Python installed on your computer. You can download it from python.org.



Step 2: Install the Required Libraries



Open your terminal or command prompt. Type in this command and press Enter to install the magic tools we need:



pip install ultralytics scikit-fuzzy streamlit opencv-python





Step 3: Organize the Files



Make sure all these files are in the same folder together:



app.py (The web app code)



bulk\_process.py (The folder processing code)



fuzzy\_logic\_severity.py (The math behind the danger scores)



best.pt (The trained AI brain - make sure the paths in the code point to this file!)



A folder with some test images.



Step 4: Run the Web App (Interactive Mode)



If you want to use the beautiful web interface to upload pictures one by one, open your terminal in the project folder and run:



streamlit run app.py





A browser window will automatically pop up where you can drop images and see the results!



Step 5: Run the Bulk Processor (Collage Maker)



If you want to process a whole folder of images at once and create side-by-side collages, open bulk\_process.py and make sure the input\_folder\_path points to your images. Then run:



python bulk\_process.py





Check the terminal for the final metrics, and look in the output folder for your collages!

