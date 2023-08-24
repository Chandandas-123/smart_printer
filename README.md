# Smart Printer with Face Detection

The Smart Printer with Face Detection project combines computer vision and printing technology to create a unique smart printing system. This project involves capturing images from a webcam, detecting faces, and printing the images with detected faces using a smart printer.

## Features
Face Detection: Utilizes OpenCV's Haarcascade classifier to detect faces in real-time.
Image Concatenation: Concatenates the captured image with a footer image to create a printable image.
Smart Printing: Prints the final concatenated image using the default printer.
User Interaction: Allows users to confirm printing or discard the image.
## Prerequisites 
* Python 3.x
* OpenCV (cv2)
* win32print and win32ui libraries
* numpy
* PIL (Python Imaging Library)

## Setup Instructions

        1. Clone this repository to your local machine:
              git clone https://github.com/your-username/smart-printer-face-detection.git
              cd smart-printer-face-detection
    
        2. cd smart-printer-face-detection
              Install the required Python packages using pip:
              pip install opencv-python numpy pillow pywin32
    
        3. Place a footer.png image in the project directory. This image will be concatenated with the detected face image before printing.


## Usage
       1. Run the Python script using:
             python smart_printer_face_detection.py
       2. The webcam feed will open, and the script will detect faces in real-time. It will draw rectangles around the detected faces.

       3. Once a face is detected, the script will capture the image and concatenate it with the footer image.

       4. The concatenated image will be displayed. Press y to print the image using the default printer or any other key to discard it.


## Contributions
Contributions to this project are welcome! If you have any suggestions or improvements, feel free to create a pull request.


