# Project: How to Detect Objects in Video Using MobileNet SSD in OpenCV
# Author: Addison Sears-Collins
# Date created: March 1, 2021
# Description: Object detection using OpenCV
 
import cv2 # Computer vision library
import numpy as np # Scientific computing library 
 
# Make sure the video file is in the same directory as your code
filename = r'C:\Users\PCM\Documents\SSD\dak.mp4'
file_size = (1920,1080) # Assumes 1920x1080 mp4
 
# We want to save the output to a video file
output_filename = r'C:\Users\PCM\Documents\SSD\adak.mp4'
output_frames_per_second = 20.0
 
RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255
 
# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe(r'C:\Users\PCM\Documents\SSD\MobileNetSSD_deploy.prototxt.txt', 
        r'C:\Users\PCM\Documents\SSD\MobileNetSSD_deploy.caffemodel')
 
# List of categories and classes
categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
               4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
               9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
              13: 'horse', 14: 'motorbike', 15: 'person', 
              16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
              19: 'train', 20: 'tvmonitor'}
 
classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", 
           "diningtable",  "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                      
# Create the bounding boxes
bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))
     
def DetectPeople(frame):
 
    # Process the video
    People = []
            # Capture the frame's height and width
    (h, w) = frame.shape[:2]
 
            # Create a blob. A blob is a group of connected pixels in a binary 
            # frame that share some common property (e.g. grayscale value)
            # Preprocess the frame to prepare it for deep learning classification
    frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS), 
                                               IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
     
            # Set the input for the neural network
    neural_network.setInput(frame_blob)
 
            # Predict the objects in the image
    neural_network_output = neural_network.forward()
 
            # Put the bounding boxes around the detected objects
    for i in np.arange(0, neural_network_output.shape[2]):
                 
        confidence = neural_network_output[0, 0, i, 2]
                 
                # Confidence must be at least 30%       
        if confidence > 0.30:
                     
            idx = int(neural_network_output[0, 0, i, 1])
 
                    # Only process if the detected object is a person
            if idx == 15:  # Index of 'person' class in the classes list
                bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([w, h, w, h])
 
                (startX, startY, endX, endY) = bounding_box.astype("int")
  
                People.append([startX,startY,endX,endY]) 
                         
                    
    return People

 