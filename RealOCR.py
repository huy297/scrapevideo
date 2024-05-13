import cv2
import pytesseract
import re
import numpy as np
from collections import Counter
from HumanBox import DetectPeople

special_chars = "=-_¿°<«“ƒ:¿+'"

def ChangeText (text):
    lines = text.splitlines() 
    max_length = 0
    longest_line = ""
    for line in lines:
        if len(line) > max_length:
            max_length = len(line)
            longest_line = line
    res = ''.join(char for char in longest_line if char not in special_chars)
    return res

def Edit_Distance(str1, str2):  
    m, n = len(str1), len(str2)  
    dp = [[0] * (n + 1) for _ in range(m + 1)]  
    for i in range(m + 1):  
        for j in range(n + 1):  
            if i == 0:  
                dp[i][j] = j  
            elif j == 0:  
                dp[i][j] = i  
            elif str1[i - 1] == str2[j - 1]:  
                dp[i][j] = dp[i - 1][j - 1]  
            else:  
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  
  
    return dp[m][n]  
def Valid_Group(str1,str2):
    Dis = Edit_Distance(str1,str2)
    Len = max(len(str1),len(str2))
    return Dis <= 0.35 * Len


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'



def TextOfFrame (frame,Num=0):
    height, width = frame.shape[:2]
    height, width = frame.shape[:2]

    roi_height = int(0.2 * height)
    roi_width = int(0.4 * width)

    roi = frame[height - roi_height:, :]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(r"C:\Scrape\wtf_{}.jpg".format(Num),gray_roi)   
    text = pytesseract.image_to_string(roi, lang='vie', config=tessdata_dir_config)
    text = ChangeText(text)
    boxes = pytesseract.image_to_data(roi)
    print(boxes)
    return text


def ProcessVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    frames = []
    totalFrame = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        print(TextOfFrame(frame))
        totalFrame += 1
        if (totalFrame > 4000):
            break

path = r"C:\Scrape\trim.mp4"

Path = r"C:\Scrape\dakdak.png"

img = cv2.imread(Path)

print(TextOfFrame(img))

ProcessVideo(path)