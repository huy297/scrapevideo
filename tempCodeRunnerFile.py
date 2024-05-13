import cv2
import pytesseract
import re
from HumanBox import DetectPeople

def ChangeText (text):
    lines = text.splitlines() 
    max_length = 0
    longest_line = ""
    for line in lines:
        if len(line) > max_length:
            max_length = len(line)
            longest_line = line
    return longest_line

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


def TextOfFrame (frame):
    height, width = frame.shape[:2]
    height, width = frame.shape[:2]

    roi_height = int(0.2 * height)
    roi_width = int(0.4 * width)

    roi = frame[height - roi_height:, :]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
    text = ChangeText(text)
    return text

def FindLastFrame (frames,id,totalFrame):
    originFrame = frames[id]
    text = TextOfFrame(originFrame)
    l = id
    r = min(totalFrame-1,l + 600)
    res = l
    while (l <= r):
        mid = int((r+l) / 2)
        if (Valid_Group(text,TextOfFrame(frames[mid])) == True):
            res = mid
            l = mid + 1
        else:
            r = mid - 1
    return res

def ProcessVideo(video_path):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    output_text_file = open(r"C:\Scrape\sub.txt", "w", encoding="utf-8")
    numberOfVideo = 0
    WriteVideo = cv2.VideoWriter(r"C:\Scrape\subvideo_{}.mp4".format(numberOfVideo),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width,frame_height))

    frames = []
    totalFrame = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        frames.append(frame)
        totalFrame += 1
        if (totalFrame > 4000):
            break

    currentFrame = -1
    last_text = ''

    subFrame = []
    lastFrame = 0
    """
    for _ in range (totalFrame):
        if (_ < 200):
            continue
        frame = frames[_]
        currentFrame += 1
        if (currentFrame > 1000):
            break
        text = TextOfFrame(frame)
        if (Valid_Group(text,last_text) == 0):
            if (last_text == -1):
                last_text = text
                lastFrame = currentFrame
            else:
                print(last_text, text, Edit_Distance(last_text, text), len(last_text), len(text))
                WriteVideo.release()
                numberOfVideo += 1
                WriteVideo = cv2.VideoWriter(r"C:\Scrape\subvideo_{}.mp4".format(numberOfVideo),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width,frame_height))
                subFrame.append([last_text,lastFrame*fps,currentFrame*fps])
                last_text = text
                lastFrame = currentFrame
                print("new sub")
            
        WriteVideo.write(frame)
        """
    currentFrame = 0
    while (currentFrame < totalFrame):
        numberOfVideo += 1
        WriteVideo = cv2.VideoWriter(r"C:\Scrape\subvideo_{}.mp4".format(numberOfVideo),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width,frame_height))
        lastFrame = FindLastFrame(frames,currentFrame,totalFrame)
        subFrame.append([currentFrame,lastFrame,TextOfFrame(frames[currentFrame])])
        print(currentFrame,lastFrame,TextOfFrame(frames[currentFrame]))
        for id in range (currentFrame,lastFrame+1):
            WriteVideo.write(frames[id])
        currentFrame = lastFrame + 1
        WriteVideo.release()
        
    for sub in subFrame:
        print(sub)
 
print("hello")
video_path = r"C:\Scrape\test.mp4"
ProcessVideo(video_path)

