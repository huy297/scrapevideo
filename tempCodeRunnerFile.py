import cv2
import pytesseract
import re

pattern = r'\b[AĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴ]+\b'

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
    return Dis <= 0.15 * Len

def ProcessVideo(video_path):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    subtitle_start_time = None
    subtitle_end_time = None

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

    output_text_file = open(r"C:\Scrape\sub.txt", "w", encoding="utf-8")
    numberOfVideo = 0
    WriteVideo = cv2.videoWriter(r"C:\Scrape\subvideo_{}.mp4".format(numberOfVideo),cv2.VideoWriter_fourcc(*'mp4v'),fps,frame_width,frame_height)

    currentFrame = 0
    last_text = ''

    subFrame = []
    lastFrame = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        height, width = frame.shape[:2]

        roi_height = int(0.2 * height)
        roi_width = int(0.4 * width)

        roi = frame[height - roi_height:, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
        text = ChangeText(text)
        if (len(text) < 4):
            continue
        if (Valid_Group(text,last_text) == 0):
            if (last_text == -1):
                last_text = text
            else:
                WriteVideo.release()
                numberOfVideo += 1
                WriteVideo = cv2.videoWriter(r"C:\Scrape\subvideo_{}.mp4".format(numberOfVideo),cv2.VideoWriter_fourcc(*'mp4v'),fps,frame_width,frame_height)
                subFrame.append([last_text,lastFrame*fps,currentFrame*fps])
                print("new sub")
            
        WriteVideo.write(frame)
        currentFrame += 1
        if (currentFrame > 2000):
            break
    for sub in subFrame:
        print(sub)

 
print("hello")
video_path = r"C:\Scrape\test.mp4"
ProcessVideo(video_path)

