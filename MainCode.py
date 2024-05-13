import cv2
import pytesseract
import os
import numpy as np
from collections import Counter
from HumanBox import DetectPeople

special_chars = "=-_¿°<«“ƒ¿+'"

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


def TextOfFrame (frame,Color=-1):
    height, width = frame.shape[:2]

    roi_height = int(0.2 * height)

    roi = frame[height - roi_height:, :]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if (Color != -1):
        _, binary_image = cv2.threshold(gray_roi, Color, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
        text = ChangeText(text)
        return text
    
    _, binary_image_black_text = cv2.threshold(gray_roi, 20, 255, cv2.THRESH_BINARY)
    _, binary_image_white_text = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
    """
    if (Path != "."):
        cv2.imwrite(os.path.join(Path, "{}.png".format(Num)),binary_image_black_text)   
        cv2.imwrite(os.path.join(Path, "dak_{}.png".format(Num)),roi)   
    """

    text_black = pytesseract.image_to_string(binary_image_black_text, lang='vie', config=tessdata_dir_config)
    text_black = ChangeText(text_black)

    text_white = pytesseract.image_to_string(binary_image_white_text, lang='vie', config=tessdata_dir_config)
    text_white = ChangeText(text_white)

    if (len(text_black) > len(text_white)):
        Color = 20
        return text_black,Color
    else:
        Color = 200
        return text_white,Color

def FindLastFrame (frames,id,totalFrame,ColorOfFrame):
    originFrame = frames[id]
    text = TextOfFrame(originFrame,ColorOfFrame)
    l = id
    r = min(totalFrame-1,l + 600)
    res = l
    while (l <= r):
        mid = int((r+l) / 2)
        if (Valid_Group(text,TextOfFrame(frames[mid],ColorOfFrame)) == True):
            res = mid
            l = mid + 1
        else:
            r = mid - 1
    return res

def FindValidText(frames,From,To,ColorOfText):
    len = (To-From)//10
    strings = []
    for id in range(From,To):
        strings.append(TextOfFrame(frames[id],ColorOfText))
        id += len
    string_counter = Counter(strings)
    res = string_counter.most_common(1)
    if res:
        return res[0][0]
    else:
        return None

def FindCenter(rectangle):
    x = rectangle[0]
    y = rectangle[1]
    u = rectangle[2]
    v = rectangle[3]
    return [(x+u)/2,(y+v)/2]

def ChangeRectangle(recA,recB):
    res = 0
    for id in range (4):
        res += abs(recA[id]-recB[id])
    return res

def uniRect (recA,recB):
    x = min(recA[0],recB[0])
    y = min(recA[1],recB[1])
    u = max(recA[2],recB[2])
    v = max(recA[3],recB[3])
    return [x,y,u,v]

def customSort (Rect):
    return FindCenter(Rect)

def CutHumanBox(frames,From,To,n,m):
    People =  DetectPeople(frames[From])
    People = sorted(People,key = customSort)
    """
    print("check")
    for u in People:
        print(u, " people")
    """
    totalChange = [0]*len(People)
    maxRect = []
    if (len(People) == 0):
        return [0,0,0,0]
    for u in  People:
        maxRect.append(u)
    for id in range(From+1,To+1):
        NewPeople = DetectPeople(frames[id])
        NewPeople = sorted(NewPeople,key = customSort)
        """
        print("Continue check")
        for u in NewPeople:
            print(u, " new people")
        """
        for _ in range (min(len(People),len(NewPeople))):
            totalChange[_] += ChangeRectangle(People[_],NewPeople[_])
            maxRect[_] = uniRect(maxRect[_],NewPeople[_])
        id += 10
    res = 0
    for _ in range (len(People)):
        if (totalChange[_] > totalChange[res]):
            res = _
    # res is the index of box
    Rect = maxRect[res]
    Rect[0] = max(0,Rect[0])
    Rect[1] = max(0,Rect[1])
    Rect[2] = min(n,Rect[2])
    Rect[3] = min(m,Rect[3])
    return maxRect[res]

def ChangeFrameToTime(id,fps):
    seconds = int(id/fps)
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return [minutes,remaining_seconds]


def ProcessVideo(video_path,save_path):
    print(video_path)
    print(save_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    output_text_file = open(os.path.join(save_path, "subtitle.txt"), "w", encoding="utf-8")
    numberOfVideo = 1

    frames = []
    totalFrame = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        frames.append(frame)
        totalFrame += 1

    currentFrame = -1
    last_text = ''

    subFrame = []
    lastFrame = 0
    currentFrame = 0
    while (currentFrame < totalFrame):
        ColorOfText = -1
        text,ColorOfText = TextOfFrame(frames[currentFrame],ColorOfText)
        lastFrame = FindLastFrame(frames,currentFrame,totalFrame,ColorOfText)
        if (len(text) < 6):
            currentFrame = lastFrame+1
            continue
        print(currentFrame,lastFrame,ColorOfText,text)
        subFrame.append([currentFrame,lastFrame,TextOfFrame(frames[currentFrame])])


        [startX,startY,endX,endY] = CutHumanBox(frames,currentFrame,lastFrame,frame_width,frame_height)
        # print(startX,startY,endX,endY)
        if (endX-startX == 0 and endY-startY == 0):
                currentFrame = lastFrame + 1
                continue
        
        output_text_file.write("Scene {}:\n".format(numberOfVideo))
        output_text_file.write("From frame: {} to {}\n".format(currentFrame,lastFrame))
        u = ChangeFrameToTime(currentFrame,fps)
        output_text_file.write("Start Time: {}:{}\n".format(u[0], u[1]))
        u = ChangeFrameToTime(lastFrame,fps)
        output_text_file.write("End Time: {}:{}\n".format(u[0], u[1]))
        text = FindValidText(frames,currentFrame,lastFrame,ColorOfText)
        output_text_file.write(text+"\n\n")

        print(currentFrame,lastFrame,text)

        WriteVideo = cv2.VideoWriter(os.path.join(save_path,"scene_{}.mp4".format(numberOfVideo)),cv2.VideoWriter_fourcc(*'mp4v'),fps,(endX-startX,endY-startY))
        # WriteVideo = cv2.VideoWriter(r"C:\Scrape\subvideo_{}.mp4".format(numberOfVideo),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame_width,frame_height))
        # print(endX-startX,endY-startY,frame_width,frame_height," thong")
        for id in range (currentFrame,lastFrame+1):
            crop_frame = frames[id][startY:endY,startX:endX]
            # cv2.rectangle(frames[id], (startX, startY), (endX, endY), (0,255,0), 2)    
            WriteVideo.write(crop_frame)
        currentFrame = lastFrame + 1 
        WriteVideo.release()
        numberOfVideo += 1
    output_text_file.close()
    """
    for sub in subFrame:
        print(sub)
    """

def Main(NumberVideo,path):
    for CountVideo in range (1,NumberVideo+1):
        if (CountVideo != 4):
            continue
        subDirectory = os.path.join(path,"SubVideo_{}".format(CountVideo))
        if (os.path.exists(subDirectory) == False):
            os.mkdir(subDirectory)
        video_path = os.path.join(path,"Video_{}.mp4".format(CountVideo))
        ProcessVideo(video_path,subDirectory)
 
path = r"C:\Scrape"
Main(4,path)

