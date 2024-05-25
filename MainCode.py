import cv2
import pytesseract
import os
import numpy as np
from collections import Counter
from HumanBox import DetectPeople

special_chars = "=-_¿°<«“ƒ¿+'Ö`›'®()>//\0123456789¬"

def Len(s):
    # Define the punctuation marks to be removed
    punctuation_marks = ".,:; "
    
    # Remove punctuation marks from the string
    filtered_string = ''.join(char for char in s if char not in punctuation_marks)
    
    # Return the length of the filtered string
    return len(filtered_string)

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

# List Rect(x,y,u,v)

ListArea = [[0,0.8,1,1],[0.55,0.6,1,0.85],[0,0.4,0.4,1]]
ListColor = [20,160,200]

def FindSubPos(frame):
    height, width = frame.shape[:2]
    res = "" 
    Col = 20
    Area = [0,0.8,1,1]
    # Prior
    for Color in ListColor:
        [x, y, u, v] = [int(ListArea[0][0] * width), int(ListArea[0][1] * height), int(ListArea[0][2] * width), int(ListArea[0][3] * height)]
        roi = frame[y:v,x:u]
        # cv2.imshow("pos",roi)
        # cv2.waitKey(0)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_roi, Color, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
        text = ChangeText(text)
        if (Len(res) < Len(text)):
            res = text
            Col = Color
    if (Len(res) > 5):
        return res,ListArea[0],Col
    #
    for AreaIndex in range (1,3):
        for Color in ListColor:
            [x, y, u, v] = [int(ListArea[AreaIndex][0] * width), int(ListArea[AreaIndex][1] * height), int(ListArea[AreaIndex][2] * width), int(ListArea[AreaIndex][3] * height)]
            roi = frame[y:v,x:u]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_roi, Color, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
            text = ChangeText(text)
            if (Len(res) < Len(text)):
                res = text
                Col = Color
                Area = ListArea[AreaIndex]
    return res,Area,Col


def TextOfFrame (frame,Area,Color):
    height, width = frame.shape[:2]

    [x, y, u, v] = [int(Area[0] * width), int(Area[1] * height), int(Area[2] * width), int(Area[3] * height)]
    roi = frame[y:v,x:u]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_roi, Color, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
    text = ChangeText(text)
    return text
    

def FindLastFrame (frames,id,totalFrame,Area,ColorOfFrame):
    originFrame = frames[id]
    text = TextOfFrame(originFrame,Area,ColorOfFrame)
    l = id
    r = min(totalFrame-1,l + 600)
    res = l
    x = Len(text)
    if (x <= 2):
        while (l <= r):
            mid = int((r+l) / 2)
            dak,Area,Col = FindSubPos(frames[mid])
            if (Valid_Group(text,dak) == True):
                res = mid
                l = mid + 1
            else:
                r = mid - 1
    else:
        while (l <= r):
            mid = int((r+l) / 2)
            dak = TextOfFrame(frames[mid],Area,ColorOfFrame)
            if (Valid_Group(text,dak) == True):
                res = mid
                l = mid + 1
            else:
                r = mid - 1
    return res

def FindValidText(frames,From,To,Area,ColorOfText):
    len = (To-From)//10
    strings = []
    for id in range(From,To):
        strings.append(TextOfFrame(frames[id],Area,ColorOfText))
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
        if (totalFrame > 4000):
            break
    print ("dcmm")
    currentFrame = -1
    last_text = ''

    lastFrame = 0
    currentFrame = 0
    while (currentFrame < totalFrame):
        ColorOfText = -1
        text,AreaOfText,ColorOfText = FindSubPos(frames[currentFrame])
        

        print(text,ColorOfText,AreaOfText)
        '''
        height, width = frames[currentFrame].shape[:2]
        [x, y, u, v] = [int(AreaOfText[0] * width), int(AreaOfText[1] * height), int(AreaOfText[2] * width), int(AreaOfText[3] * height)]
        roi = frames[currentFrame][y:v,x:u]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_roi, ColorOfText, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(binary_image, lang='vie', config=tessdata_dir_config)
        print("ta heo",text)
        cv2.imwrite(os.path.join(save_path, "frames_{}.jpg".format(currentFrame)), binary_image)
        '''
        lastFrame = FindLastFrame(frames,currentFrame,totalFrame,AreaOfText,ColorOfText)
        print("Last Frame: ", lastFrame)
        print(currentFrame,lastFrame)
        if (len(text) < 3):
            currentFrame = lastFrame+1
            continue
        print(currentFrame,lastFrame,ColorOfText,text)

        if (lastFrame - currentFrame <= 80):
            currentFrame = lastFrame+1
            continue
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
        text = FindValidText(frames,currentFrame,lastFrame,AreaOfText,ColorOfText)
        if (text == None):
            text = "None"
        #print(text)
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

