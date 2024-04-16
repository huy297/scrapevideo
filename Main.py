from pytube import YouTube
from pytube.exceptions import RegexMatchError
import os

def change_to_current_directory():
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

def download_video (url, save_path):
    link = url;
    try:
        yt = YouTube(link)
    except RegexMatchError:
        print("Invalid YouTube URL")
        return False;
    except Exception as e:
        print("Other Error:", e)
        return False;
    streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc()
    for stream in streams:
        print("Resolution:", stream.resolution)

    if not streams:
        print("Can't find streams")
        return False;
    video_stream = streams.first()

    try:
        video_stream.download(output_path = save_path)
        print("Download successfully")
    except Exception as e:
        print("Can't Download")
        return False;
    return True;

SAVE_PATH = r"C:\Scrape"

change_to_current_directory()


current_directory = os.getcwd()
print("Thư mục làm việc hiện tại:", current_directory)

with open("URLVideos.txt", "r") as file:
    url_list = file.readlines()

for url in url_list:
    download_video(url,SAVE_PATH)