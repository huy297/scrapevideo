from pytube import YouTube
from pytube.exceptions import RegexMatchError
import os





def change_to_current_directory():
    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

def download_video (url, save_path,CountVideo):
    link = url
    try:
        yt = YouTube(link)
    except RegexMatchError:
        print("Invalid YouTube URL")
        return False
    except Exception as e:
        print("Other Error:", e)
        return False
    streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc()
    for stream in streams:
        print("Resolution:", stream.resolution)

    if not streams:
        print("Can't find streams")
        return False
    video_stream = streams.first()

    try:
        video_stream.download(output_path = save_path, filename = "video_{}.mp4".format(CountVideo))
        print("Download successfully")
    except Exception as e:
        print("Can't Download")
        return False
    return True

def Main(filename):

    SAVE_PATH = r"C:\Scrape"

    change_to_current_directory()


    CountVideo = 5

    with open(filename, "r") as file:
        url_list = file.readlines()

    for url in url_list:
        CountVideo += 1
        download_video(url,SAVE_PATH,CountVideo)

#Pass the directory of file link
Main(r"C:\Scrape\sub.txt")

