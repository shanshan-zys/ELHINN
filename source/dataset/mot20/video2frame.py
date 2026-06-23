import os
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def video_to_frame(filename):
    video_path = os.path.join(ROOT,'data','mot20','video',f'{filename}.mp4')
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    # Read all frames; save only the last one as the representative background frame
    for i in range(frame_count):
        _,frame = video.read()
    cv2.imwrite(os.path.join(ROOT,'data','mot20','frame',f'{filename}.png'),frame)
    video.release()

if __name__=="__main__":
    frame_dir = os.path.join(ROOT,'data','mot20','frame')
    os.makedirs(frame_dir,exist_ok=True)
    video_dir = os.path.join(ROOT,'data','mot20','video')
    filelist = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    for idx in range(len(filelist)):
        filename = filelist[idx].replace('.mp4','')
        print(f'Processing: {filename}')
        video_to_frame(filename)
