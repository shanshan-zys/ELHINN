import os
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))

def video_to_frame(category,filename):
    video_path = os.path.join(ROOT,'data','dcfd','video',category,f'{filename}.mp4')
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    # Read all frames; save only the last one as the representative background frame
    for i in range(frame_count):
        _,frame = video.read()
    cv2.imwrite(os.path.join(ROOT,'data','dcfd','frame',category,f'{filename}.png'),frame)
    video.release()

if __name__=="__main__":
    video_dir = os.path.join(ROOT,'data','dcfd','video')
    folderlist = sorted(os.listdir(video_dir))
    for category in folderlist:
        frame_dir = os.path.join(ROOT,'data','dcfd','frame',category)
        os.makedirs(frame_dir,exist_ok=True)
        filelist = sorted([f for f in os.listdir(os.path.join(video_dir,category)) if f.endswith('.mp4')])
        for idx in range(len(filelist)):
            filename = filelist[idx].replace('.mp4','')
            print(f'Processing: {filename}')
            video_to_frame(category,filename)
