# Requires MiDaS: https://github.com/isl-org/MiDaS
# Clone the repository and download model weights (dpt_beit_large_512.pt) before running.

import os
import cv2
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..'))
MIDAS_DIR = os.path.join(ROOT,'third_party','midas')
MIDAS_WEIGHTS = os.path.join(MIDAS_DIR,'weights','dpt_beit_large_512.pt')

def frame_to_depth(category):
    input_path = os.path.join(ROOT,'data','dcfd','frame',category)
    output_path = os.path.join(ROOT,'data','dcfd','depth',category)
    subprocess.run(['python',os.path.join(MIDAS_DIR,'run.py'),
                    '--input_path',input_path,
                    '--output_path',output_path,
                    '--model_weights',MIDAS_WEIGHTS,
                    '--model_type','dpt_beit_large_512'])
    filelist = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
    for idx in range(len(filelist)):
        filename = filelist[idx].replace('.png','')
        os.rename(os.path.join(output_path,f'{filename}-dpt_beit_large_512.pfm'),
                  os.path.join(output_path,f'{filename}.pfm'))
        os.rename(os.path.join(output_path,f'{filename}-dpt_beit_large_512.png'),
                  os.path.join(output_path,f'{filename}.png'))
        depth = cv2.imread(os.path.join(output_path,f'{filename}.png'))
        frame = cv2.imread(os.path.join(input_path,f'{filename}.png'))
        cv2.imwrite(os.path.join(output_path,f'{filename}.png'),cv2.addWeighted(frame,0.5,depth,0.5,0))

if __name__=="__main__":
    frame_dir = os.path.join(ROOT,'data','dcfd','frame')
    folderlist = sorted(os.listdir(frame_dir))
    for category in folderlist:
        output_path = os.path.join(ROOT,'data','dcfd','depth',category)
        os.makedirs(output_path,exist_ok=True)
        frame_to_depth(category)
