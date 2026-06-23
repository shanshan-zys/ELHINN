import os
import json
import numpy
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    trajs = {}
    for tid,pts in data.items():
        trajs[tid] = sorted(pts,key=lambda x: x['frame'])
    return trajs

def jade_jfde(gt_json,pred_json,max_frame=None):
    gt = load_json(gt_json)
    pred = load_json(pred_json)
    if max_frame is None:
        max_frame = max(max(p['frame'] for p in pts) for pts in gt.values())
    total_dist = 0.0
    count = 0
    last_frame_dist = []
    for t in range(max_frame+1):
        gt_pts = [(p['x'],p['y']) for pts in gt.values() for p in pts if p['frame']==t]
        pred_pts = [(p['x'],p['y']) for pts in pred.values() for p in pts if p['frame']==t]
        if len(gt_pts)==0:
            continue
        gt_arr = numpy.array(gt_pts)
        pred_arr = numpy.array(pred_pts) if len(pred_pts)>0 else numpy.empty((0,2))
        if len(pred_arr)==0:
            dists = numpy.full(len(gt_arr),fill_value=1000.0)
        else:
            tree = cKDTree(pred_arr)
            dists,_ = tree.query(gt_arr,k=1)
        total_dist += numpy.sum(dists)
        count += len(dists)
        if t==max_frame:
            last_frame_dist = dists
    jade = total_dist/count if count>0 else None
    jfde = numpy.mean(last_frame_dist) if len(last_frame_dist)>0 else None
    return jade,jfde

def process_folder(baseline):
    total_jade = 0.0
    total_jfde = 0.0
    count_files = 0
    gt_folder = os.path.join(ROOT,'output','evaluation','trajectory','gt')
    pred_folder = os.path.join(ROOT,'output','evaluation','trajectory',baseline)
    for fname in os.listdir(gt_folder):
        if fname.endswith('.json'):
            gt_path = os.path.join(gt_folder,fname)
            pred_path = os.path.join(pred_folder,fname)
            if not os.path.exists(pred_path):
                continue
            jade,jfde = jade_jfde(gt_path,pred_path)
            if jade is None or jfde is None:
                continue
            total_jade += jade
            total_jfde += jfde
            count_files += 1
    avg_jade = total_jade/count_files if count_files>0 else None
    avg_jfde = total_jfde/count_files if count_files>0 else None
    print(f'{baseline} JADE: {avg_jade}, JFDE: {avg_jfde}')

if __name__=="__main__":
    baselines = ['gt','ours']
    for baseline in baselines:
        process_folder(baseline)
