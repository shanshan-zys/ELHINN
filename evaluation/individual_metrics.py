import os
import json
import numpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

def load_trajectories(json_path):
    with open(json_path) as f:
        data = json.load(f)
    trajs = {}
    for id_,pts in data.items():
        trajs[id_] = sorted(pts,key=lambda x: x['frame'])
    return trajs

def compute_metrics(trajs,frame_rate=1.0,radius=3):
    results = []
    max_frame = max(max(p['frame'] for p in pts) for pts in trajs.values())
    for _,pts in trajs.items():
        frames = numpy.array([p['frame'] for p in pts])
        xs = numpy.array([p['x'] for p in pts])
        ys = numpy.array([p['y'] for p in pts])
        pos = numpy.stack([xs,ys],axis=1)
        traj_duration = frames[-1]-frames[0]
        if traj_duration==0:
            traj_duration = 1
        deltas = numpy.diff(pos,axis=0)
        L = numpy.sum(numpy.linalg.norm(deltas,axis=1))/traj_duration
        start,end = pos[0],pos[-1]
        if numpy.all(start==end):
            D = 0.0
        else:
            line_vec = end-start
            norm_line = numpy.linalg.norm(line_vec)
            v = pos-start
            proj = (v.dot(line_vec)/norm_line**2)[:,None]*line_vec
            perp = numpy.linalg.norm(v-proj,axis=1)
            D = numpy.sum(perp)
        speeds = numpy.linalg.norm(deltas,axis=1)*frame_rate
        V = numpy.sum(numpy.abs(numpy.diff(speeds)))/traj_duration
        angles = numpy.arctan2(deltas[:,1],deltas[:,0])
        dtheta = numpy.diff(angles)
        dtheta = numpy.mod(dtheta+numpy.pi,2*numpy.pi)-numpy.pi
        A = numpy.sum(numpy.abs(dtheta))*180.0/numpy.pi/traj_duration
        es,ew,m = 2.23,1.26,70
        E = numpy.sum(m*(es+ew*speeds**2)*(1/frame_rate))/traj_duration
        delta_v = numpy.diff(numpy.vstack(([0,0],deltas)),axis=0)*frame_rate
        steerE = numpy.sum(m*(es+ew*numpy.sum(delta_v**2,axis=1))*(1/frame_rate))/traj_duration
        results.append({'D':D,'V':V,'A':A,'steerE':steerE,'L':L,'E':E})
    collision_counts = []
    for t in range(max_frame+1):
        points = []
        for pts in trajs.values():
            for p in pts:
                if p['frame']==t:
                    points.append((p['x'],p['y']))
        collision_agents = set()
        for i in range(len(points)):
            for j in range(i+1,len(points)):
                if numpy.linalg.norm(numpy.array(points[i])-numpy.array(points[j]))<2*radius:
                    collision_agents.add(i)
                    collision_agents.add(j)
        if points:
            collision_rate = len(collision_agents)/len(points)
            collision_counts.append(collision_rate)
    C = numpy.mean(collision_counts) if collision_counts else 0
    return results,C

def process_folder(baseline,radius=10):
    all_metrics = []
    folder_path = os.path.join(ROOT,'output','evaluation','trajectory',baseline)
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            trajs = load_trajectories(os.path.join(folder_path,file))
            metrics,collision_rate = compute_metrics(trajs,radius=radius)
            for m in metrics:
                m['C'] = collision_rate
                all_metrics.append(m)
    avg_metrics = {k:numpy.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    print(f'{baseline} Metrics: {avg_metrics}')

if __name__=="__main__":
    baselines = ['gt','ours']
    for baseline in baselines:
        process_folder(baseline)
