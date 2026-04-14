import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from motrackers import SORT
import json
import wfdb
import numpy as np
import ast
import pandas as pd

def add_ego(sample):
    width = sample['image']['dimensions']['width']
    height = sample['image']['dimensions']['height']
    sample['annotations'].append({'class': 'ego', 'score': 1.0, 'bbox': {'type': '@stremf/bbox/aabb', 'region': {'center': {'x': width/2, 'y': height/2}, 'dimensions': {'w': 1, 'h': 1}}}})
    return sample

def read_boxes(file_path: str):
    f = open(file_path, 'r')
    data = json.load(f)
    traces = {}
    for frame in data['frames']:
        for sample in frame['samples']:
             if sample['channel'] not in traces:
                  traces[sample['channel']] = [] 
             traces[sample['channel']].append(add_ego(sample))
    return traces

def read_tracked_boxes(file_path: str):
     traces = read_boxes(file_path)
     tracked_traces = {}
     for channel, trace in traces.items():
          tracked_traces[channel] = track(trace)
     return tracked_traces

def track(trace):
     tracker = SORT(max_lost=0, tracker_output_format='mot_challenge', iou_threshold=0.3)
     tracked_trace = []
     for frame in trace:
          obj_num = len(frame['annotations'])
          bboxes = np.zeros((obj_num, 4), 'float')
          confidences = np.zeros((obj_num), 'float')
          class_ids = np.zeros((obj_num), 'str')
          for i in range(obj_num):
               obj = frame['annotations'][i]
               bbox = obj['bbox']['region']
               w = bbox['dimensions']['w']
               h = bbox['dimensions']['h']
               x = bbox['center']['x']-w/2
               y = bbox['center']['y']-h/2
               bboxes[i, :] = np.array([x, y, w, h])
               confidences[i] = obj['score']
               class_ids[i] = obj['class']
          output = tracker.update(bboxes, confidences, class_ids)
          tracked_trace.append({})
          for i in range(len(output)):
               obj = output[i]
               class_id = ""
               for frame_obj in frame['annotations']:
                    dims = frame_obj['bbox']['region']['dimensions']
                    if obj[4] == dims['w'] and obj[5] == dims['h']:
                         class_id = frame_obj['class']
               tracked_trace[-1][obj[1]] = {'x': obj[2], 'y': obj[3], 'w': obj[4], 'h': obj[5], 'score': obj[6], 'class': class_id}
     return tracked_trace

def load_raw_ecg_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = data[:int(len(data)/2)]
    num_samples = len(data)
    sample_shape = data[0][0].shape # Assuming all signals are same shape
    final_data = np.empty((num_samples, sample_shape[0]), dtype=np.float64)
    for i, (signal, _) in enumerate(data):
         final_data[i] = signal[:, 11]
    return final_data 

def load_ecg_data(sampling_rate = 100):
    path = '../PTB-XL/ptb-xl/'
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # Load raw signal data
    columns = pd.read_csv(path+'ptbxl_database.csv')
    X = load_raw_ecg_data(Y, sampling_rate, path)
    return X, columns['report']


def draw_sequence(trace, name):
     n_values = list(range(0, len(trace)))
     plt.plot(n_values, trace, marker='o', linestyle='--', color='b', label=r'$a_n$')
     plt.xlabel('n (Index)')
     plt.ylabel(r'$a_n$ (Value)')
     plt.xticks(n_values if len(n_values)>20 else n_values[::int(len(n_values)/20)+1])  # Ensure all integer indices are shown
     ax = plt.gca() # Get current axes
     ax.xaxis.set_major_locator(MaxNLocator(nbins=20)) 
     plt.grid(True, linestyle='--', alpha=0.7)
     plt.legend()
     plt.tight_layout()
     plt.savefig(name+'.png')
     plt.clf()