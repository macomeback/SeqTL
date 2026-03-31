from prop import *
from lark import Lark
from prop_builder import PropBuilder
from matcher import Matcher
from oracle import ShapeExpressionOracle, VideoOracle, StremOracle
import os
import json
import wfdb
import numpy as np
import pandas as pd
import ast
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from motrackers import SORT

def read_boxes(file_path: str):
    f = open(file_path, 'r')
    data = json.load(f)
    traces = {}
    for frame in data['frames']:
        for sample in frame['samples']:
             if sample['channel'] not in traces:
                  traces[sample['channel']] = [] 
             traces[sample['channel']].append(sample)
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

parser = Lark(r"""
    start: OPENBASE SIGNED_NUMBER DASH SIGNED_NUMBER CLOSEBASE
                   | OPENPAR start CLOSEPAR
              | start OR start
                    | start start
                   | start STAR
              | start REFINE OPEN SIGNED_NUMBER COMMA SIGNED_NUMBER COMMA NOTCLOSE CLOSE

    OR: "|"
    STAR: "*"
    REFINE: "^"
    OPEN: "<"
    CLOSE: ">"
    COMMA: ","
    DASH: "-"
    OPENBASE: "["
    CLOSEBASE: "]"
    OPENPAR: "("
    CLOSEPAR: ")" 
    DOT: "."
    NOTCLOSE: /[^>][^>]*/
    
    %import common.SIGNED_NUMBER
    %import common.WORD
    %import common.WS
    %ignore WS

    """, start='start')

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

def match_moving(prop, query_map):
    dir_path = '../lyft-dataset/processed'
    print("File", "Channel", "Score", "QueryCount", "Time", sep=',')        
    for file_name in os.listdir(dir_path):
            if 'sample' in file_name:
                 continue
            file_path = os.path.join(dir_path, file_name)
            traces = read_tracked_boxes(file_path)
            for channel, trace in traces.items():
                start_time = time.perf_counter()
                now_oracle = VideoOracle(query_map)
                matcher = Matcher(prop, now_oracle)
                score = -1
                for frame in trace:    
                    score = matcher.match(frame)
                end_time = time.perf_counter()
                print(file_name[:4], channel, score, now_oracle.query_count, end_time-start_time, sep=',')
    
def match_strem(prop, query_map):
    dir_path = '../lyft-dataset/processed'
    print("File", "Channel", "Score", "QueryCount", "Time", sep=',')  
    for file_name in os.listdir(dir_path):
            if 'sample' in file_name:
                 continue
            file_path = os.path.join(dir_path, file_name)
            traces = read_boxes(file_path)
            for channel, trace in traces.items():
                start_time = time.perf_counter()
                now_oracle = StremOracle(query_map)
                matcher = Matcher(prop, now_oracle)
                score = -1
                for frame in trace:  
                    score = matcher.match(frame)
                end_time = time.perf_counter()
                print(file_name[:4], channel, score, now_oracle.query_count, end_time-start_time, sep=',')

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

def match_shapexp(prop, query_map):
     X, reports = load_ecg_data(500)
     print("File","Score","QueryCount","Time")
     for i in range(7140, X.shape[0], 20):
          report = reports[i].lower()
          # if "left anterior fascicular block" not in report or "left axis deviation" not in report or "right bundle branch block" not in report:
          #    continue
          score = -1
          trace = X[i, :]
          #draw_sequence(trace[:500], str(i))
          start_time = time.time()
          now_oracle = ShapeExpressionOracle(query_map, 0.96, 100)
          matcher = Matcher(prop, now_oracle)
          for frame in trace:
               score = matcher.match(frame) 
          end_time = time.time()
          if score:
               print(report)
          print(i, score, now_oracle.query_count, end_time-start_time, sep=',')
          #return matcher, trace
     
def filter_matching_props(matcher: Matcher):
     for key, val in matcher._refined_cache.items():
          to_trues = set()
          for to, output in val.items():
               if output == 1:
                    to_trues.add(to)
          if len(to_trues)>0:
               prop, fromm = key
               print(prop.__str__(), fromm)
               print(to_trues)
               print()
     #
shapexp_semres = ["[1-1]*[10-30]^<10,30,l_0.01_inf_inf_inf>[10-30]^<10,30,l_inf_-0.01_inf_inf>[10-30]^<10,30,l_0.01_inf_inf_inf>[20-30]^<20,30,e_-3_3_inf_0_inf_0>[1-1]*",
                  ]
moving_semres = ["[1-1]*[5-100]^<5,100,car_pedestrian_1.1>[1-1]*",
                ]
strem_semres = ["[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&:bicycle:)>[1-1]*",
                "[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&:car:)>([1-1]^<1,1,~[emp](:pedestrian:&:car:)>)*[1-1]^<1,1,:pedestrian:[and]([emp](:pedestrian:&:car:))>([1-1]^<1,1,:pedestrian:[and]([emp](:pedestrian:&:car:))>)*[1-1]^<1,1,~[emp](:pedestrian:&:car:)>([1-1]^<1,1,~[emp](:pedestrian:&:car:)>)*[1-1]*",
                "[1-1]*[1-1]^<1,1,[exists]_p(:pedestrian:)([exists]_q(:truck:)([leq]([y](_p),[y](_q))[and]([leq]([dist](_p,_q),2)[and][leq]([x](_p),[x](:ego:)))))>([1-1]^<1,1,[exists]_p(:pedestrian:)([exists]_q(:truck:)([leq]([y](_p),[y](_q))[and]([leq]([dist](_p,_q),2)[and][leq]([x](_p),[x](:ego:)))))>)*[1-1]*",
                "[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&(:othervehicle:[union]:emergencyvehicle:))>[1-1]*",
                "[1-1]*[1-200]^<1,200,:sign:>[1-1]^<1,1,~[emp](((:othervehicle:[union]:emergencyvehicle:)[union]:pedestrian:)&:sign:)>[1-1]*",
                "[1-1]*[80-80]^<80,80,[exists]_v(:bicycle:)([leq]([x](_v),[x](:ego:))[and]([leq]([y](_v),[y](:ego:))[and][leq]([dist](_v,[:ego:]),1.0)))>[1-1]*"]

argparser = argparse.ArgumentParser()
argparser.add_argument("type", type=str)
argparser.add_argument("idx", type=int)
args = argparser.parse_args()
parsed_tree = None
if args.type == "shapexp":
     parsed_tree = parser.parse(shapexp_semres[args.idx])
elif args.type == "moving":
     parsed_tree = parser.parse(moving_semres[args.idx])
else:
     parsed_tree = parser.parse(strem_semres[args.idx])
builder = PropBuilder(parsed_tree)
prop, query_map = builder.build_prop()
matcher = None
trace = None
if args.type == "shapexp":
     match_shapexp(prop, query_map)
     #filter_matching_props(matcher)
elif args.type == "moving":
     match_moving(prop, query_map)
else:
     match_strem(prop, query_map)
#print(parsed_tree.pretty())
#f = open('418_C_BBB_101_9s_full.csv', 'r')

