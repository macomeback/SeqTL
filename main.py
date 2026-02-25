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

def read_boxes(file_path:str):
    f = open(file_path, 'r')
    data = json.load(f)
    traces = {}
    for frame in data['frames']:
        for sample in frame['samples']:
             if sample['channel'] not in traces:
                  traces[sample['channel']] = [] 
             traces[sample['channel']].append(sample)
    return traces

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
    data = np.array([signal for signal, meta in data])
    return data

def load_ecg_data(sampling_rate = 100):
    path = '../PTB-XL/ptb-xl/'
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # Load raw signal data
    X = load_raw_ecg_data(Y, sampling_rate, path)
    return X, Y

def match_moving(prop, query_map):
    dir_path = '../lyft-dataset/processed'
    print("File", "Channel", "Score", "QueryCount", "Time", sep=',')        
    for file_name in os.listdir(dir_path):
            if 'sample' in file_name:
                 continue
            file_path = os.path.join(dir_path, file_name)
            traces, labels = read_boxes(file_path)
            labels_set = labels_set.union(labels)
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
     plt.plot(n_values, trace, marker='o', linestyle='-', color='b', label=r'$a_n = n^2$')
     plt.xlabel('n (Index)')
     plt.ylabel(r'$a_n$ (Value)')
     plt.xticks(n_values)  # Ensure all integer indices are shown
     plt.grid(True, linestyle='--', alpha=0.7)
     plt.legend()
     plt.savefig(name+'.png')

def match_shapexp(prop, query_map):
     X, _ = load_ecg_data()
     trace = X[0, :, 11]
     #draw_sequence(trace)
     score = -1
     print("File","Score","QueryCount","Time")
     start_time = time.perf_counter()
     now_oracle = ShapeExpressionOracle(query_map, 0.02, 120)
     matcher = Matcher(prop, now_oracle)
     for frame in trace:
        score = matcher.match(frame)     
     end_time = time.perf_counter()
     print("0",  score, now_oracle.query_count, end_time-start_time, sep=',')
     
shapexp_semres = ["[1-1]*[1-1]*^<4,120,e_inf_inf_0_inf_inf_10>[1-1]*^<4,120,e_inf_inf_inf_0_inf_10>[1-1]*^<4,120,l_inf_inf_0_inf>[1-1]*^<4,120,l_inf_inf_0_inf>[1-1]*^<4,120,l_inf_inf_inf_0>[1-1]*^<4,120,e_inf_inf_0_inf_inf_10>[1-1]*^<4,120,e_inf_inf_inf_0_inf_10>[1-1]*",
                  ]
moving_semres = ["[1-1]*[10-20]^<10,20,car_pedestrian_close>[10-20][10-20]^<10,20,car_pedestrian_far>",
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
if args.type == "shapexp":
     match_shapexp(prop, query_map)
elif args.type == "moving":
     match_moving(prop, query_map)
else:
     match_strem(prop, query_map)
#print(parsed_tree.pretty())
#f = open('418_C_BBB_101_9s_full.csv', 'r')

