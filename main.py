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
    print("File", "Channel", "Score", "Query Count", "Time")
    for file_name in os.listdir(dir_path):
            if 'sample' in file_name:
                 continue
            file_path = os.path.join(dir_path, file_name)
            traces = read_boxes(file_path)
            for channel, trace in traces.items():
                start_time = time.perf_counter()
                now_oracle = VideoOracle(query_map)
                matcher = Matcher(prop, now_oracle)
                score = -1
                for frame in trace:    
                    score = matcher.match(frame)
                end_time = time.perf_counter()
                print(file_name, channel, score, now_oracle.query_count, end_time-start_time)

def match_strem(prop, query_map):
    dir_path = '../lyft-dataset/processed'
    for file_name in os.listdir(dir_path):
            if 'sample' in file_name:
                 continue
            print(file_name)
            file_path = os.path.join(dir_path, file_name)
            traces = read_boxes(file_path)
            matcher = Matcher(prop, StremOracle(query_map))
            score = False
            for channel, trace in traces.items():
                for frame in trace:    
                    score = matcher.match(frame)
                if score:
                    print("Match", channel)

def match_shapexp(prop, query_map):
     X, Y = load_ecg_data()
     trace = X[0, :, 11]
     i = 0
     matcher = Matcher(prop, ShapeExpressionOracle(query_map, 0.02))
     for frame in trace:
        score = matcher.match(frame)
        i += 1
     
shapexp_semres = ["[1-1]*[1-1]*^<3,1000,e_inf_inf_0_inf_inf_inf>[1-1]*^<3,1000,e_inf_inf_inf_0_inf_inf>[1-1]*^<3,1000,l_0_inf_inf_inf>[1-1]*^<3,1000,l_0_inf_inf_inf>[1-1]*^<3,1000,l_inf_0_inf_inf>[1-1]*^<3,1000,e_inf_inf_0_inf_inf_inf>[1-1]*^<3,1000,e_inf_inf_inf_0_inf_inf>[1-1]*",
                  ]
moving_semres = ["[1-1]*[24-40]^<24,40,car_pedestrian_close>[24-40][24-40]^<24,40,car_pedestrian_far>",
                ]
strem_semres = ["[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&:bicycle:)>[1-1]*"]

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

