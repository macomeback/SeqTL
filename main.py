from prop import *
from lark import Lark
from prop_builder import PropBuilder
from matcher import Matcher
from oracle import ShapeExpressionOracle, VideoOracle, StremOracle
import os
import pandas as pd
import argparse
import time
from data_handler import read_boxes, read_tracked_boxes, load_ecg_data
import numpy as np

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

# path = ../lyft-dataset/processed
def match_moving(prop, query_map, dir_path):
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

# path = ../lyft-dataset/processed  
def match_strem(prop, query_map, dir_path):
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

def sample_from(X, start_range: int, end_range: int, num_samples: int):
     X = X[start_range: end_range, :]
     X_indices = np.array([i for i in range(start_range, end_range)])
     shuffled_indices = np.random.permutation(len(X))
     X = X[shuffled_indices]
     X_indices = X_indices[shuffled_indices]
     return X[:num_samples], X_indices[:num_samples]

# Path: ../PTB-XL/ptb-xl/
def match_ecg(prop, query_map, path):
     inputs = input("Enter start range, end range and number of samples separated by space.\n").split(" ")
     start_range: int = int(inputs[0])
     end_range: int = int(inputs[1])
     num_samples: int = int(inputs[2])
     X = load_ecg_data(path, 500)
     X, X_indices = sample_from(X, start_range, end_range, num_samples)
     print("File","Score","QueryCount","Time",sep=',')
     for i in range(X.shape[0]):
          score = -1
          trace = X[i, :]
          for should_optimize in [True, False]:
               start_time = time.time()
               now_oracle = ShapeExpressionOracle(query_map, 0.96, 100, should_optimize)
               matcher = Matcher(prop, now_oracle)
               for frame in trace:
                    score = matcher.match(frame) 
               end_time = time.time()
               optimize_flag = "-o" if should_optimize else "-n"
               print(f"{X_indices[i]}{optimize_flag}", score, now_oracle.query_count, end_time-start_time, sep=',')

# Path = ../aircraft_aggregate/? 
def match_aircraft(prop, query_map, dir_path):
     print("File", "Score", "QueryCount", "Time", sep=',')  
     for file_name in os.listdir(dir_path):
          file_path = os.path.join(dir_path, file_name)
          df = pd.read_csv(file_path)
          trace = df[df.keys()[1]].to_numpy()
          score = -1
          start_time = time.time()
          now_oracle = ShapeExpressionOracle(query_map, 0.96, 100)
          matcher = Matcher(prop, now_oracle)
          for frame in trace:
               score = matcher.match(frame) 
          end_time = time.time()
          print(file_name, score, now_oracle.query_count, end_time-start_time, sep=',')
     #
ecg_semres = ["[1-1]*[10-30]^<10,30,l_0.01_inf_inf_inf>[10-30]^<10,30,l_inf_-0.01_inf_inf>[10-30]^<10,30,l_0.01_inf_inf_inf>[20-30]^<20,30,e_-3_3_inf_0_inf_0>[1-1]*"]
aircraft_semres = ["[1-1]*[30-75]^<30,75,l_0.5_inf_inf_inf>[150-250]^<150,250,s_inf_inf_inf_inf_inf_inf_inf_inf>[1-1]*"]
moving_semres = ["[1-1]*[5-100]^<5,100,car_pedestrian_1.1>[1-1]*",
                ]
strem_semres = ["[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&:bicycle:)>[1-1]*",
                "[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&:car:)>([1-1]^<1,1,~[emp](:pedestrian:&:car:)>)*[1-1]^<1,1,:pedestrian:[and]([emp](:pedestrian:&:car:))>([1-1]^<1,1,:pedestrian:[and]([emp](:pedestrian:&:car:))>)*[1-1]^<1,1,~[emp](:pedestrian:&:car:)>([1-1]^<1,1,~[emp](:pedestrian:&:car:)>)*[1-1]*",
                "[1-1]*[1-1]^<1,1,[exists]_p(:pedestrian:)([exists]_q(:truck:)([leq]([y](_p),[y](_q))[and]([leq]([dist](_p,_q),20)[and][leq]([x](_p),[x](:ego:)))))>([1-1]^<1,1,[exists]_p(:pedestrian:)([exists]_q(:truck:)([leq]([y](_p),[y](_q))[and]([leq]([dist](_p,_q),20)[and][leq]([x](_p),[x](:ego:)))))>)*[1-1]*",
                "[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&(:othervehicle:[union]:emergencyvehicle:))>[1-1]*",
                "[1-1]*[1-200]^<1,200,:sign:>[1-1]^<1,1,~[emp](((:othervehicle:[union]:emergencyvehicle:)[union]:pedestrian:)&:sign:)>[1-1]*",
                "[1-1]*[80-80]^<80,80,[exists]_v(:bicycle:)([leq]([x](_v),[x](:ego:))[and]([leq]([y](_v),[y](:ego:))[and][leq]([dist](_v,:ego:),20)))>[1-1]*"]

argparser = argparse.ArgumentParser()
argparser.add_argument("type", type=str)
argparser.add_argument("idx", type=int)
argparser.add_argument("path", type=str)
args = argparser.parse_args()
arg_path = args.path
if arg_path[len(arg_path)-1] != '/':
     arg_path = arg_path+'/'
parsed_tree = None
if args.type == "ecg":
     parsed_tree = parser.parse(ecg_semres[args.idx])
elif args.type == "aircraft":
     parsed_tree = parser.parse(aircraft_semres[args.idx])
elif args.type == "moving":
     parsed_tree = parser.parse(moving_semres[args.idx])
elif args.type == "strem":
     parsed_tree = parser.parse(strem_semres[args.idx])
builder = PropBuilder(parsed_tree)
prop, query_map = builder.build_prop()
matcher = None
trace = None
if args.type == "ecg":
     match_ecg(prop, query_map, args.path)
elif args.type == "aircraft":
     match_aircraft(prop, query_map, args.path)
elif args.type == "moving":
     match_moving(prop, query_map, args.path)
elif args.type == "strem":
     match_strem(prop, query_map, args.path)
#print(parsed_tree.pretty())
#f = open('418_C_BBB_101_9s_full.csv', 'r')

