from prop import *
from lark import Lark
from prop_builder import PropBuilder
from matcher import Matcher
from oracle import ShapeExpressionOracle, VideoOracle, StremOracle
import os
import json

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

def match_videos(prop, query_map):
    dir_path = '../lyft-dataset/processed'
    for file_name in os.listdir(dir_path):
            print(file_name)
            file_path = os.path.join(dir_path, file_name)
            traces = read_boxes(file_path)
            matcher = Matcher(prop, VideoOracle(query_map))
            for channel, trace in traces.items():
                for frame in trace:    
                    score = matcher.match(frame)
                    if score>0.0:
                        print(file_name)
                        print("Matched", score)
                        exit()

def match_strem(prop, query_map):
    dir_path = '../lyft-dataset/processed'
    for file_name in os.listdir(dir_path):
            if 'sample' in file_name:
                 continue
            file_path = os.path.join(dir_path, file_name)
            traces = read_boxes(file_path)
            matcher = Matcher(prop, StremOracle(query_map))
            score = False
            for channel, trace in traces.items():
                for frame in trace:    
                    score = matcher.match(frame)
                if score:
                    print("Match", channel, file_name)



#parsed_tree = parser.parse("[1-1]*[10-361]^<1,361,l-0.1-.-.-0-.>[1-1]*")
#parsed_tree = parser.parse("[1-1]*[24-40]^<24,40,close>[24-40][24-40]^<24,40,far>")
parsed_tree = parser.parse("[1-1]*[1-1]^<1,1,~[emp](:pedestrian:&:bicycle:)>[1-1]*")
builder = PropBuilder(parsed_tree)
prop, query_map = builder.build_prop()
match_strem(prop, query_map)
#print(parsed_tree.pretty())
#f = open('418_C_BBB_101_9s_full.csv', 'r')

