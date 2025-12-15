from prop import *
from lark import Lark
from prop_builder import PropBuilder
from matcher import Matcher
from oracle import ShapeExpressionOracle, VideoOracle
import os

def read_boxes(file_path:str) -> list[list[list[float]]]:
    trace = []
    f = open(file_path, 'r')
    for line in f:
        boxes = []
        for box in line.split():
            boxes.append(box.split(',')[:-1])
        trace.append(boxes)
    return trace

parser = Lark(r"""
    start: OPENBASE SIGNED_NUMBER DASH SIGNED_NUMBER CLOSEBASE
                   | OPENPAR start CLOSEPAR
              | start OR start
                    | start start
                   | start STAR
              | start REFINE OPEN SIGNED_NUMBER COMMA SIGNED_NUMBER COMMA WORD (DASH (SIGNED_NUMBER | DOT))* CLOSE

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
    
    %import common.SIGNED_NUMBER
    %import common.WORD
    %import common.WS
    %ignore WS

    """, start='start')

#parsed_tree = parser.parse("[1-1]*[10-361]^<1,361,l-0.1-.-.-0-.>[1-1]*")
parsed_tree = parser.parse("[1-1]*[24-40]^<24,40,close>[24-40][24-40]^<24,40,far>")
builder = PropBuilder(parsed_tree)
prop, query_map = builder.build_prop()
#print(parsed_tree.pretty())
#f = open('418_C_BBB_101_9s_full.csv', 'r')
dir_path = '../prep-charades'
for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        trace = read_boxes(file_path)
        matcher = Matcher(prop, VideoOracle(query_map), trace)
        print(matcher.match())
