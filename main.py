from prop import *
from lark import Lark
from prop_builder import build_prop
from matcher import Matcher
from oracle import RandomOracle

parser = Lark(r"""
    start: OPENBASE SIGNED_NUMBER DASH SIGNED_NUMBER CLOSEBASE
                   | OPENPAR start CLOSEPAR
              | start OR start
                    | start start
                   | start STAR
              | start REFINE OPEN SIGNED_NUMBER COMMA SIGNED_NUMBER COMMA SIGNED_NUMBER CLOSE

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
    
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """, start='start')

parsed_tree = parser.parse("[6-10]^<5,7,9>*|[3-5]^<4,7,9>")
prop = build_prop(parsed_tree)
#print(parsed_tree.pretty())
matcher = Matcher(prop, RandomOracle(), [0 for i in range(0,50)])
print(matcher.match())
