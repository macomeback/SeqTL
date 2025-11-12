from prop import *
from lark import Lark
from prop_builder import PropBuilder
from matcher import Matcher
from oracle import ShapeExpressionOracle

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

parsed_tree = parser.parse("[1-1]*[10-361]^<1,361,l-0.1-.-.-0-.>[1-1]*")
builder = PropBuilder(parsed_tree)
prop, query_map = builder.build_prop()
#print(parsed_tree.pretty())
f = open('418_C_BBB_101_9s_full.csv', 'r')
trace = []
for line in f:
    trace.append(float(line))
matcher = Matcher(prop, ShapeExpressionOracle(query_map), trace)
print(matcher.match())
