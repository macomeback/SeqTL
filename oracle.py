import random
import numpy as np
#import torch
#from numpy.typing import NDArray, Shape
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from lark import Lark
from strem.builder import build_formula
from strem.evaluator import Evaluator, obj_to_box
import shapely

strem_parser = Lark(r"""
    formula: ATOM
                   | EXISTS VAR OPENPAR ATOM CLOSEPAR formula
              | EMP OPENPAR term CLOSEPAR
                    | SUBSET OPENPAR term COMMA term CLOSEPAR
                   | NOT formula
              | formula AND formula
			| formula OR formula
			| LEQ OPENPAR exp COMMA exp CLOSEPAR
			| OPENPAR formula CLOSEPAR
					
	exp: SIGNED_NUMBER
					| X OPENPAR term CLOSEPAR
					| Y OPENPAR term CLOSEPAR
					| DIST OPENPAR term COMMA term CLOSEPAR
					| MINUS exp
					| exp SUM exp
					| exp MUL exp
					| exp EXP exp
					| OPENPAR exp CLOSEPAR

	term: ATOM
		| VAR
		| COMP OPENPAR term CLOSEPAR
		| term UNION term
		| term INTERSECTION term
		| OPENPAR term CLOSEPAR
					

    UNION: "|"
	INTERSECTION: "&"
	COMP: "[comp]"
	MINUS: "-"
	SUM: "+"
	MUL: "*"
	EXP: "**"	
	VAR: "_" WORD
	SUBSET: "[subset]"
	LEQ: "[leq]"				
	AND: "/\"
	OR: "\/"
	NOT: "~"
    ATOM: ":" WORD ":"
    X: "[x]"
	Y: "[y]"
	DIST: "[dist]"
	EXISTS: "[exists]"
	EMPTY: "[emp]"
    COMMA: ","
    OPENPAR: "("
    CLOSEPAR: ")" 
    
    %import common.SIGNED_NUMBER
    %import common.WORD
    %import common.WS
    %ignore WS

    """, start='formula')


class RandomOracle:
	def __init__(self, query_map: dict[str, int]):
		pass

	def _fill_queries(query_map):
		pass

	def compute(self, query_id: int, trace: list[object]) -> float:
		response = random.random()
		return response
	
class ShapeExpressionOracle:
	def __init__(self, query_map: dict[str, int]):
		self._queries: list = [None]*len(query_map)
		self._fill_queries(query_map)
		self._current_x = []

	def _fill_queries(self, query_map):
		func_map = {"s": self.sin, "e": self.exp, "l": self.lin}
		for query_str, query_id in query_map.items():
			lb_list = []
			ub_list = []
			parts = query_str.split("-")
			func = func_map[parts[0]]
			mse = float(parts[1])
			for i in range(2, len(parts)):
				if parts[i] == '.':
					if i %2 == 0:
						lb_list.append(-np.inf)
					else:
						ub_list.append(np.inf)
				elif i % 2 == 0:
					lb_list.append(float(parts[i]))
				else:
					ub_list.append(float(parts[i]))
			self._queries[query_id] = (func, mse, lb_list, ub_list)

	def compute(self, query_id: int, trace: list[object]) -> float:
		for i in range(len(self._current_x), len(trace)):
			self._current_x.append(i)
		return self.atomic_match(trace, *self._queries[query_id])

	def atomic_match(self, trace, shape, condition, lb_list, ub_list) -> float:
		try:
			popt, pcov = curve_fit(f = shape, xdata = self._current_x[:len(trace)], ydata=trace, bounds=(lb_list,ub_list), method='dogbox')
			y_pred = shape(np.array(self._current_x[:len(trace)]), *popt)
			r2 = r2_score(trace, y_pred)
			return max(0.0,1-(1-r2)/condition)
		except RuntimeError as e:
			i = str(e).find("maximum number of function evaluations")
			print("error",i)
			if "maximum number of function evaluations" in str(e):
				return 0.0
		return 0.0

	def lin(self, t, a, b):
		return a*t+b
	
	def exp(self, t, a, b, c) -> float:
		return a+b*np.exp(c*t)
	
	def sin(self, t, a, b, c, d) -> float:
		return a+b*np.sin(c*t+d)
	
def iou(shape1 , shape2) -> float:
	intersection = shapely.intersection(shape1, shape2)
	union = shapely.union(shape1, shape2)
	if union.area > 0.0:
		return intersection.area/union.area
	return 0.0
	
class VideoOracle:
	def __init__(self, query_map: dict[str, int]):
		self._queries: list = [None]*len(query_map)
		self._fill_queries(query_map)
		self._cache = {}
		self.trace = []
		self.obj_traces = {}
		self.max_id = -1

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			self._queries[query_id] = query_str
	    
	def match_boxes(self, prev_objs_map, objs) -> dict[int, list[float]]:
		matching = {}
		for obj in objs:
			box = obj_to_box(obj)
			threshold = 0.5
			matched = -1
			for prev_idx, prev_obj in prev_objs_map.items():
				prev_box = obj_to_box(prev_obj)
				iou_val = iou(box, prev_box)
				if iou_val >= threshold:
					matched = prev_idx
					threshold = iou_val
			if matched >=0:
				matching[matched] = obj
			else:
				self.max_id += 1
				matching[self.max_id] = obj
		return matching
	    
	def track(self, obj_type, objs):
		if obj_type not in self.obj_traces:
			self.obj_traces[obj_type] = {}
		track = self.obj_traces[obj_type]
		if len(self.trace)-2 not in track:
			track[len(self.trace)-1] = {}
			for obj in objs:
				self.max_id += 1
				track[len(self.trace)-1][self.max_id] = obj
			return
		track[len(self.trace)-1] = self.match_boxes(track[len(self.trace)-2], objs)

	def get_type_bucket(self):
		type_bucket = {}
		for obj in self.trace[-1]:
			obj_type = obj['class']
			if obj_type not in type_bucket:
				type_bucket[obj_type] = []
			type_bucket[obj_type].append(obj)
		return type_bucket
	
	def get_distance(self, obj1, obj2) -> float:
		center1 = obj1['bbox']['region']['center']
		center2 = obj2['bbox']['region']['center']
		x1 = center1['x']
		y1 = center1['y']
		x2 = center2['x']
		y2 = center2['y']
		return (x1-x2)**2+(y1-y2)**2

	def getting_closer(self, track, fromm: int, to: int , idx1: int, idx2: int) -> float:
		obj1 = track[fromm][idx1]
		obj2 = track[fromm][idx2]
		init_distance = self.get_distance(obj1, obj2)
		confidence = min(obj1['score'], obj2['score'])
		for i in range(fromm+1, to):
			if i not in track:
				return 0.0
			if idx1 not in track[i] or idx2 not in track[i]:
				return 0.0
			obj1 = track[fromm][idx1]
			obj2 = track[fromm][idx2]
			distance = self.get_distance(obj1, obj2)
			if init_distance*1.05 < distance or (i == len(track)-1 and init_distance*0.95 < distance):
				return 0.0
			confidence = min(confidence, obj1['score'], obj2['score'])
		return confidence
	
	def getting_further(self, track, fromm: int, to: int , idx1: int, idx2: int) -> float:
		obj1 = track[fromm][idx1]
		obj2 = track[fromm][idx2]
		init_distance = self.get_distance(obj1, obj2)
		confidence = min(obj1['score'], obj2['score'])
		for i in range(fromm+1, to):
			if i not in track:
				return 0.0
			if idx1 not in track[i] or idx2 not in track[i]:
				return 0.0
			obj1 = track[fromm][idx1]
			obj2 = track[fromm][idx2]
			distance = self.get_distance(obj1, obj2)
			if init_distance*0.95 > distance or (i == len(track)-1 and init_distance*1.05 > distance):
				return 0.0
			confidence = min(confidence, obj1['score'], obj2['score'])
		return confidence
			
	def distance_direction(self, fromm: int, to: int, is_closer: bool) -> float:
		confidence = 0.0
		for _, track in self.obj_traces.items():
			if fromm not in track:
				continue
			idxs = list(track[fromm].keys())
			for i in range(len(idxs)):
				for j in range(i+1, len(idxs)):
					confidence = max(confidence, self.getting_closer(track, fromm, to, idxs[i], idxs[j]) if is_closer else self.getting_further(track, fromm, to, idxs[i], idxs[j]))
		return confidence
	
	def add_frame(self, frame):
		self.trace.append(frame)
		type_bucket = self.get_type_bucket()
		for obj_type, objs in type_bucket.items():
			self.track(obj_type, objs)
			
	def compute(self, query_id: int, fromm: int, to: int) -> float:
		if (query_id, fromm, to) in self._cache:
			return self._cache[query_id, fromm, to]
		output = self.distance_direction(fromm, to, self._queries[query_id] == "close")
		self._cache[query_id, fromm, to] = output
		return output
			
class StremOracle:
	def __init__(self, query_map: dict[str, int]):
		self._queries: list = [None]*len(query_map)
		self._fill_queries(query_map)
		self.trace = []
		self._cache = {}

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			self._queries[query_id] = build_formula(query_str)

	def add_frame(self, frame):
		self.trace.append(frame)

	def match(self, formula, frame):
		evaluator = Evaluator(frame)
		evaluator.eval(formula)

	def compute(self, query_id: int, fromm: int, to: int) -> float:
		if to-fromm !=1:
			raise Exception("More than one frame queried")
		if (query_id, fromm) in self._cache:
			return self._cache[query_id, fromm]
		formula = self._queries[query_id]
		frame = self.trace[fromm]
		output = self.match(formula, frame)
		self._cache[query_id, fromm] = output
		return output