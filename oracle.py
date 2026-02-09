import random
import numpy as np
#import torch
#from numpy.typing import NDArray, Shape
from lark import Lark
from strem.builder import build_formula
from strem.evaluator import Evaluator, obj_to_box
import shapely
from lmfit import Model, Parameters

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
	AND: "[and]"
	OR: "[or]"
	NOT: "~"
    ATOM: ":" WORD ":"
    X: "[x]"
	Y: "[y]"
	DIST: "[dist]"
	EXISTS: "[exists]"
	EMP: "[emp]"
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
	def __init__(self, query_map: dict[str, int], noise_tolerance: float):
		self._queries: list = [None]*len(query_map)
		self._tolerance = noise_tolerance
		self._fill_queries(query_map)
		self._cache = {}
		self._current_x = np.zeros(0, dtype=int)
		self.trace = np.zeros(0, dtype=float)

	def _fill_queries(self, query_map):
		func_map = {"s": self.sin, "e": self.exp, "l": self.lin}
		for query_str, query_id in query_map.items():
			lb_list = []
			ub_list = []
			parts = query_str.split("_")
			func = func_map[parts[0]]
			for i in range(1, len(parts)):
				if parts[i] == 'inf':
					if i %2 == 1:
						lb_list.append(-np.inf)
					else:
						ub_list.append(np.inf)
				elif i % 2 == 1:
					lb_list.append(float(parts[i]))
				else:
					ub_list.append(float(parts[i]))
			if query_str == "e":
				ub_list[2] = min(ub_list[2], 700)
			self._queries[query_id] = (func, lb_list, ub_list)

	def compute(self, query_id: int, fromm: int, to: int) -> bool:
		if (fromm, to) in self._cache:
			return self._cache[fromm, to]
		result = self.atomic_match(self.trace[fromm: to], *self._queries[query_id])
		self._cache[query_id, fromm, to] = result
		return result
	
	def atomic_match(self, trace, shape, lb_list, ub_list) -> bool:
		min_trace = trace.min()
		scale = trace.max()-min_trace
		if scale > 0.0:
			normalized_trace = (trace-min_trace)/scale
		else:
			normalized_trace = trace
		param_names = ['a','b','c','d']
		model = Model(shape, independent_vars=['t'])
		params = model.make_params()
		for i in range(len(lb_list)):
			if lb_list[i] != np.inf and lb_list[i] != -np.inf:
				params[param_names[i]].set(min=lb_list[i])
			if ub_list[i] != np.inf and ub_list[i] != -np.inf:
				params[param_names[i]].set(min=ub_list[i])
		result = model.fit(normalized_trace, t=self._current_x[:len(trace)])
		return result.summary()['rsquared'] >= .98
	
	def add_frame(self, frame):
		self.trace = np.append(self.trace, [frame])
		self._current_x = np.append(self._current_x, [len(self.trace)-1])

	def lin(self, t, a=0, b=0):
		return a*t+b
	
	def exp(self, t, a=0, b=0, c=0) -> float:
		return a+b*np.exp(c*t)
	
	def sin(self, t, a=0, b=0, c=0, d=0) -> float:
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
		self.type_buckets = []
		self.query_count = 0

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			self._queries[query_id] = query_str.split("_")
	
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

	def getting_closer(self, fromm: int, to: int, type1, type2) -> float:
		last_dist = -1
		min_score = -1
		init_dist = -1
		for i in range(fromm, to):
			if type1 not in self.type_buckets[i] or type2 not in self.type_buckets[i]:
				return 0.0
			bucket1 = self.type_buckets[i][type1]
			bucket2 = self.type_buckets[i][type2]
			if type1==type2 and (len(bucket1)==1 or len(bucket2)==1):
				return 0.0
			dist, score = self.get_min_dist(bucket1, bucket2)
			if i == fromm:
				last_dist = dist
				init_dist = dist
				min_score = score
			elif last_dist > 1.05*dist:
				return 0.0
			else:
				last_dist = dist
				min_score = min(score, min_score)
		if init_dist*0.95>last_dist:
			return min_score
		return 0.0
	
	
	def get_min_dist(self, bucket1, bucket2):
		min_dist = -1
		score = -1
		for i in range(len(bucket1)):
			for j in range(len(bucket2)):
				if bucket1[i] != bucket2[j]:
					dist = self.get_distance(bucket1[i], bucket2[j])
					ij_score = min(bucket1[i]['score'], bucket2[j]['score'])
					if min_dist<0:
						min_dist = dist
						score = ij_score
					else:
						min_dist = min(min_dist, dist)
						score = min(score, ij_score)
		return min_dist, score

	
	def getting_further(self, fromm: int, to: int, type1, type2) -> float:
		last_dist = -1
		min_score = -1
		init_dist = -1
		for i in range(fromm, to+1):
			if type1 not in self.type_buckets[i] or type2 not in self.type_buckets[i]:
				return 0.0
			bucket1 = self.type_buckets[i][type1]
			bucket2 = self.type_buckets[i][type2]
			if type1==type2 and (len(bucket1)==1 or len(bucket2)==1):
				return 0.0
			dist, score = self.get_min_dist(bucket1, bucket2)
			if i == fromm:
				last_dist = dist
				init_dist = dist
				min_score = score
			elif last_dist < 0.95*dist:
				return 0.0
			else:
				last_dist = dist
				min_score = min(score, min_score)
		if init_dist*1.05<last_dist:
			return min_score
		return 0.0
			
	def add_frame(self, frame):
		self.trace.append(frame['annotations'])
		self.type_buckets.append(self.get_type_bucket())
			
	def compute(self, query_id: int, fromm: int, to: int) -> float:
		if (query_id, fromm, to) in self._cache:
			return self._cache[query_id, fromm, to]
		self.query_count += 1
		query_parts = self._queries[query_id]
		if "close" == query_parts[2]:
			output = self.getting_closer(fromm, to, query_parts[0], query_parts[1])
		else:
			output = self.getting_further(fromm, to, query_parts[0], query_parts[1])
		self._cache[query_id, fromm, to] = output
		return output

# Unlike VideoOracle, frame here is a sample not just sample['annotations'] since
# we need to know the sample image width and hight here.	
class StremOracle:
	def __init__(self, query_map: dict[str, int]):
		self._queries: list = [None]*len(query_map)
		self._fill_queries(query_map)
		self.trace = []
		self._cache = {}

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			self._queries[query_id] = build_formula(strem_parser.parse(query_str))

	def add_frame(self, frame):
		self.trace.append(frame)

	def match(self, formula, frame) -> bool:
		evaluator = Evaluator(frame)
		return evaluator.eval(formula)

	def compute(self, query_id: int, fromm: int, to: int) -> float:
		if to-fromm !=1:
			raise Exception("More than one frame queried")
		if (query_id, fromm) in self._cache:
			return self._cache[query_id, fromm]
		formula = self._queries[query_id]
		frame = self.trace[fromm]
		output = 1.0 if self.match(formula, frame) else 0.0
		self._cache[query_id, fromm] = output
		return output
	
#try:
		# 	popt, pcov = curve_fit(f = shape, xdata = self._current_x[:len(trace)], ydata=trace, bounds=(lb_list,ub_list), method='dogbox')
		# 	y_pred = shape(self._current_x[:len(trace)], *popt)
		# 	r2 = r2_score(trace, y_pred)
		# 	return r2 > 1-self._tolerance
		# except RuntimeError as e:
		# 	i = str(e).find("maximum number of function evaluations")
		# 	print("error",i)
		# 	if "maximum number of function evaluations" in str(e):
		# 		return False