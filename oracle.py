import random
import numpy as np
from scipy.optimize import lsq_linear
import warnings
#import torch
#from numpy.typing import NDArray, Shape
from lark import Lark
from strem.builder import build_formula
from strem.evaluator import Evaluator
from lmfit import Parameters, Minimizer
from lmfit.models import ExponentialModel, LinearModel, ConstantModel, Model
from numpy.linalg import LinAlgError
import time
import matplotlib.pyplot as plt

def draw_sequence2(trace, name):
	n_values = list(range(0, len(trace)))
	plt.plot(n_values, trace, marker='o', linestyle='None', color='b', label=r'$a_n$')
	plt.xlabel('n (Index)')
	plt.ylabel(r'$a_n$ (Value)')
	plt.xticks(n_values)  
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.legend()
	plt.savefig(name+'.png')
	plt.clf()

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
					

    UNION: "[union]"
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
	def __init__(self, query_map: dict[str, int], noise_tolerance: float, max_query_length):
		warnings.filterwarnings("ignore", category=FutureWarning, module="uncertainties")
		self._queries: list = [None]*len(query_map)
		self._tolerance = noise_tolerance
		self._fill_queries(query_map)
		self._cache = {}
		self._current_x = np.zeros(0, dtype=int)
		self.trace = np.zeros(0, dtype=float)
		self.query_count = 0
		self._params = {}
		self._r2 = {}
		self._funcs = {"e": self.exp, "l": self.lin, "s": self.sin}
		self._max_query_length = max_query_length

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			lb_list = []
			ub_list = []
			parts = query_str.split("_")
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
			self._queries[query_id] = (parts[0], lb_list, ub_list)

	def compute(self, query_id: int, fromm: int, to: int) -> bool:
		if (query_id, fromm, to) in self._cache:
			return self._cache[query_id, fromm, to]
		self.query_count += 1
		if self.query_count % 1000 == 0:
			print(self.query_count)
		result = self.atomic_match(fromm, to, *self._queries[query_id])
		self._cache[query_id, fromm, to] = result
		if result and to-fromm>20:
			print("Good", self._queries[query_id], fromm, to)
		return result
	
	def normalize(self, trace):
		min_trace = trace.min()
		scale = trace.max()-min_trace
		if scale > 0.0:
			return (trace-min_trace)/scale
		return trace
	
	def exp_objective(self, params, t, data, lb_list, ub_list):
		c = params['c'].value
		phi = np.column_stack([np.ones_like(t), np.exp(c * t)])
		res = lsq_linear(phi, data, bounds=(lb_list, ub_list), lsq_solver='lsmr')
		a, b = res.x
		self._exp_params[c] = a, b
		output = (a + b * np.exp(c * t)) - data
		return output
	
	def exp_match(self, fromm, to, lb_list, ub_list) -> bool:
		trace = self.trace[fromm: to]
		x = self._current_x[:len(trace)]/self._max_query_length
		if self.incremental_check(fromm, to, "e"):
			return True
		model = Model(self.exp)
		params = model.make_params()
		param_names = ['a','b','c']
		for i in range(len(param_names)):
			name = param_names[i]
			params[name].min = lb_list[i]
			params[name].max = ub_list[i]
		a, b, c = 0, 1, 0
		if ("e",fromm, to-1) in self._params:
			a, b, c = self._params["e", fromm, to-1]
		result = model.fit(trace, t=x, a=a, b=b, c=c)
		if result.rsquared > 0.7:
			self._params["e", fromm, to] = result.best_values['a'], result.best_values['b'], result.best_values['c']
		if result.rsquared>.98:
			self._r2["e", fromm, to] = self.get_res_tot_mean(trace, float(result.rsquared))
			return True
		return False
	
	def incremental_check(self, fromm, to, func_type):
		if (func_type, fromm, to-1) not in self._r2:
			return False
		res, tot, mean = self._r2[func_type, fromm, to-1]
		func = self._funcs[func_type]
		params = self._params[func_type, fromm, to-1]
		element_new = self.trace[to-1]/self._max_query_length
		pred_new = func(element_new, *params)
		res, tot, mean = self.incremental_r2(res, tot, mean, element_new, pred_new, to-fromm-1)
		r2 = 1-res/tot
		if r2 <= .98:
			return False
		self._r2[func_type, fromm, to] = res, tot, mean
		self._params[func_type, fromm, to] = self._params[func_type, fromm, to-1]
		return True
	
	def incremental_r2(self, res, tot, mean, element_new, pred_new, len):
		mean_new = (len*mean+element_new)/(len+1)
		tot_new = tot + (element_new-mean)*(element_new-mean_new)
		res_new = res + (element_new-pred_new)**2
		return res_new, tot_new, mean_new
	
	def get_res_tot_mean(self, trace, r2):
		mean = float(np.mean(trace))
		tot = float(np.sum((trace-mean)**2))
		res = (1-r2)*tot
		return res, tot, mean
	
	def get_decay(self, lb_list, ub_list, is_positive) -> bool:
		if is_positive:
			return 1, max(lb_list[2], .01), ub_list[2]
		return -1, lb_list[2], min(ub_list[2], -1e-100)
	
	def lin_match(self, fromm, to, lb_list, ub_list) -> bool:
		trace = self.trace[fromm: to]
		model = LinearModel()
		x = self._current_x[:len(trace)]
		if self.incremental_check(fromm, to, "l"):
			return True
		params = None
		try:
			if ("l", fromm, to-1) in self._params:
				slope, intercept = self._params["l", fromm, to-1]
				params = model.make_params()
				params['slope'].set(value=slope)
				params['intercept'].set(value=intercept)
			else:
				params = model.guess(x, trace)
		except LinAlgError:
			print("BBBB")
			print(trace)
			return True
		params['slope'].set(min=lb_list[1], max=ub_list[1])
		params['intercept'].set(min=lb_list[0], max=ub_list[0])
		result = model.fit(trace, params, x=x)
		if result.rsquared>.7:
			self._params["l", fromm, to] = result.params["slope"], result.params["intercept"]
		if result.rsquared>.98:
			self._r2["l", fromm, to] = self.get_res_tot_mean(trace, float(result.rsquared))
			return True
		return False

	def r2sq(self, residual, y):
		rss = np.sum(residual**2)
		tss = np.sum((y - np.mean(y))**2)
		if tss==0:
			return 0.0
		return 1 - (rss / tss)
	
	def atomic_match(self, fromm, to, shape, lb_list, ub_list) -> bool:
		#trace = self.normalize_trace(trace)
		# param_names = ['a','b','c','d']
		# model = Model(shape, independent_vars=['t'])
		# params = model.make_params()
		# for i in range(len(lb_list)):
		# 	if lb_list[i] != np.inf and lb_list[i] != -np.inf:
		# 		params[param_names[i]].set(min=lb_list[i])
		# 	if ub_list[i] != np.inf and ub_list[i] != -np.inf:
		# 		params[param_names[i]].set(max=ub_list[i])
		# result = model.fit(trace, params, t=self._current_x[:len(trace)])
		# summary = result.summary()
		if shape == "e":
			return self.exp_match(fromm, to, lb_list, ub_list)
		if shape == "l":
			return self.lin_match(fromm, to, lb_list, ub_list)
		return 0
	
	def add_frame(self, frame):
		self.trace = np.append(self.trace, [frame])
		self._current_x = np.append(self._current_x, [len(self.trace)-1])

	def lin(self, t, a=0, b=0):
		return a*t+b
	
	def exp(self, t, a=0, b=0, c=0) -> float:
		exponent = np.clip(c * t, -700, 700)
		return a+b*np.exp(exponent)
	
	def sin(self, t, a=0, b=0, c=0, d=0) -> float:
		return a+b*np.sin(c*t+d)
	
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
		self.query_count = 0

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			self._queries[query_id] = build_formula(strem_parser.parse(query_str))

	def add_frame(self, frame):
		self.trace.append(frame)

	def match(self, formula, frame) -> bool:
		evaluator = Evaluator(frame)
		return evaluator.eval(formula)

	def compute(self, query_id: int, fromm: int, to: int) -> float:
		if (query_id, fromm) in self._cache:
			return self._cache[query_id, fromm]
		output = -1
		if to-fromm >1:
			output = 1.0
			for i in range(fromm, to):
				output = min(output, self.compute(query_id, i, i+1))
			return output
		else:
			self.query_count += 1
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

