import random
import numpy as np
import warnings
#import torch
#from numpy.typing import NDArray, Shape
from lark import Lark
from strem.builder import build_formula
from strem.evaluator import Evaluator
from lmfit.models import Model
import random
import matplotlib.pyplot as plt
import time

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
	def __init__(self, query_map: dict[str, int], noise_tolerance: float, scale_factor):
		warnings.filterwarnings("ignore", category=FutureWarning, module="uncertainties")
		self._queries: list = [None]*len(query_map)
		self._tolerance = noise_tolerance
		self._fill_queries(query_map)
		self._cache = {}
		self._current_x = np.zeros(0, dtype=int)
		self.trace = np.zeros(0, dtype=float)
		self.trace_sum = np.zeros(0, dtype=float)
		self.query_count = 0
		self._params = {}
		self._r2 = {}
		self._funcs = {"e": self.exp, "l": self.lin, "s": self.sinc}
		self.scale_factor = scale_factor

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
		result = self.atomic_match(fromm, to, query_id)
		if result:
			print(self._queries[query_id], fromm, to)
		self._cache[query_id, fromm, to] = result
		return result
	
	def heuristic_exp_reject(self, fromm, to) -> bool:
		trace = self.trace[fromm: to]
		if len(trace)<10:
			return False
		mid = trace[int(len(trace)/2)]
		if mid>=trace[0] and mid>=trace[-1] or mid <=trace[0] and mid <= trace[-1]:
			return True
		third = len(trace)/3
		min_idx = np.argmin(trace)
		max_idx = np.argmax(trace)
		return (max_idx > third and max_idx <2*third) or (min_idx > third and min_idx <2*third)
	
	def exponent_heuristics(self, trace) -> float:
		sample_size = 10
		gradient_approx = np.zeros(sample_size)
		for i in range(len(gradient_approx)):
			idx = random.randint(0, len(trace)-3)
			if trace[idx+1] != trace[idx]:
				gradient_approx[i] = (trace[idx+2]-trace[idx+1])/(trace[idx+1]-trace[idx])
		result = gradient_approx[gradient_approx != 0]
		if len(result) == 0:
			return 0
		mean = np.mean(result)
		if mean<= 0:
			return 0
		return self.scale_factor*float(np.log(mean))

	def exp_match(self, fromm, to, query_id) -> bool:
		_, lb_list, ub_list = self._queries[query_id]
		trace = self.trace[fromm: to]
		x = self._current_x[:len(trace)]/self.scale_factor
		if self.incremental_check(fromm, to, x[-1], query_id):
			return True
		if self.heuristic_exp_reject(fromm, to):
			return False
		model = self.get_model(self.exp, lb_list, ub_list)
		a, b, c = 0, 1, 0
		if (query_id,fromm, to-1) in self._params:
			a, b, c = self._params[query_id, fromm, to-1]
		else:
			c = self.exponent_heuristics(trace)
		result = model.fit(trace, t=x, a=a, b=b, c=c)
		duration = time.perf_counter()-start
		if result.rsquared > 0.7:
			self._params[query_id, fromm, to] = result.best_values['a'], result.best_values['b'], result.best_values['c']
		if result.rsquared>.98:
			mean = self.get_trace_mean(fromm, to)
			self._r2[query_id, fromm, to] = self.get_res_tot(trace, float(result.rsquared), mean)
			return True
		return False
	
	def incremental_check(self, fromm, to, x_new, query_id):
		if (query_id, fromm, to-1) not in self._r2:
			return False
		res, tot = self._r2[query_id, fromm, to-1]
		mean = self.get_trace_mean(fromm, to)
		func = self._funcs[self._queries[query_id][0]]
		params = self._params[query_id, fromm, to-1]
		y_new = self.trace[to-1]
		pred_new = func(x_new, *params)
		res, tot = self.incremental_r2(res, tot, mean, y_new, pred_new, to-fromm-1)
		r2 = 1-res/tot if tot != 0.0 else 0.0
		if r2 <= .98:
			return False
		self._r2[query_id, fromm, to] = res, tot
		self._params[query_id, fromm, to] = self._params[query_id, fromm, to-1]
		return True
	
	def incremental_r2(self, res, tot, mean, element_new, pred_new, len):
		mean_new = (len*mean+element_new)/(len+1)
		tot_new = tot + (element_new-mean)*(element_new-mean_new)
		res_new = res + (element_new-pred_new)**2
		return res_new, tot_new
	
	def get_res_tot(self, trace, r2, mean):
		tot = float(np.sum((trace-mean)**2))
		res = (1-r2)*tot
		return res, tot
	
	def get_trace_mean(self, fromm, to):
		sum = self.trace_sum[to-1] if fromm == 0 else self.trace_sum[to-1]-self.trace_sum[fromm-1]
		return sum/(to-fromm)
	
	def lin_match(self, fromm, to, query_id) -> bool:
		trace = self.trace[fromm:to]
		_, lb_list, ub_list = self._queries[query_id]
		trace = self.trace[fromm: to]
		x = self._current_x[:len(trace)]
		if self.incremental_check(fromm, to, x[-1], query_id):
			return True
		res = np.polyfit(x, trace, 1)
		a, b = res[0], res[1]
		a = max(lb_list[0], min(ub_list[0], a))
		b = max(lb_list[1], min(ub_list[1], b))
		mean = self.get_trace_mean(fromm, to)
		res, tot = self.r2sq(trace-(a*x+b), trace, mean)
		if tot == 0:
			return False
		r2 = 1-res/tot
		if r2>.98:
			self._params[query_id, fromm, to] = a, b
			self._r2[query_id, fromm, to] = res, tot
			return True
		return False
	
	def get_model(self, func, lb_list, ub_list):
		model = Model(func)
		params = model.make_params()
		param_names = ['a','b','c']
		for i in range(len(param_names)):
			name = param_names[i]
			params[name].min = lb_list[i]
			params[name].max = ub_list[i]
		return model
	
	def heuristic_sinc_reject(self, fromm, to):
		trace = self.trace[fromm: to]
		length = to-fromm
		mean = self.get_trace_mean(fromm, to)
		normalized_trace = trace-mean
		ratio = np.max(np.abs(normalized_trace))/np.sqrt(np.sum(normalized_trace**2)/length)
		return float(ratio)<1.5
	
	def sinc_match(self, fromm, to, query_id):
		_, lb_list, ub_list = self._queries[query_id]
		trace = self.trace[fromm: to]
		x = self._current_x[:len(trace)]/self.scale_factor
		if self.incremental_check(fromm, to, x[-1], query_id):
			return True
		if self.heuristic_sinc_reject(trace):
			return False
		model = self.get_model(self.sinc, lb_list, ub_list)
		a, b, c = 0, 1, 0
		if (query_id,fromm, to-1) in self._params:
			a, b, c = self._params[query_id, fromm, to-1]
		else:
			c = self.exponent_heuristics(trace)
		result = model.fit(trace, t=x, a=a, b=b, c=c)
		if result.rsquared > 0.7:
			self._params[query_id, fromm, to] = result.best_values['a'], result.best_values['b'], result.best_values['c']
		if result.rsquared>.98:
			self._r2[query_id, fromm, to] = self.get_res_tot(trace, float(result.rsquared))
			return True
		return False

	def r2sq(self, residual, y, mean):
		res = np.sum(residual**2)
		tot = np.sum((y - mean)**2)
		return res, tot
	
	def atomic_match(self, fromm, to, query_id) -> bool:
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
		shape, _, _ = self._queries[query_id]
		if shape == "e":
			return self.exp_match(fromm, to, query_id)
		elif shape == "l":
			return self.lin_match(fromm, to, query_id)
		else:
			return self.sinc_match(fromm, to, query_id)
	
	def add_frame(self, frame):
		self.trace = np.append(self.trace, [frame])
		sum = self.trace[0] if len(self.trace) == 1 else self.trace_sum[-1]+self.trace[-1]
		self.trace_sum = np.append(self.trace_sum, [sum])
		self._current_x = np.append(self._current_x, [len(self.trace)-1])

	def lin(self, t, a=0, b=0):
		return a*t+b
	
	def exp(self, t, a=0, b=0, c=0) -> float:
		exponent = np.clip(c * t, -700, 700)
		return a+b*np.exp(exponent)
	
	def sinc(self, t, a=0, b=0, c=0, d=0) -> float:
		return a+b*np.sinc((c*t+d)/np.pi)
	
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

