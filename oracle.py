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
from scipy.optimize import lsq_linear

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
	def __init__(self, query_map: dict[str, int], threshold: float, scale_factor):
		warnings.filterwarnings("ignore", category=FutureWarning, module="uncertainties")
		self._queries: list = [None]*len(query_map)
		self._threshold = threshold
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

	def calculate_incremental_range(self, first_true_to, last_true_to, first_false_to, to):
		if to>last_true_to and first_false_to == float('inf'):
			return last_true_to+1, to+1
		to_range = min(first_true_to, first_false_to, len(self.trace)+1)
		return to, to_range

	def compute(self, query_id: int, fromm: int, to: int) -> bool:
		first_true_to = last_true_to = first_false_to = float('inf')
		if (query_id, fromm) in self._cache:
			first_true_to, last_true_to, first_false_to = self._cache[query_id, fromm]
			if first_false_to != float('inf'):
				if to >= first_false_to:
					return False
				if to >= first_true_to:
					return True
			elif to>= first_true_to and to <= last_true_to:
				return True
		self.query_count += 1
		fromm_range, to_range = self.calculate_incremental_range(first_true_to, last_true_to, first_false_to, to)
		result = False
		final_result = False
		for i in range(fromm_range, to_range):
			result = self.atomic_match(fromm, i, query_id)
			if i == to:
				final_result = result
			if not result:
				first_false_to = i
				break
			last_true_to = i if last_true_to == float('inf') else max(last_true_to, i)
			first_true_to = min(first_true_to, i)
		self._cache[query_id, fromm] = (first_true_to, last_true_to, first_false_to)
		return final_result
	
	def get_exp_coeffs_integral(self, y, t):
		n = len(t)
		S = np.zeros(n)
		sign_change = 0
		for i in range(1, n):
			if i<n-1 and (y[i]-y[i-1])*(y[i+1]-y[i])<=0:
				sign_change += 1
			S[i] = S[i-1] + 0.5 * (y[i] + y[i-1]) * (t[i] - t[i-1])
		if sign_change>=0.35*n:
			return None
		M1 = np.column_stack((S, t - t[0]))
		Y1 = y - y[0]
		coeffs, _, _, _ = np.linalg.lstsq(M1, Y1, rcond=None)
		c_est = coeffs[0]
		X = np.exp(c_est * t)
		M2 = np.column_stack((np.ones(n), X))
		final_coeffs, _, _, _ = np.linalg.lstsq(M2, y, rcond=None)
		a_est, b_est = final_coeffs
		return a_est, b_est, c_est
	
	def check_bounds(self, lb_list, ub_list, *params):
		for i in range(len(lb_list)):
			if params[i] <= lb_list[i] or params[i]>=ub_list[i]:
				return False
		return True
	
	def exp_match(self, fromm, to, query_id) -> bool:
		_, lb_list, ub_list = self._queries[query_id]
		trace = self.trace[fromm: to]
		x = self._current_x[:len(trace)]/self.scale_factor
		if self.incremental_check(fromm, to, x[-1], query_id):
			return True
		coeffs = self.get_exp_coeffs_integral(trace, x)
		if coeffs is None or not self.check_bounds(lb_list, ub_list, *coeffs):
			return False
		a, b, c = coeffs
		model = self.get_model(self.exp, lb_list, ub_list, a, b, c)
		result = model.fit(trace, t=x)
		if result.rsquared<self._threshold:
			return False
		best_vals = result.params
		a, b, c = float(best_vals['a']), float(best_vals['b']), float(best_vals['c'])
		if b==0 or c==0:
			return False
		mean = self.get_trace_mean(fromm, to)
		pred = a+b*np.exp(c*x)
		res, tot = self.r2sq(trace-pred, trace, mean)
		self._params[query_id, fromm, to] = a, b, c
		self._r2[query_id, fromm, to] = res, tot
		return True
	
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
		if r2 <= self._threshold:
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
		A = np.column_stack([x, np.ones_like(x)])
		res = lsq_linear(A, trace, bounds=(lb_list, ub_list))
		a, b = res.x
		mean = self.get_trace_mean(fromm, to)
		res, tot = self.r2sq(trace-(a*x+b), trace, mean)
		if tot == 0:
			return False
		r2 = 1-res/tot
		if r2>self._threshold:
			self._params[query_id, fromm, to] = a, b
			self._r2[query_id, fromm, to] = res, tot
			return True
		return False
	
	def get_model(self, func, lb_list, ub_list, *params_initial):
		model = Model(func)
		params = model.make_params()
		param_names = ['a','b','c']
		for i in range(len(param_names)):
			name = param_names[i]
			params[name].set(
			value=params_initial[i],
			min=lb_list[i],
			max=ub_list[i],
			vary=True)
		return model
	
	def heuristic_sinc_reject(self, fromm, to):
		trace = self.trace[fromm: to]
		length = to-fromm
		mean = self.get_trace_mean(fromm, to)
		normalized_trace = trace-mean
		ratio = np.max(np.abs(normalized_trace))/np.sqrt(np.sum(normalized_trace**2)/length)
		return float(ratio)<1.5

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
			
	def add_frame(self, frame):
		self.trace.append(frame)

	def _get_area(self, obj):
		return obj['w']*obj['h']

	def _get_sum_ratios(self, trace, ids):
		ratios = {id: 0 for id in ids}
		for i in range(1, len(trace)):
			for id in ids:
				if ratios[id] == -1 or id not in trace[i] or id not in trace[i-1]:
					ratios[id] = -1
				else:
					area_now = self._get_area(trace[i][id])
					area_prev = self._get_area(trace[i-1][id])
					ratios[id] = ratios[id]+area_now/area_prev
		return ratios
	
	def _exceeds_ratio(self, n1: float, n2: float, min_ratio: float):
		return n1>0 and n2>0 and (n1/n2>=min_ratio or n2/n1>=min_ratio) 
	
	def _get_score(self, trace, id):
		score = trace[0][id]['score']
		for frame in trace:
			score = min(score, frame[id]['score'])
		return score

	def _velocity_min_ratio_exists(self, trace, class_id1: str, class_id2: str, min_ratio: float):
		if len(trace)<2:
			return 0
		class1_objs = [id for id, val in trace[0].items() if val['class']==class_id1]
		class2_objs = [id for id, val in trace[0].items() if val['class']==class_id2]
		if len(class1_objs) == 0 or len(class2_objs) == 0:
			return 0
		ratios1 = self._get_sum_ratios(trace, class1_objs)
		ratios2 = self._get_sum_ratios(trace, class2_objs)
		ratio = 0
		for id1 in class1_objs:
			for id2 in class2_objs:
				if id1 != id2 and self._exceeds_ratio(ratios1[id1], ratios2[id2], min_ratio):
					ratio = max(min(self._get_score(trace, id1), self._get_score(trace, id2)), ratio)
		return ratio

			
	def compute(self, query_id: int, fromm: int, to: int) -> float:
		if (query_id, fromm, to) in self._cache:
			return self._cache[query_id, fromm, to]
		self.query_count += 1
		query_parts = self._queries[query_id]
		output = self._velocity_min_ratio_exists(self.trace[fromm:to], query_parts[0], query_parts[1], float(query_parts[2]))
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

