from abc import ABC, abstractmethod
import random
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

class Oracle(ABC):
	@abstractmethod
	def __init__(self, query_map: dict[str, int]):
		pass

	@abstractmethod
	def _fill_queries(query_map):
		pass

	@abstractmethod
	def compute(self, query_id: int, trace: list[object]) -> float:
		pass

class RandomOracle(Oracle):
	def __init__(self, query_map: dict[str, int]):
		pass

	def _fill_queries(query_map):
		pass

	def compute(self, query_id: int, trace: list[object]) -> float:
		response = random.random()
		return response
	
class ShapeExpressionOracle(Oracle):
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
