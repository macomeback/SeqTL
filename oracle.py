import random
import numpy as np
#import torch
#from numpy.typing import NDArray, Shape
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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
	
class VideoOracle:
	def __init__(self, query_map: dict[str, int]):
		self._queries: list = [None]*len(query_map)
		self._fill_queries(query_map)

	def _fill_queries(self, query_map):
		for query_str, query_id in query_map.items():
			self._queries[query_id] = query_str
			
	def iou(self, boxA: list[float], boxB: list[float]) -> float:
		# Determine the coordinates of the intersection rectangle
		x_left = max(boxA[0], boxB[0])
		y_top = max(boxA[1], boxB[1])
		x_right = min(boxA[2], boxB[2])
		y_bottom = min(boxA[3], boxB[3])
		# Compute the area of intersection rectangle
		intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
		# Compute the area of both bounding boxes
		boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
		boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
		# Compute the union area
		union_area = boxA_area + boxB_area - intersection_area
		# Compute the IoU, handling potential division by zero
		if union_area == 0:
			return 0.0
		iou_val = intersection_area / float(union_area)
		return iou_val
	    
	def match_boxes(self, prev_boxes: dict[int, list[float]], boxes: list[list[float]], max_index: int) -> dict[int, list[float]]:
		matching = {}
		for box in boxes:
			threshold = 0.5
			matched = -1
			for i, prev_box in prev_boxes.items():
				iou_val = self.iou(box, prev_box)
				if iou_val >= threshold:
					matched = i
					threshold = iou_val
			if matched >=0:
				matching[matched] = box
			else:
				matching[max_index] = box
				max_index += 1
		return matching
	    
	def track(self, id_trace: list[list[list[float]]]) -> list[dict[int, list[float]]]:
		tracks = []
		max_index = len(id_trace[0])-1
		initial_objs = {}
		for i in range(0, len(id_trace[0])):
			initial_objs[i] = id_trace[i]
		tracks.append(initial_objs)
		for i in range(1, len(id_trace)):
			tracks.append(self.match_boxes(tracks[i-1], id_trace[i],max_index))
		return tracks

	def get_id_traces(self, trace: list[list[list[float]]]) -> dict[int, list[list[list[float]]]]:
		id_traces: dict[int, list[list[list[float]]]] = {}
		for box in trace[0]:
			class_id = int(float(box[5]))
			if class_id not in id_traces:
				id_traces[class_id] = [[]]
			id_traces[class_id][0].append(box)
		id_traces_new = {}
		for class_id, id_trace in id_traces.items():
			if len(id_trace[0]) >= 2:
				id_traces_new[class_id] = id_trace
		id_traces = id_traces_new
		for i in range(1, len(trace)):
			for box in trace[i]:
				class_id = int(float(box[5]))
				if class_id in id_traces:
					if i == len(id_traces[class_id]):
						id_traces[class_id].append([[]])
					id_traces[class_id][i].append(box)
			for class_id in id_traces.keys():
				if len(id_traces[class_id]) < i+1 or len(id_traces[class_id][i]) < 2:
					del id_traces[class_id]
		return id_traces
	
	def get_distance(self, box1: list[float], box2: list[float]) -> float:
		x1 = (box1[0]+box1[2])/2
		y1 = (box1[1]+box1[3])/2
		x2 = (box2[0]+box2[2])/2
		y2 = (box2[1]+box2[3])/2
		return (x1-x2)**2+(y1-y2)**2

	def getting_closer(self, id_track:list[dict[int, list[float]]] , idx1: int, idx2: int) -> float:
		init_distance = self.get_distance(id_track[0][idx1], id_track[0][idx2])
		confidence = min(id_track[0][idx1][4], id_track[0][idx2][4])
		for i in range(1, len(id_track)):
			distance = self.get_distance(id_track[i][idx1], id_track[i][idx2])
			if init_distance*1.05 < distance or (i == len(id_track)-1 and init_distance*0.95 < distance):
				return 0.0
			confidence = min(confidence, id_track[i][idx1][4], id_track[i][idx2][4])
		return confidence
	
	def getting_further(self, id_track:list[dict[int, list[float]]] , idx1: int, idx2: int) -> float:
		init_distance = self.get_distance(id_track[0][idx1], id_track[0][idx2])
		confidence = min(id_track[0][idx1][4], id_track[0][idx2][4])
		for i in range(1, len(id_track)):
			distance = self.get_distance(id_track[i][idx1], id_track[i][idx2])
			if init_distance*0.95 > distance or (i == len(id_track)-1 and init_distance*1.05 > distance):
				return 0.0
			confidence = min(confidence, id_track[i][idx1][4], id_track[i][idx2][4])
		return confidence
			
	def distance_direction(self, trace: list[list[list[float]]], is_closer: bool) -> float:
		id_traces = self.get_id_traces(trace)
		confidence = 0.0
		for _, id_trace in id_traces.items():
			id_track = self.track(id_trace)
			objects = list(id_track[0].keys())
			for i in range(len(objects)):
				for j in range(i+1, len(objects)):
					confidence = max(confidence, self.getting_closer(id_track, i, j) if is_closer else self.getting_further(id_track, i, j))
			
	def compute(self, query_id: int, trace: list[list[list[float]]]) -> float:
		return self.distance_direction(trace, self._queries[query_id] == "close")
			
