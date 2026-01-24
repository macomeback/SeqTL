from prop import *

class Matcher:
	def __init__(self, prop: SeqTLProp, oracle):
		self._prop = prop
		self.oracle = oracle
		self.counter = 0
		self._cache = {}
		self._refine_frees = set()
		self.add_largest_refine_free_parts(self._prop)
		self._epsilon_accepting = set()
		for prop in self._refine_frees:
			self.add_epsilon_accepting(prop)

	def add_epsilon_accepting(self, prop: SeqTLProp):
		if type(prop) == StarProp:
			self.add_epsilon_accepting(prop.child)
			self._epsilon_accepting.add(prop)
		elif type(prop) == LengthProp and prop.lb == 0:
			self._epsilon_accepting.add(prop)
		elif type(prop) == UnionProp:
			self.add_epsilon_accepting(prop.left)
			self.add_epsilon_accepting(prop.right)
			if prop.left in self._epsilon_accepting or prop.right in self._epsilon_accepting:
				self._epsilon_accepting.add(prop)
		elif type(prop) == ConcatProp:
			self.add_epsilon_accepting(prop.left)
			self.add_epsilon_accepting(prop.right)
			if prop.left in self._epsilon_accepting and prop.right in self._epsilon_accepting:
				self._epsilon_accepting.add(prop)

	def add_largest_refine_free_parts(self, prop: SeqTLProp):
		if type(prop) == LengthProp:
			return True
		if type(prop) == RefineProp:
			_ = self.add_largest_refine_free_parts(prop.child)
			return False
		if type(prop) == StarProp:
			result = self.add_largest_refine_free_parts(prop.child)
			if result:
				self._refine_frees.discard(prop.child)
				self._refine_frees.add(prop)
			return result
		if type(prop) in [UnionProp, ConcatProp]:
			first = self.add_largest_refine_free_parts(prop.left)
			last = self.add_largest_refine_free_parts(prop.right)
			result = first and last
			if result:
				self._refine_frees.discard(first)
				self._refine_frees.discard(last)
				self._refine_frees.add(prop)
			return result
		
	def norefine_update(self, prop: SeqTLProp):
		if type(prop) == LengthProp:
			self.length_norefine_update(prop)
		elif type(prop) == UnionProp:
			self.union_norefine_update(prop)
		elif type(prop) == ConcatProp:
			self.concat_norefine_update(prop)
		elif type(prop) == StarProp:
			self.star_norefine_prop(prop)

	def star_norefine_prop(self, prop: StarProp):
		self.norefine_update(prop.child)
		trace_length = len(self.oracle.trace)
		for i in range(trace_length-1, -1, -1):
			result = 0.0
			for mid in range(i+1, trace_length+1):
				left_val = self._cache[prop.child, i, mid]
				right_val = self._cache[prop, mid, trace_length] if mid < trace_length else 1.0
				result = max(result, min(left_val, right_val))
			self._cache[prop, i, trace_length] = result

	def union_norefine_update(self, prop: UnionProp):
		self.norefine_update(prop.left)
		self.norefine_update(prop.right)
		trace_length = len(self.oracle.trace)
		for i in range(trace_length):
			self._cache[prop, i, trace_length] = max(self._cache[prop.left, i, trace_length], self._cache[prop.right, i, trace_length])

	def concat_norefine_update(self, prop: ConcatProp):
		self.norefine_update(prop.left)
		self.norefine_update(prop.right)
		trace_length = len(self.oracle.trace)
		for i in range(trace_length-1, -1, -1):
			result = 0.0
			for mid in range(i, trace_length+1):
				left_val = self._cache[prop.left, i, mid] if i < mid else prop.left in self._epsilon_accepting
				right_val = self._cache[prop.right, mid, trace_length] if mid < trace_length else prop.right in self._epsilon_accepting
				result = max(result, min(left_val, right_val))
			self._cache[prop, i, trace_length] = result
			
	def length_norefine_update(self, prop: LengthProp):
		trace_length = len(self.oracle.trace)
		for i in range(trace_length):
			length = len(self.oracle.trace)-i
			self._cache[prop, i, trace_length] = length >= prop.lb and length <= prop.ub

	def evaluate(self, prop: SeqTLProp, fromm: int, to: int): # fromm inclusive, to exclusive
		if (prop, fromm, to) in self._cache:
			return self._cache[prop, fromm, to]
		if type(prop) == LengthProp:
			length = to-fromm
			return length >= prop.lb and length <= prop.ub
		if type(prop) == UnionProp:
			return self.eval_union(prop, fromm, to)
		if type(prop) == ConcatProp:
			return self.eval_concat(prop, fromm, to)
		if type(prop) == StarProp:
			return self.eval_star(prop, fromm, to)
		if type(prop) == RefineProp:
			return self.eval_refine(prop, fromm, to)
		
	def eval_refine(self, prop: RefineProp, fromm: int, to: int):
		length = to-fromm
		if length > prop.ub or prop.lb > length:
			return 0.0
		child_val = self.evaluate(prop.child, fromm, to)
		if child_val == 0.0:
			self._cache[prop, fromm, to] = 0.0
			return 0.0
		result = min(self.oracle.compute(prop.query_id, fromm, to), child_val)
		self._cache[prop, fromm, to] = result
		return result
		
	def eval_star(self, prop: StarProp, fromm: int, to: int):
		if to == fromm:
			return 1.0
		output = 0.0
		for mid in range(fromm+1, to+1):
			child_val = self.evaluate(prop.child, fromm, mid)
			if child_val <= output:
				continue
			recursive_val = self.evaluate(prop, mid, to)
			output = max(min(child_val, recursive_val), output)
			if output == 1.0:
				break
		self._cache[prop, fromm, to] = output
		return output

	def eval_concat(self, prop: ConcatProp, fromm: int, to: int):
		output = 0
		for mid in range(fromm, to+1):
			left_val = self.evaluate(prop.left, fromm, mid)
			if left_val <= output:
				continue
			right_val = self.evaluate(prop.right, mid, to)
			output = max(min(right_val, left_val), output)
			if output == 1.0:
				break
		self._cache[prop, fromm, to] = output
		return output

	def eval_union(self, prop: UnionProp, fromm: int, to: int):
		first_val = self.evaluate(prop.left, fromm, to)
		if first_val == 1.0:
			self._cache[prop, fromm, to] = 1.0
			return 1.0
		second_val = self.evaluate(prop.right, fromm, to)
		val = max(first_val, second_val)
		self._cache[prop, fromm, to] = val
		return val

	def match(self, frame):
		self.oracle.add_frame(frame)
		print(len(self.oracle.trace))
		for prop in self._refine_frees:
			self.norefine_update(prop)
		return self.evaluate(self._prop, 0, len(self.oracle.trace))
