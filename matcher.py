from prop import *

class Matcher:
	def __init__(self, prop: SeqTLProp, oracle):
		self._prop = prop
		self.oracle = oracle
		self.counter = 0
		self._refined_cache = {}
		self._refine_frees_cache = {}
		self._refine_frees = set()
		self.add_refine_free_parts(self._prop)
		for prop in self._refine_frees:
			self._refine_frees_cache[prop] = set()
		self.refine_free_update_all(0)

	def refine_free_update_all(self, length: int):
		for prop in self._refine_frees:
			self.refine_free_update(prop, length)

	def refine_free_update(self, prop: SeqTLProp, length: int):
		if type(prop) == StarProp:
			self.star_refine_free_update(prop, length)
		elif type(prop) == LengthProp:
			if length >= prop.lb and length <= prop.ub:
				self._refine_frees_cache[prop].add(length)
		elif type(prop) == UnionProp:
			self.union_refine_free_update(prop)
		elif type(prop) == ConcatProp:
			self.concat_refine_free_update(prop)

	def concat_refine_free_update(self, prop: UnionProp, length: int):
		self.refine_free_update(prop.left)
		self.refine_free_update(prop.right)
		set1 = self._refine_frees_cache[prop.left]
		set2 = self._refine_frees_cache[prop.right]
		swap_set = set()
		if len(set1) > len(set2):
			swap_set = set1
			set1 = set2
			set2 = swap_set
		for i in set1:
			if length-i in set2:
				self._refine_frees_cache[prop].add(length)
				break
	
	def union_refine_free_update(self, prop: UnionProp, length: int):
		self.refine_free_update(prop.left)
		self.refine_free_update(prop.right)
		if length in self._refine_frees_cache[prop.left] or length in self._refine_frees_cache[prop.right]:
			self._refine_frees_cache[prop].add(length)
	
	def star_refine_free_update(self, prop: StarProp, length: int):
		if length == 0:
				self._refine_frees_cache[prop].add(0)
		self.refine_free_update(prop.child, length)
		for i in self._refine_frees_cache[prop.child]:
			if length-i in self._refine_frees_cache[prop]:
				self._refine_frees_cache[prop].add(length)
				break
	
	def add_refine_free_parts(self, prop: SeqTLProp):
		if type(prop) == LengthProp:
			self._refine_frees.add(prop)
			return True
		if type(prop) == RefineProp:
			self.add_refine_free_parts(prop.child)
			return False
		if type(prop) == StarProp:
			result = self.add_refine_free_parts(prop.child)
			if result:
				self._refine_frees.add(prop)
			return result
		if type(prop) in [UnionProp, ConcatProp]:
			first = self.add_refine_free_parts(prop.left)
			last = self.add_refine_free_parts(prop.right)
			result = first and last
			if result:
				self._refine_frees.add(prop)
			return result
		
	def evaluate(self, prop: SeqTLProp, fromm: int, to: int): # fromm inclusive, to exclusive
		if prop in self._refine_frees:
			return to-fromm in self._refine_frees_cache[prop]
		if (prop, fromm, to) in self._refined_cache:
			return self._refined_cache[prop, fromm, to]
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
			self._refined_cache[prop, fromm, to] = 0.0
			return 0.0
		result = min(self.oracle.compute(prop.query_id, fromm, to), child_val)
		self._refined_cache[prop, fromm, to] = result
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
		self._refined_cache[prop, fromm, to] = output
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
		self._refined_cache[prop, fromm, to] = output
		return output

	def eval_union(self, prop: UnionProp, fromm: int, to: int):
		first_val = self.evaluate(prop.left, fromm, to)
		if first_val == 1.0:
			self._refined_cache[prop, fromm, to] = 1.0
			return 1.0
		second_val = self.evaluate(prop.right, fromm, to)
		val = max(first_val, second_val)
		self._refined_cache[prop, fromm, to] = val
		return val

	def match(self, frame):
		self.oracle.add_frame(frame)
		self.refine_free_update_all(len(self.oracle.trace))
		return self.evaluate(self._prop, 0, len(self.oracle.trace))
