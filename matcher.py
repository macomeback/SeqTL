from oracle import Oracle
from prop import *
from typing import Tuple, Optional

class Matcher:
	def __init__(self, prop: SeqTLProp, oracle: Oracle, trace: list[object]):
		self._prop = prop
		self._oracle = oracle
		self._trace = trace
		self._query_cache: dict[Tuple[int,int,int], float] = {}
		self._query_dnf_cache: dict[Tuple[SeqTLProp,int], Optional[set[set[Tuple[int,int,int]]]]] = {}
	
	def _get_query_dnf(self, prop: SeqTLProp, length: int) -> Optional[set[set[Tuple[int,int,int]]]]:
		if (prop, length) in self._query_dnf_cache:
			return self._query_dnf_cache[prop, length]
		if isinstance(prop, LengthProp):
			if length >= prop.lb and length <= prop.ub:
				return self._cache_query_pair(prop, length, set())
			return self._cache_query_pair(prop, length, None)
		if isinstance(prop, UnionProp):
			return self._get_union_query_dnf(prop, length)
		if isinstance(prop, ConcatProp):
			return self._get_concat_query_dnf(prop, length)
		if isinstance(prop, StarProp):
			return self._get_star_query_dnf(prop, length)
		if isinstance(prop, RefineProp):
			return self._get_refine_query_dnf(prop, length)
		
	def _get_refine_query_dnf(self, prop: RefineProp, length: int):
		if length < prop.lb or length > prop.ub:
			return self._cache_query_pair(prop, length, None)
		child_clause = self._get_query_dnf(prop.child, length)
		if child_clause is not None:
			clause = set()
			clause.update(child_clause)
			clause.add((prop.query_id, 0, length))
			print(clause)
			return self._cache_query_pair(prop, length, clause)
		return self._cache_query_pair(prop, length, None)
		
	def _conjunct_query_dnfs(self, idx: int, output_clause:Optional[set[set[Tuple[int,int,int]]]], dnf_left: Optional[set[set[Tuple[int,int,int]]]], dnf_right: Optional[set[set[Tuple[int,int,int]]]]):
		for clause_left in dnf_left:
				for clause_right in dnf_right:
					dnf_clause = set()
					dnf_clause.update(clause_left)
					for pair in clause_right:
						dnf_clause.add((pair[0], pair[1]+idx, pair[2]+idx))
					output_clause.add(dnf_clause)
		
	def _get_star_query_dnf(self, prop: SeqTLProp, length: int) -> Optional[set[set[Tuple[int,int,int]]]]:
		if length == 0:
			return self._cache_query_pair(prop, length, set())
		output_clause = set()
		valid_match = False
		for i in range(0, length+1):
			dnf_left = self._get_query_dnf(prop.child, i)
			if dnf_left is None:
				continue
			dnf_right = self._get_query_dnf(prop, length-i)
			if dnf_right is None:
				continue
			valid_match = True
			self._conjunct_query_dnfs(i,output_clause, dnf_left, dnf_right)
		if not valid_match:
			return self._cache_query_pair(prop, length, None)
		return self._cache_query_pair(prop, length, output_clause)
		
	def _get_concat_query_dnf(self, prop: SeqTLProp, length: int) -> Optional[set[set[Tuple[int,int,int]]]]:
		output_clause = set()
		valid_match = False
		for i in range(0, length+1):
			dnf_left = self._get_query_dnf(prop.left, i)
			if dnf_left is None:
				continue
			dnf_right = self._get_query_dnf(prop.right, length-i)
			if dnf_right is None:
				continue
			valid_match = True
			self._conjunct_query_dnfs(i,output_clause, dnf_left, dnf_right)
		if not valid_match:
			return self._cache_query_pair(prop, length, None)
		return self._cache_query_pair(prop, length, output_clause)
		
		
	def _get_union_query_dnf(self, prop: SeqTLProp, length: int) -> Optional[set[set[Tuple[int,int,int]]]]:
		clause_left = self._get_query_dnf(prop.left, length)
		clause_right = self._get_query_dnf(prop.right, length)
		if clause_left is None:
			if clause_right is None:
				return self._cache_query_pair(prop, length, None)
			return self._cache_query_pair(prop, length, clause_right)
		if clause_right is None:
			return self._cache_query_pair(prop, length, clause_left)
		return self._cache_query_pair(prop, length, clause_left.union(clause_right))

	def _cache_query_pair(self, prop: SeqTLProp, length: int, value: Optional[set[set[Tuple[int,int,int]]]]) -> Optional[set[set[Tuple[int,int,int]]]]:
		self._query_dnf_cache[prop, length] = value 
		return value
	
	def evaluate(self, query_dnf: set[set[Tuple[int,int,int]]]):
		output = 0.0
		for clause in query_dnf:
			clause_val = 1
			for pair in clause:
				if pair not in self._query_cache:
					self._query_cache[pair] = self._oracle.compute(pair[0], self._trace, pair[1], pair[2])
				clause_val = min(1,self._query_cache[pair])
				if clause_val == 0.0:
					break
			output = max(output, clause_val)
			if output == 1.0:
				return output
		return output

	def match(self) -> float:
		query_dnf = self._get_query_dnf(self._prop, len(self._trace))
		if query_dnf is None:
			return 0.0
		return self.evaluate(query_dnf)
