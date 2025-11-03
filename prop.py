class SeqTLProp():
	pass
		
class LengthProp(SeqTLProp):
	
	def __init__(self, lb: int, ub: int):
		self.lb = lb
		self.ub = ub
		
class UnionProp(SeqTLProp):
	
	def __init__(self, prop1: SeqTLProp, prop2: SeqTLProp):
		self.left = prop1
		self.right = prop2

class ConcatProp(SeqTLProp):
	
	def __init__(self, prop1: SeqTLProp, prop2: SeqTLProp):
		self.left = prop1
		self.right = prop2
		
class StarProp(SeqTLProp):
	
	def __init__(self, prop: SeqTLProp):
		self.child = prop
		
class RefineProp(SeqTLProp):

	def __init__(self, prop: SeqTLProp, query_id: int, lb: int, ub: int):
		self.child = prop
		self.query_id = query_id
		self.lb = lb
		self.ub = ub	


	
