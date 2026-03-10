class SeqTLProp():
	pass
		
class LengthProp(SeqTLProp):
	
	def __init__(self, lb: int, ub: int):
		self.lb = lb
		self.ub = ub

	def __str__(self):
		return "["+str(self.lb)+","+str(self.ub)+"]"
		
class UnionProp(SeqTLProp):
	
	def __init__(self, prop1: SeqTLProp, prop2: SeqTLProp):
		self.left = prop1
		self.right = prop2
		self.lb = min(prop1.lb, prop2.lb)
		self.ub = max(prop1.ub, prop2.ub)

	def __str__(self):
		return "("+self.left.__str__()+"+"+self.right.__str__()+")"

class ConcatProp(SeqTLProp):
	
	def __init__(self, prop1: SeqTLProp, prop2: SeqTLProp):
		self.left = prop1
		self.right = prop2
		self.lb = prop1.lb+prop2.lb
		self.ub = prop1.ub+prop2.ub

	def __str__(self):
		return self.left.__str__()+self.right.__str__()
		
class StarProp(SeqTLProp):
	
	def __init__(self, prop: SeqTLProp):
		self.child = prop
		self.lb = 0
		self.ub = float('inf')

	def __str__(self):
		return "("+self.child.__str__()+")*"
		
class RefineProp(SeqTLProp):

	def __init__(self, prop: SeqTLProp, query_id: int, lb: int, ub: int):
		self.child = prop
		self.query_id = query_id
		self.lb = max(lb, prop.lb)
		self.ub = min(ub, prop.ub)

	def __str__(self):
		return "("+self.child.__str__()+")^<"+str(self.lb)+","+str(self.ub)+","+str(self.query_id)+">"


	
