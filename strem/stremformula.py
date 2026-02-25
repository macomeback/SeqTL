from strem.term import Atom, Var, SpatialTerm
from strem.expression import MetricExpression

class SpatialFormula:
    pass

class AtomFormula(SpatialFormula):
    def __init__(self, atom: Atom):
        self.atom = atom

class Exists(SpatialFormula):
    def __init__(self, var: Var, atom: Atom, child: SpatialFormula):
        self.var = var
        self.atom = atom
        self.child = child

class Empty(SpatialFormula):
    def __init__(self, term: SpatialTerm):
        self.term = term

class Inclusion(SpatialFormula):
    def __init__(self, left: SpatialFormula, right: SpatialFormula):
        self.left = left
        self.right = right

class Not(SpatialFormula):
    def __init__(self, child: SpatialFormula):
        self.child = child

class And(SpatialFormula):
    def __init__(self, left: SpatialFormula, right: SpatialFormula):
        self.left = left
        self.right = right

class Or(SpatialFormula):
    def __init__(self, left: SpatialFormula, right: SpatialFormula):
        self.left = left
        self.right = right

class LEQ(SpatialFormula):
    def __init__(self, left: MetricExpression, right: MetricExpression):
        self.left = left
        self.right = right