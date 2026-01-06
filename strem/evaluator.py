from strem.formula import *
from strem.term import *
from strem.expression import *
import shapely
from shapely.geometry import Polygon
    
def obj_to_box(obj):
        center = obj['bbox']['region']['center']
        dims = obj['bbox']['region']['dimensions']
        x = center['x']
        y = center['y']
        w = dims['w']
        h = dims['h']
        return shapely.box(x-w/2,y-h/2,x+w/2,y+h/2)

class Evaluator:
    def __init__(self, frame):
        self._frame = frame
        self._universe = self.build_universe(frame['image']['dimensions'])
        self._var_map = {}

    def build_universe(self, dims):
        w = dims['width']
        h = dims['height']
        return shapely.box(0, 0, w, h)

    def eval(self, formula: SpatialFormula) -> bool:
        formula_type = type(formula)
        if formula_type == AtomFormula:
            return self.eval_atom(formula.atom.name)
        if formula_type == Exists:
            return self.eval_exists(formula)
        if formula_type == Empty:
            return self.eval_term(formula.term).is_empty
        if formula_type == Inclusion:
            return self.eval_inclusion(formula)
        if formula_type == Not:
            return not self.eval(formula.child)
        if formula_type == And:
            return self.eval(formula.left) and self.eval(formula.right)
        if formula_type == Or:
            return self.eval(formula.left) or self.eval(formula.right)
        return self.eval_exp(formula.left) <= self.eval_exp(formula.right)

    def eval_exp(self, exp: MetricExpression) -> float:
        exp_type = type(exp)
        if exp_type == Constant:
            return exp.val
        if exp_type == CenterX:
            return shapely.centroid(self.eval_term(exp.child)).x
        if exp_type == CenterY:
            return shapely.centroid(self.eval_term(exp.child)).y
        if exp_type == Dist:
            return self.eval_distance(exp)
        if exp_type == Minus:
            return -self.eval_exp(exp.child)
        if exp_type == Sum:
            return self.eval_exp(exp.left)+self.eval_exp(exp.right)
        if exp_type == Mul:
            return self.eval_exp(exp.left)*self.eval_exp(exp.right)
        return self.eval_exp(exp.child)**exp.exponent
        
    def eval_distance(self, exp: Dist):
        center1 = self.eval_term(exp.first)
        center2 = self.eval_term(exp.second)
        return shapely.distance(center1, center2)
        
    def eval_inclusion(self, formula: Inclusion) -> bool:
        left = self.eval(formula.left) 
        right = self.eval(formula.right)
        return left.difference(right).is_empty()
        
    def eval_term(self, term: SpatialTerm):
        term_type = type(term)
        if term_type == Atom:
            return self.eval_atom_boxes(term.obj_type)
        if term_type == Var:
            return obj_to_box(self._var_map[term.name])
        if term_type == Complement:
            return self._universe.difference(self.eval_term(term.child))
        if term_type == Union:
            return shapely.union(self.eval_term(term.left), self.eval_term(term.right))
        return shapely.intersection(self.eval_term(term.left), self.eval_term(term.right))
        
    def eval_atom_boxes(self, name: str):
        shape = Polygon()
        for obj in self._frame['annotations']:
            if obj['class'] == name: 
                shape = shapely.union(shape, obj_to_box(obj))
        return shape

    def eval_exists(self, exists_formula: Exists) -> bool:
        var_name = exists_formula.var.name
        obj_type = exists_formula.atom.obj_type
        child = exists_formula.child
        for obj in self._frame:
            if obj['class'] == obj_type:
                self._var_map[var_name] = obj
                result = self.eval(child)
                del self._var_map[var_name]
                if result == True:
                    return True
        return False

    def eval_atom(self, name: str) -> bool:
        for obj in self._frame:
            if name == obj['class']:
                return True
        return False