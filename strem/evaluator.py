from formula import *
from term import *

class Box:
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def intersect(self, box: "Box") -> "Box":
        x_left = max(self.x-self.w/2, box.x-box.w/2)
        x_right = min(self.x+self.w/2, box.x+box.w/2)
        y_top = max(self.y-self.h/2, box.y-box.h/2)
        y_bottom = min(self.y+self.h/2, box.y+box.h/2)
        if x_right <= x_left or y_top >= y_bottom:
            return None
        center_x = (x_left+x_right)/2
        center_y = (y_top+y_bottom)/2
        return Box(center_x, center_y, x_right-x_left, y_bottom-y_top)
    
class BoxUnion:
    def __init__(self, boxes: set[Box]):
        self.boxes = boxes

    def union(self, boxUnion: "BoxUnion") -> "BoxUnion":
        union_boxes = set(self.boxes)
        union_boxes.update(boxUnion.boxes)
        return union_boxes
    
    def intersect(self, boxUnion: "BoxUnion") -> "BoxUnion":
        intersection_boxes = set()
        for box1 in self.boxes:
            for box2 in boxUnion.boxes:
                result = box1.intersect(box2)
                if result is not None:
                    intersection_boxes.add(result)
        return intersection_boxes

class Evaluator:
    def __init__(self, frame):
        self._frame = frame
        self._var_map = {}

    def eval(self, formula: SpatialFormula) -> float:
        formula_type = type(formula)
        if formula_type == AtomFormula:
            return self.eval_atom(formula.atom.name)
        if formula_type == Exists:
            return self.eval_exists(formula)
        if formula_type == Empty:
            return self.eval_empty(formula.term)
        
    def eval_term(self, term: SpatialTerm) -> BoxUnion:
        term_type = type(term)
        if term_type == Atom:
            return self.eval_atom_boxes(term.name)
        if term_type == Var:
            self._var_map[term.name] 

    def eval_atom_boxes(self, name: str):
        pass

    def eval_exists(self, exists_formula: Exists) -> float:
        output = 0.0
        var_name = exists_formula.var.name
        obj_type = exists_formula.atom.obj_type
        child = exists_formula.child
        for obj in self.frame:
            if obj['class'] == obj_type:
                self._var_map[var_name] = obj
                output = max(output, self.eval(child))
                del self._var_map[var_name]
        return output

    def eval_atom(self, name: str) -> float:
        output = 0.0
        for obj in self.frame:
            if name == obj['class']:
                output = max(output , obj['score'])
        return output