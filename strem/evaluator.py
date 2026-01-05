from formula import *

class Evaluator:
    def __init__(self, frame):
        self._frame = frame

    def evaluate(self, formula: SpatialFormula):
        formula_type = type(formula)
        if formula_type == AtomFormula:
            self.evaluate_atom(formula)

    def evaluate_atom(self, atom_formula: AtomFormula):
        output = 0.0
        for obj in self.frame:
            if atom_formula.atom.name == obj['class']:
                output = max(output , obj['score'])
        return output