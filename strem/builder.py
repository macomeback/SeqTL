from formula import *
from expression import *
from term import *

def build_formula(tree) -> SpatialFormula:
    if "Token" in str(type(tree.children[0])):
        first_token_type = tree.children[0].type 
        if first_token_type== "ATOM":
            return AtomFormula(Atom(tree.children[0][1:-1]))
        if first_token_type == "OPENPAR":
            return build_formula(tree.children[1])
        if first_token_type  == "EXISTS":
            var = build_term(tree.children[1])
            atom = build_term(tree.children[3])
            child = build_formula(tree.children[5])
            return Exists(var, atom, child)
        if first_token_type == "EMP":
            return Empty(build_term(tree.children[2]))
        if first_token_type == "SUBSET":
            return Inclusion(build_term(tree.children[2]), build_term(tree.children[4]))
        if first_token_type == "NOT":
            return Not(build_formula(tree.children[1]))
        return LEQ(build_exp(tree.children[2]), build_exp(tree.children[4]))
    if tree.children[1].type == "AND":
        return And(build_formula(tree.children[0]), build_formula(tree.children[2]))
    return Or(build_formula(tree.children[0]), build_formula(tree.children[2]))

def build_exp(tree) -> MetricExpression:
    if "Token" in str(type(tree.children[0])):
        first_token_type = tree.children[0].type
        if first_token_type == "OPENPAR":
            return build_exp(tree.children[1])
        if first_token_type == "SIGNED_NUMBER":
            return Constant(float(tree.children[0]))
        if first_token_type == "X":
            return CenterX(build_term(tree.children[2]))
        if first_token_type == "Y":
            return CenterY(build_term(tree.children[2]))
        if first_token_type == "DIST":
            return Dist(build_term(tree.children[2]), build_term(tree.children[4]))
        return Minus(build_exp(tree.children[1]))
    second_token_type = tree.children[1].type
    if second_token_type == "SUM":
        return Sum(build_exp(tree.children[0]), build_exp(tree.children[2]))
    if second_token_type == "MUL":
        return Mul(build_exp(tree.children[0]), build_exp(tree.children[2]))
    return Exp(build_exp(tree.children[0]), build_exp(tree.children[2]))
        

def build_term(tree) -> SpatialTerm:
    if "Token" in str(type(tree.children[0])):
        first_token_type = tree.children[0].type
        first_token_val = tree.children[0].val
        if first_token_type == "OPENPAR":
            return build_term(tree.children[1])
        if first_token_type == "ATOM":
            return Atom(first_token_val)
        if first_token_type == "VAR":
            return Var(first_token_val)
        return Complement(build_term(tree.children[2]))
    if tree.children[1].type == "UNION":
        return Union(build_term(tree.children[0]), build_term(tree.children[2]))
    return Intersection(build_term(tree.children[0]), build_term(tree.children[2]))
        