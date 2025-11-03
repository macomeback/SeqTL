from lark.tree import Tree
from prop import *

def build_prop(parsed_tree: Tree) -> SeqTLProp:
    if len(parsed_tree.children) == 5:
        return _build_length_prop(parsed_tree)
    if len(parsed_tree.children) == 3:
        if parsed_tree.children[1] == "|":
            return _build_union_prop(parsed_tree)
        return build_prop(parsed_tree.children[1])
    if len(parsed_tree.children) == 2:
        if parsed_tree.children[1] == "*":
            return _build_star_prop(parsed_tree)
        return _build_concat_prop(parsed_tree)
    return _build_refine_prop(parsed_tree)

def _build_length_prop(parsed_tree: Tree) -> LengthProp:
    lb = int(parsed_tree.children[1])
    ub = int(parsed_tree.children[3])
    return LengthProp(lb, ub)

def _build_union_prop(parsed_tree: Tree) -> UnionProp:
    left = build_prop(parsed_tree.children[0])
    right = build_prop(parsed_tree.children[2])
    return UnionProp(left, right)

def _build_concat_prop(parsed_tree: Tree) -> ConcatProp:
    left = build_prop(parsed_tree.children[0])
    right = build_prop(parsed_tree.children[1])
    return ConcatProp(left, right)

def _build_star_prop(parsed_tree: Tree) -> StarProp:
    child = build_prop(parsed_tree.children[0])
    return StarProp(child)

def _build_refine_prop(parsed_tree: Tree) -> RefineProp:
    child = build_prop(parsed_tree.children[0])
    query_id = int(parsed_tree.children[3])
    lb = int(parsed_tree.children[5])
    ub = int(parsed_tree.children[7])
    return RefineProp(child, query_id, lb, ub)