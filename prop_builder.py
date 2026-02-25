from lark.tree import Tree
from prop import *
from typing import Tuple

class PropBuilder:
    def __init__(self, parsed_tree: Tree):
        self._tree = parsed_tree
        self._query_map: dict[str,int] = {}

    def build_prop(self) -> Tuple[SeqTLProp,dict[str,int]]:
        return (self._build_prop(self._tree), self._query_map)

    def _build_prop(self, parsed_tree: Tree) -> SeqTLProp:
        if len(parsed_tree.children) == 5:
            return self._build_length_prop(parsed_tree)
        if len(parsed_tree.children) == 3:
            if parsed_tree.children[1] == "|":
                return self._build_union_prop(parsed_tree)
            return self._build_prop(parsed_tree.children[1])
        if len(parsed_tree.children) == 2:
            if parsed_tree.children[1] == "*":
                return self._build_star_prop(parsed_tree)
            return self._build_concat_prop(parsed_tree)
        return self._build_refine_prop(parsed_tree)

    def _build_length_prop(self, parsed_tree: Tree) -> LengthProp:
        lb = int(parsed_tree.children[1])
        ub = int(parsed_tree.children[3])
        return LengthProp(lb, ub)

    def _build_union_prop(self, parsed_tree: Tree) -> UnionProp:
        left = self._build_prop(parsed_tree.children[0])
        right = self._build_prop(parsed_tree.children[2])
        return UnionProp(left, right)

    def _build_concat_prop(self, parsed_tree: Tree) -> ConcatProp:
        left = self._build_prop(parsed_tree.children[0])
        right = self._build_prop(parsed_tree.children[1])
        return ConcatProp(left, right)

    def _build_star_prop(self, parsed_tree: Tree) -> StarProp:
        child = self._build_prop(parsed_tree.children[0])
        return StarProp(child)

    def _build_refine_prop(self, parsed_tree: Tree) -> RefineProp:
        child = self._build_prop(parsed_tree.children[0])
        query_str = ""
        for i in range(7, len(parsed_tree.children)-1):
            query_str = query_str + parsed_tree.children[i]
        lb = int(parsed_tree.children[3])
        ub = int(parsed_tree.children[5])
        if query_str not in self._query_map:
            self._query_map[query_str] = len(self._query_map)
        return RefineProp(child, self._query_map[query_str], lb, ub)