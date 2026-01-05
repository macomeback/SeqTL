class SpatialTerm:
    pass

class Atom(SpatialTerm):
    def __init__(self, obj_type: str):
        self.obj_type = obj_type

class Var(SpatialTerm):
    def __init__(self, name: str):
        self.name = name
        pass

class Complement(SpatialTerm):
    def __init__(self, child: SpatialTerm):
        self.child = child

class Union(SpatialTerm):
    def __init__(self, left: SpatialTerm, right: SpatialTerm):
        self.right = right
        self.left = left

class Intersection(SpatialTerm):
    def __init__(self, left: SpatialTerm, right: SpatialTerm):
        self.right = right
        self.left = left