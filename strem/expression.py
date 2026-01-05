from strem.term import SpatialTerm

class MetricExpression:
    pass

class Constant(MetricExpression):
    def __init__(self, val: float):
        self.val = val

class CenterX(MetricExpression):
    def __init__(self, child: SpatialTerm):
        self.child = child

class CenterY(MetricExpression):
    def __init__(self, child: SpatialTerm):
        self.child = child

class Dist(MetricExpression):
    def __init__(self, first: SpatialTerm, second: SpatialTerm):
        self.first = first
        self.second = second

class Minus(MetricExpression):
    def __init(self, child: MetricExpression):
        self.child = child

class Exp(MetricExpression):
    def __init(self, child: MetricExpression, exponent: float):
        self.child = child
        self.exponent = exponent

class Sum(MetricExpression):
    def __init__(self, left: MetricExpression, right: MetricExpression):
        self.left = left
        self.right = right

class Mul(MetricExpression):
    def __init__(self, left: MetricExpression, right: MetricExpression):
        self.left = left
        self.right = right