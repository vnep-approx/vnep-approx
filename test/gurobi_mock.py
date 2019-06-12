from gurobipy import GRB


class MockLinExpr(object):
    def __init__(self, expr):
        self.expr = frozenset(expr)

    def __eq__(self, other):
        return self.expr == other.expr

    def __hash__(self):
        return hash(self.expr)

    def __repr__(self):
        return "MockLinExpr({})".format(self.expr)


class MockConstr(object):
    def __init__(self, lhs, sense, rhs, name=""):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs
        self.name = name

    def __eq__(self, other):
        return (self.name == other.name
                and self.lhs == other.lhs
                and self.sense == other.sense
                and self.rhs == other.rhs)

    def __repr__(self):
        return "MockConstr({})".format(self.name)

    def __hash__(self):
        return hash((self.name,
                     self.lhs,
                     self.sense,
                     self.rhs))


class MockVar(object):
    def __init__(self, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="", column=None):
        self.lb = lb
        self.ub = ub
        self.obj = obj
        self.vtype = vtype
        self.name = name
        self.column = column
        self.x = 0.0

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return "MockVar({})".format(self.name)


class MockModel(object):
    def __init__(self, name):
        self.name = name
        self.vars = []
        self.constrs = []

    def addVar(self, lb=0.0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="", column=None):
        var = MockVar(lb, ub, obj, vtype, name, column)
        self.vars.append(var)
        return var

    def addConstr(self, lhs, sense, rhs, name=""):
        constr = MockConstr(lhs, sense, rhs, name)
        self.constrs.append(constr)
        return constr

    def update(self):
        pass
