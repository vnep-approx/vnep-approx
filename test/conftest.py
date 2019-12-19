import gurobipy
import pytest

import gurobi_mock


@pytest.fixture
def mock_gurobi(monkeypatch):
    monkeypatch.setattr(gurobipy, "LinExpr", gurobi_mock.MockLinExpr)
    monkeypatch.setattr(gurobipy, "Model", gurobi_mock.MockModel)


@pytest.fixture
def import_gurobi_mock():
    return gurobi_mock
