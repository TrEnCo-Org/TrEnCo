# NumPy.
import numpy as np

# Gurobi.
import gurobipy as gp
from gurobipy import GRB

from ...ensemble import Ensemble

def set_gurobi_params(
    m: gp.Model,
    **kwargs
):
    # This function sets the parameters of
    # a Gurobi model m. The parameters are
    # passed as keyword arguments:
    # - verbose: bool, default False
    # - timelimit: float, default 3600

    verbose = kwargs.get("verbose", False)
    timelimit = kwargs.get("timelimit", 3600)

    m.setParam("OutputFlag", verbose)
    m.setParam("TimeLimit", timelimit)

def get_mip_params(
    H: Ensemble,
    X,
    w = None
):
    n = len(X) # Number of samples.
    ne = len(H) # Number of estimators.
    nc = H.n_classes_ # Number of classes.

    # For now we only support single output
    # classification.
    assert (isinstance(nc, int) and nc > 0)
    
    # Check the weights:
    # If the weights are not provided,
    # we assume that all estimators have
    # the same weight.
    # Otherwise, we check that the weights
    # are provided and that they are
    # all positive and have the same length
    # as the number of estimators.
    if w is None:
        w = np.ones(ne)
    else:
        w = np.array(w)
    assert len(w) == ne

    wm = w.min() # Minimum weight.
    assert wm > 0
    
    return n, ne, nc, w, wm

def create_base_mip(ne: int, **kwargs):
    # This function creates a base Gurobi model
    # for the MIP pruning algorithms.
    # The function returns the model and the
    # variables u.
    
    # Create a new gurobi model.
    mip = gp.Model("MIP")

    # Set Gurobi parameters:
    set_gurobi_params(mip, **kwargs)

    # Variables:
    # u[e] = 1 if the estimator e is used,
    # 0 otherwise.
    u = mip.addVars(ne, vtype=GRB.BINARY, name="u")
    
    # Objective function:
    obj = gp.quicksum(u)
    mip.setObjective(obj, GRB.MINIMIZE)

    # Return the model and the variables u.
    return mip, u

def get_u(u: gp.tupledict[int, gp.Var]):
    # This function returns the values of the
    # variables u as binary numpy array.
    n = len(u)
    v = np.array([u[i].X for i in range(n)])
    return (v >= 0.5).astype(int)