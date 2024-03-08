from itertools import product

from ._utils import *
from ...ensemble import predict_proba

def prune_mip_acc(
    H: Ensemble,
    X,
    y,
    w = None,
    voting: str = "hard",
    eps: float = 1.0,
    **kwargs
):
    # Get the MIP parameters:
    n, ne, nc, w, wm = get_mip_params(H, X, w)

    # Create a new gurobi model, and
    # retrieve the variables u.
    mip, u = create_base_mip(ne, **kwargs)

    # Variables:
    # v[i,k] = 1 if the sample i is classified
    # to class k, 0 otherwise.
    idx = product(range(n), range(nc))
    v = mip.addVars(idx, vtype=GRB.BINARY, name="v")

    # z[i,k,e] = 1 if the sample i is classified
    # to class k and the classifier e is used,
    # 0 otherwise.    
    idx = product(range(n), range(nc), range(ne))
    z = mip.addVars(idx, vtype=GRB.BINARY, name="z")
    
    # Constraints:
    # Each sample is classified to one class.
    for i in range(n):
        lhs = gp.LinExpr()
        for k in range(nc):
            lhs += v[i,k]
        rhs = 1
        mip.addConstr(lhs == rhs)

    # Linearization of z[i,k,e] = v[i,k] * u[e]
    # using the following constraints:
    # z[i,k,e] <= v[i,k]
    # z[i,k,e] <= u[e]
    # z[i,k,e] >= v[i,k] + u[e] - 1
    for i, k, e in product(range(n), range(nc), range(ne)):
        cons = mip.addLConstr(z[i,k,e] <= v[i,k])
        cons = mip.addLConstr(z[i,k,e] <= u[e])
        cons = mip.addLConstr(z[i,k,e] >= v[i,k]+u[e]-1)

    acc = 0
    expr = gp.LinExpr()
    for i in range(n):
        # Get the sample x and 
        # the target class t.
        x, t = X[i], y[i]
        assert (isinstance(t, int) and 0 <= t < nc)
        
        # Predicted probabilities:
        # for each classifier e,
        # for each class k.
        p = predict_proba(H, x, voting=voting)
        
        # Predicted class: l
        # by the ensemble for the sample x.
        l = np.argmax(p.dot(w))
        
        # Accuracy value:
        acc += (l == t)
        expr += (v[i,t])

        for k in range(nc):
            for kk in range(nc):
                if kk == k: continue
                lhs = gp.LinExpr()
                for e in range(ne):
                    lhs += (w[e]*(p[k,e]-p[kk,e])*z[i,k,e])
                if kk < k:
                    rhs = eps*wm*v[i,k]
                else:
                    rhs = 0.0
                
                # Add the voting constraint to the model
                # as a lazy constraint.
                cons = mip.addLConstr(lhs >= rhs)
                cons.Lazy = 1

    # Accuracy constraint:
    mip.addLConstr(expr >= acc)

    # Optimize the MIP model.
    mip.optimize()
    
    # If the MIP model is infeasible:
    if mip.status == GRB.INFEASIBLE:
        return None

    # Retrieve the solution best solution.
    return get_u(u)