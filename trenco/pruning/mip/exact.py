from ._utils import *
from ...ensemble import predict_proba

def prune_mip_exact(
    H: Ensemble,
    X,
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

    # For each sample:
    for i in range(n):
        # Get the sample x.
        x = X[i]

        # Reshape the sample x.
        # such that it has a single row.
        # and can be used as input to the
        # estimators.
        x = x.reshape(1, -1)
        
        # Predicted probabilities
        # for each classifier e,
        # for each class k.
        p = predict_proba(H, x, voting=voting)

        # Predicted class: l
        # by the ensemble for the sample x.
        l = np.argmax(p.dot(w))

        # For each class k:
        for k in range(nc):
            if k == l: continue
            lhs = gp.LinExpr()
            for e in range(ne):
                lhs += (w[e]*(p[l,e]-p[k,e])*u[e])
            if k < l:
                rhs = eps*wm
            else:
                rhs = 0.0
                
            # Add the voting constraint to the model
            # as a lazy constraint.
            cons = mip.addLConstr(lhs >= rhs)
            cons.Lazy = 1

    # Optimize the model.
    mip.optimize()
    
    # If the model is infeasible,
    # return None.
    if mip.status == GRB.INFEASIBLE:
        return None

    return get_u(u)
