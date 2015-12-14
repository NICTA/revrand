from scipy.optimize import minimize as sp_min


def minimize(fun, x0, args=(), method=None, jac=None, bounds=None,
             constraints=[], **options):
    return sp_min(fun, x0, args=args, method=method, jac=jac, bounds=bounds,
                  constraints=constraints, options=options)
