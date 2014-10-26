import numpy as np

#TODO: implement more intelligent line searches
#TODO: implement line searches that leverage available 2nd order info

def backtracking_linesearch(f, x, p, fx, gx, rho = 0.75):
    gxp = np.dot(gx, p)
    alpha = 1
    bound = lambda a : fx + 0.5 * a * gxp
    while True:
        fp = f(x + alpha * p)
        if fp < bound(alpha):
            return alpha
        else:
            alpha = rho * alpha
