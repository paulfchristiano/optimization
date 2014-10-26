import numpy as np
from numpy import linalg
from copy import copy
from numpy.linalg import norm
from ipdb import set_trace as debug
from itertools import count

#TODO: this might not be working right when the subspace size is 0
#TODO: adjust the i=0 behavior so that convergence can be a test for done

def augmented_cg_iterates(apply_A, b, x0, U, Tu, lambdas, Mdiag, cutofftol=1e-4):

    n = len(b)
    k = Tu.shape[1]
    numL = len(lambdas)

    Cdiag = np.sqrt(Mdiag)
    Cx0 = x0 * Cdiag
    normCx0 = norm(Cx0)

    UmU_proj = U

    Ax0 = apply_A(x0)
    rcg = Ax0 - b
    y = rcg/Mdiag
    p = -y

    v_0 = -rcg/Cdiag
    beta0 = norm(v_0)
    v_i = normalize(v_0)

    x_unaug = np.array([x0 for l in lambdas])
    x = np.array([x0 for l in lambdas])
    val = np.zeros(numL)
    D_im1_im1 = np.zeros(numL)
    D_i_i = np.zeros(numL)
    L_i_im1 = np.zeros(numL)
    L_ip1_i = np.zeros(numL)
    rho_im1 = np.zeros(numL)
    rho_i = np.zeros(numL)

    D0_im1_im1 = 0.0
    L0_i_im1 = 0.0

    c_i = np.zeros((numL, n))
    c_im1 = np.zeros((numL, n))

    v_iU = v_i.dot(UmU_proj)


    UmU_projUmU_proj = np.dot(UmU_proj.transpose(), UmU_proj)

    for i in count():

        alpha, v, T_ip1_i, p, rcg, y = cg_iteration(apply_A, p, rcg, y, Mdiag)
        v_ip1 = (-1)**i * v 

        UmU_proj = UmU_proj - np.outer(v_i, v_iU)
        UmU_projUmU_proj = np.dot(UmU_proj.transpose(), UmU_proj)
        invcholUmU_proj2 = truncate_and_transform(UmU_projUmU_proj[:k,:k], lambda x : x > cutofftol, lambda x : 1 / np.sqrt(x))

        v_ip1U = np.dot(v_ip1,UmU_proj)

        T_W_W = invcholUmU_proj2.transpose().dot(
                UmU_projUmU_proj[:k,:].dot(Tu) - T_ip1_i*np.outer(v_ip1U[:k],v_iU[:k])
            ).dot(invcholUmU_proj2)
        T_W_i = invcholUmU_proj2.transpose().dot(T_ip1_i * v_ip1U[:k])

        WCx0mCx0_proj = invcholUmU_proj2.transpose().dot(UmU_projUmU_proj[:k,0])*normCx0
        Cx0v_i = np.dot(v_i, Cx0)
        WWv_ip1 = UmU_proj[:,:k].dot(invcholUmU_proj2).dot(v_ip1U[:k].dot(invcholUmU_proj2).transpose())
        WWUmU_proj_1 = UmU_proj[:,:k].dot(invcholUmU_proj2).dot(invcholUmU_proj2.transpose()).dot(UmU_projUmU_proj[:k,0])

        D0_i_i = 1/alpha
        T_i_i = D0_i_i + L0_i_im1**2 * D0_im1_im1
        L0_ip1_i = T_ip1_i / D0_i_i


        #TODO: vectorize this loop?
        for m in range(numL):

    #AGAINST THE (Ax - b) SPACE

            D_i_i[m] = T_i_i - L_i_im1[m]**2 * D_im1_im1[m] + lambdas[m]

            L_ip1_i[m] = T_ip1_i / D_i_i[m]

            c_i[m] = v_i - L_i_im1[m]*c_im1[m]

            rho_i[m] = (
                (0 if i else beta0) - L_i_im1[m]*D_im1_im1[m]*rho_im1[m] - lambdas[m]*Cx0v_i

            ) / D_i_i[m]

            x_unaug[m] = x_unaug[m] + rho_i[m]*c_i[m] / Cdiag

    #AGAINST THE Ax SPACE

            L_W_i = T_W_i / D_i_i[m] 
            D_W_W = T_W_W + lambdas[m]*np.identity(np.shape(T_W_W)[0]) - np.outer(L_W_i, L_W_i)*D_i_i[m]
            rho_W = (linalg.inv(D_W_W) if D_W_W.size else D_W_W).dot(
                -L_W_i*D_i_i[m] * rho_i[m] - lambdas[m]*WCx0mCx0_proj
            )


            x[m] = x_unaug[m] \
                    + UmU_proj[:,:k].dot(invcholUmU_proj2).dot(rho_W)/Cdiag \
                    - c_i[m] *np.dot(L_W_i, rho_W) / Cdiag

            r = np.zeros(n)
            
            r += lambdas[m]*normCx0*(UmU_proj[:,0] - WWUmU_proj_1)
            r += T_ip1_i * (v_ip1 - WWv_ip1) * (
                    rho_i[m] - np.dot(L_W_i, rho_W) - v_iU[:k].dot(invcholUmU_proj2).dot(rho_W) 
                )
            r += UmU_proj.dot(np.identity(k+1) - np.vstack([
                invcholUmU_proj2.dot(invcholUmU_proj2.transpose()).dot(UmU_projUmU_proj[:k,:]),
                np.zeros(k+1)
            ])).dot(Tu).dot(invcholUmU_proj2).dot(rho_W)

            val[m] = 0.5*np.dot(r*Cdiag - b, x[m])

        yield copy(x), copy(val)

#UPDATES---------------------
        v_i = copy(v_ip1)
        v_iU = copy(v_ip1U)
        D_im1_im1 = copy(D_i_i)
        L_i_im1 = copy(L_ip1_i)
        D0_im1_im1 = copy(D0_i_i)
        L0_i_im1 = copy(L0_ip1_i)
        rho_im1 = copy(rho_i)
        c_im1 = copy(c_i)



def run_cg( cg_iterates, objective, lambdas, critratio=5e-4, gapratio = 0.1, mingap = 10, 
    maxiters=300, miniters=10, iinc=10, imult=1.3, bail=False ):

    numL = len(lambdas)
    #TODO: handle the case where A(x0) = b correctly
    x_best = None
    obj_best = float('inf')
    i_best = 0
    m_best = None
    m_last = None
    x_last = None
    obj = -float('inf')*np.ones(numL)
    obj_previous = -float('inf')*np.ones(numL)
    vals = []
    logs = []

    inext = 0

    val = np.zeros(numL)

    done = False

    for i in count():
        try:
            x, val = next(cg_iterates)
            vals.append(val)
            testgap = int(max(i * gapratio, mingap))
            if i > testgap:
                prevval = vals[-testgap-1]
                if i >= miniters and all(prevval < 0) and all( (val - prevval) / val < critratio * testgap):
                    print("Stopping because of no [approximated] progress!")
                    done = True

            if i >= maxiters:
                print("Stopping becuase i > maxiters!")
                done = True
        #If we have already converged, then we should actually have stopped last step
        #It might be more elegant to have convergence as another test for done.
        #But note that this would cause trouble when the gradient is 0.
        except Convergence:
            print("Stopping because of convergence!")
            done = True
        finally:
            if i > inext or done:
                if bail and i >= miniters and all(obj >= obj_previous):
                    print("Bailing because of no [objective] progress!")
                    done=True
                inext = i * imult
                current_best = float('inf')
                for m in range(numL):
                    obj_previous[m] = obj[m]
                    obj[m] = objective(x[m])
                    record = False
                    if obj[m] < current_best:
                        current_best = obj[m]
                        m_last = m
                        x_last = x[m]
                    if obj[m] < obj_best:
                        record = True
                        i_best = i
                        x_best = copy(x[m])
                        m_best = m
                        obj_best = copy(obj[m])
                    
                    status = dict(
                        m=m,
                        i=i,
                        obj=obj[m],
                        val=val[m],
                        l = lambdas[m] if lambdas else "?",
                        star=' *' if record else ''
                    )
                    print("Iteration: {i}, m = {m}, lambda({m}) = {l}, val({m})={val}, obj({m})={obj}{star}".format(**status))
                    logs.append(status)
            if done:
                return  x_best, i_best, m_best, x_last, m_last, obj_best, logs


def augmented_cg( apply_A, b, x0, objective, Mdiag, lambda0, lambdas, cutofftol = 1e-4,
        subspace_size = 5, **kwargs):

    apply_damped_A = lambda v : lambda0*v*Mdiag + apply_A(v)
    delta_lambdas = np.array(lambdas) - lambda0

    if is_zero(x0):
        cg_iterates = simple_cg_iterates(apply_damped_A, b, delta_lambdas, Mdiag)
    else:

        #NOTE: these don't necessarily have size k, if the KS is degenerate
        U, Tu, k = KS(apply_damped_A, x0, subspace_size, Mdiag)

        cg_iterates = augmented_cg_iterates(apply_damped_A, b, x0, U, Tu, 
            delta_lambdas,Mdiag,cutofftol)

    return run_cg(cg_iterates, objective, lambdas, **kwargs)

#CG_ITER and KS -----------------------------------------

class Convergence(Exception):
    def __init__(self):
        pass


#Runs a single iteration of CG
#Raises Convergence if p = 0
def cg_iteration( apply_A, p, rcg, y, Mdiag, tolerance=1e-10 ):
    if all (abs(p) < tolerance):
        raise Convergence()
    Cdiag = np.sqrt(Mdiag)

    Ap = apply_A(p)
    pAp = p.dot(Ap)
    assert pAp > 0, "Curvature is negative" 

    ry = rcg.dot(y)
    alpha = ry / pAp

    rcg_new = rcg + alpha*Ap
    y_new = rcg_new/ Mdiag

    beta = rcg_new.dot(y_new)/ rcg.dot(y)

    p_new = -y_new + beta*p

    v = rcg_new/ Cdiag
    return alpha, normalize(v), norm(v)/ (np.sqrt(ry)*alpha), p_new, rcg_new, y_new

#Computes the KS subspace of dimension k (or smaller if there is degeneracy) starting from x0
def KS( apply_A, x0, k, Mdiag ):
    n = len(x0)
    U = np.zeros((n, k+1))
    Tu = np.zeros((k+1, k))

    Cdiag = np.sqrt(Mdiag)
    U[:,0] = normalize(x0 * Cdiag)

    rcg = -x0*Mdiag
    y = rcg/ Mdiag
    p = -y

    D0_im1_im1 = 0.0
    L0_i_im1 = 0.0

    for i in range(k):

        try:
            alpha, u, Tu[i+1,i], p, rcg, y = cg_iteration(apply_A, p, rcg, y, Mdiag)
        except Convergence:
            #In this case the KS subspace collapses to dimension i, so we set k=i and return what we have so far
            k = i
            return U[:,:k+1], Tu[:k+1,:k], k


        U[:,i+1] = (-1)**(i) * u

        if i < k-1:
            Tu[i,i+1] = Tu[i+1,i]

        D0_i_i = 1/ alpha
        Tu[i,i] = D0_i_i + L0_i_im1**2 * D0_im1_im1
        L0_ip1_i = Tu[i+1,i]/ D0_i_i

        D0_im1_im1 = D0_i_i
        L0_i_im1 = L0_ip1_i

    return U, Tu, k

#SMALL UTILITIES ---------------------------------

def reconstruct_A(f, n):
    rows = []
    x = np.zeros(n)
    for i in range(n):
        x[i] = 1.0
        rows.append(f(x))
        x[i] = 0.0
    return np.vstack(rows)

def is_zero(v):
    return all(v < 1e-10)

def normalize(v):
    return v/ norm(v) if not is_zero(v) else np.zeros_like(v)

def export_to_matlab(d):
    from scipy import io
    mdict = { key : value for key, value in d.items() if type(value) in [np.ndarray, np.float64, int, float] }
    io.savemat('A.mat', mdict=mdict)


#This computes the spectrum of B, filters out eigenvalues that don't meet test,
#transforms eigenvalues with transform, and then reconstitutes B
#The result is an n x k matrix, where n is the original dimension
#and k is the number of eigenvalues that pass test
#
#This is used in the above routines to cut out small eigenvalues and invert others
def truncate_and_transform(B, test , transform):
    w, v = linalg.eigh(B) if B.size else ( np.zeros(B.shape[0]), np.zeros(B.shape))
    include = test(w)
    w, v = w[include], v[:,include] if v.size else v
    D = np.diag(np.vectorize(transform)(w) if w.size else [])
    return np.dot(v, D)

#BANDED VERSION -------------------------------------

def banded_cg( apply_A, b, x0, objective, Mdiag, lambda0, lambdas, **kwargs):

    apply_damped_A = lambda v : lambda0*v*Mdiag + apply_A(v)
    delta_lambdas = np.array(lambdas)-lambda0

    if is_zero(x0):
        cg_iterates = simple_cg_iterates(apply_damped_A, b, delta_lambdas, Mdiag)
    else:
        cg_iterates = banded_cg_iterates(apply_damped_A, b, x0, delta_lambdas, Mdiag)

    return run_cg(cg_iterates, objective, lambdas, **kwargs)

#This computes the cg iterates for x0 = 0
def simple_cg_iterates(apply_A, b, lambdas, Mdiag):
    Cdiag = np.sqrt(Mdiag)
    rcg = - b
    y = rcg/Mdiag
    p = -y

    v_0 = -rcg/Cdiag
    beta0 = norm(v_0)
    v_i = normalize(v_0)

    numL = len(lambdas)
    n = len(b)
    D_i_i = np.zeros(numL)
    D_im1_im1 = np.zeros(numL)
    L_ip1_i = np.zeros(numL)
    L_i_im1 = np.zeros(numL)
    c_i = np.zeros((numL, n))
    c_im1 = np.zeros((numL, n))
    x = np.zeros((numL, n))
    val = np.zeros(numL)
    rho_i = np.zeros(numL)
    rho_im1 = np.zeros(numL)

    D0_im1_im1 = 0.0
    L0_i_im1 = 0.0

    for i in count():
        alpha, v, T_ip1_i, p, rcg, y = cg_iteration(apply_A, p, rcg, y, Mdiag)
        v_ip1 = (-1)**i * v 

        D0_i_i = 1/alpha
        T_i_i = D0_i_i + L0_i_im1**2 * D0_im1_im1
        L0_ip1_i = T_ip1_i / D0_i_i

        for m in range(numL):
            D_i_i[m] = T_i_i + lambdas[m] - L_i_im1[m]**2*D_im1_im1[m]
            L_ip1_i[m] = T_ip1_i / D_i_i[m]

            c_i[m] = v_i - L_i_im1[m]*c_im1[m]

            #this is the formula for rho_i:
            #rho_i = b*v_i; 
            #but the below "exact" version works better in practice:
            rho_i[m] = ((beta0 if i == 0 else 0) - L_i_im1[m]*D_im1_im1[m]*rho_im1[m]) / D_i_i[m]
            x[m] = x[m] + rho_i[m]*c_i[m]/Cdiag

            r = T_ip1_i * rho_i[m] * v_ip1
            val[m] = 0.5 * (r * Cdiag - b).dot(x[m])

        yield copy(x), copy(val)

        v_i = copy(v_ip1)
        D_im1_im1 = copy(D_i_i)
        L_i_im1 = copy(L_ip1_i)
        D0_im1_im1 = copy(D0_i_i)
        L0_i_im1 = copy(L0_ip1_i)
        rho_im1 = copy(rho_i)
        c_im1 = copy(c_i)


def banded_cg_iterates(apply_A, b, x0, lambdas, Mdiag):
    n = len(b)
    numL = len(lambdas)

    Cdiag = np.sqrt(Mdiag)
    v_i = b/ Cdiag
    beta1 = norm(v_i)
    v_i = v_i/ beta1
    v_ip1 = (x0 * Cdiag) - (x0*Cdiag).dot(v_i)*v_i
    beta2 = norm(v_ip1)
    v_ip1 = normalize(v_ip1)

    x = np.zeros((numL,n))
    val = np.zeros(numL)

    T_im2_i = 0.0
    T_im1_i = 0.0
    T_im1_ip1 = 0.0

    v_im1 = np.zeros(n) 
    v_im2 = np.zeros(n)

    D_i_i = np.zeros(numL)
    D_im1_im1 = np.zeros(numL)
    D_im2_im2 = np.zeros(numL)
    L_i_im1 = np.zeros(numL)
    L_i_im2 = np.zeros(numL)
    L_ip1_im1 = np.zeros(numL)
    L_ip1_i = np.zeros(numL)
    L_ip2_i = np.zeros(numL)

    rho_i = np.zeros(numL)
    rho_im1 = np.zeros(numL)
    rho_im2 = np.zeros(numL)

    c_i = np.zeros((numL,n))
    c_im1 = np.zeros((numL,n))
    c_im2 = np.zeros((numL,n))

    #TODO: use the cg iteration stuff for this?
    for i in count():
            
        v_ip2 = apply_A(v_i/ Cdiag)/ Cdiag
        
        v_ip2 = v_ip2 - T_im2_i*v_im2
        v_ip2 = v_ip2 - T_im1_i*v_im1
        
        T_i_i = v_i.dot(v_ip2)
        v_ip2 = v_ip2 - T_i_i*v_i
        T_ip1_i = v_ip1.dot(v_ip2)
        T_i_ip1 = T_ip1_i
        v_ip2 = v_ip2 - T_ip1_i*v_ip1
        
        T_ip2_i = norm(v_ip2)
        T_i_ip2 = T_ip2_i
        
        v_ip2 = v_ip2 /T_ip2_i

        
        for m in range(numL):
            D_i_i[m] = T_i_i + lambdas[m] - L_i_im1[m]**2*D_im1_im1[m] - L_i_im2[m]**2*D_im2_im2[m]

            L_ip1_i[m] = (T_ip1_i - L_ip1_im1[m]*L_i_im1[m]*D_im1_im1[m])/ D_i_i[m]
            L_ip2_i[m] = T_ip2_i/ D_i_i[m]

            c_i[m] = v_i - L_i_im1[m]*c_im1[m] - L_i_im2[m]*c_im2[m]

            #this is the formula for rho_i:
            #rho_i = b'*v_i 
            #but the below "exact" version works better in practice:
            rho_i[m] =  ((beta1 if i == 0 else 0)- L_i_im1[m]*D_im1_im1[m]*rho_im1[m] 
                    - L_i_im2[m]*D_im2_im2[m]*rho_im2[m])/ D_i_i[m]

            x[m] = x[m] + rho_i[m]*(c_i[m]/Cdiag)

            r = v_ip1*T_im1_ip1*(rho_im1[m] - L_i_im1[m]*rho_i[m]) + (v_ip2*T_ip2_i + T_ip1_i*v_ip1)*rho_i[m]
            r = r * Cdiag
            #rexact = Afunc(x) + lambda*M*x - b

            val[m] = 0.5*(r - b).dot(x[m])

        yield copy(x), copy(val)
  
#UPDATES---------------------
        v_im2, v_im1 = copy(v_im1), copy(v_i)
        v_i, v_ip1 = copy(v_ip1), copy(v_ip2) 
        T_im2_i, T_im1_ip1 = copy(T_im1_ip1), copy(T_i_ip2)
        T_im1_i = copy(T_i_ip1)
        
        D_im2_im2, D_im1_im1 = copy(D_im1_im1),copy(D_i_i)
        L_i_im2,L_ip1_im1 =copy(L_ip1_im1), copy(L_ip2_i)
        L_i_im1 = copy(L_ip1_i)
        
        c_im2, c_im1 = copy(c_im1), copy(c_i)
        rho_im2, rho_im1 = copy(rho_im1), copy(rho_i)
