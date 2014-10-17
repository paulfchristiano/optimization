import numpy as np
from numpy import linalg
from copy import copy
from numpy.linalg import norm
from ipdb import set_trace as debug

#TODO: this might not be working right when the subspace size is 0
#TODO: adjust the i=0 behavior so that convergence can be a test for done

def damped_cg( apply_A, b, x0, objective, Mdiag, lambda0, delta_lambdas,
        critratio=5e-4, gapratio = 0.1, mingap = 10, 
        maxiters=300, miniters=10, cutofftol = 1e-4,
        subspace_size=5 ):


    #debug()
#INITIALIZATION -------------------------

    apply_damped_A = lambda v : lambda0*v*Mdiag + apply_A(v)

    numL = len(delta_lambdas)
    x_best = x0
    obj_best = float('inf')
    i_best = 0
    m_best = None

    vals = []

    inext = 5
    iinc = 10
    imult = 1.3

    Cdiag = np.sqrt(Mdiag)
    Cx0 = x0 * Cdiag
    normCx0 = norm(Cx0)

    n = len(b)

    k = subspace_size if any(x0 != 0) else 0

    #NOTE: these don't necessarily have size k, if the thing converges
    U, Tu, k = KS(apply_damped_A, x0, k, Mdiag)

    UmU_proj = U

    Ax0 = apply_damped_A(x0)
    rcg = Ax0 - b
    y = rcg/Mdiag
    p = -y

    v_0 = -rcg/Cdiag
    beta0 = norm(v_0)
    v_i = normalize(v_0)

    x_unaug = np.array([x0 for l in delta_lambdas])
    x = np.array([x0 for l in delta_lambdas])
    D_im1_im1 = np.zeros(numL)
    D_i_i = np.zeros(numL)
    L_i_im1 = np.zeros(numL)
    L_ip1_i = np.zeros(numL)
    rho_im1 = np.zeros(numL)
    rho_i = np.zeros(numL)
    obj = -float('inf')*np.ones(numL)
    obj_previous = -float('inf')*np.ones(numL)

    D0_im1_im1 = 0
    L0_i_im1 = 0

    c_i = np.zeros((numL, n))
    c_im1 = np.zeros((numL, n))

    v_iU = v_i.dot(UmU_proj)


    UmU_projUmU_proj = np.dot(UmU_proj.transpose(), UmU_proj)

    val = np.zeros(numL)

    done = False
    i = 0


#MAIN LOOP----------------------------------
    while not done:


        try:
            alpha, v, T_ip1_i, p, rcg, y = cg_iter(apply_damped_A, p, rcg, y, Mdiag)
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


#CALCULATING FOR EACH LAMBDA ----------------------------

            for m in range(numL):

        #AGAINST THE (Ax - b) SPACE

                D_i_i[m] = T_i_i - L_i_im1[m]**2 * D_im1_im1[m] + delta_lambdas[m]

                L_ip1_i[m] = T_ip1_i / D_i_i[m]

                c_i[m] = v_i - L_i_im1[m]*c_im1[m]

                rho_i[m] = (
                    (0 if i else beta0) - L_i_im1[m]*D_im1_im1[m]*rho_im1[m] - delta_lambdas[m]*Cx0v_i

                ) / D_i_i[m]

                x_unaug[m] = x_unaug[m] + rho_i[m]*c_i[m] / Cdiag

        #AGAINST THE Ax SPACE

                L_W_i = T_W_i / D_i_i[m] 
                D_W_W = T_W_W + delta_lambdas[m]*np.identity(np.shape(T_W_W)[0]) - np.outer(L_W_i, L_W_i)*D_i_i[m]
                rho_W = (linalg.inv(D_W_W) if D_W_W.size else D_W_W).dot(
                    -L_W_i*D_i_i[m] * rho_i[m] - delta_lambdas[m]*WCx0mCx0_proj
                )


                x[m] = x_unaug[m] \
                        + UmU_proj[:,:k].dot(invcholUmU_proj2).dot(rho_W)/Cdiag \
                        - c_i[m] *np.dot(L_W_i, rho_W) / Cdiag

                r = np.zeros(n)
                
                r += delta_lambdas[m]*normCx0*(UmU_proj[:,0] - WWUmU_proj_1)
                r += T_ip1_i * (v_ip1 - WWv_ip1) * (
                        rho_i[m] - np.dot(L_W_i, rho_W) - v_iU[:k].dot(invcholUmU_proj2).dot(rho_W) 
                    )
                r += UmU_proj.dot(np.identity(k+1) - np.vstack([
                    invcholUmU_proj2.dot(invcholUmU_proj2.transpose()).dot(UmU_projUmU_proj[:k,:]),
                    np.zeros(k+1)
                ])).dot(Tu).dot(invcholUmU_proj2).dot(rho_W)

                val[m] = 0.5*np.dot(r*Cdiag - b, x[m])

#TESTING SOLUTIONS-------------------------------------

            vals.append(copy(val))

            testgap = int(max(np.ceil(i * gapratio), mingap))
            if i > testgap:
                prevval = vals[-testgap-1]
                if i >= miniters and all(prevval < 0) and all( (val - prevval) / val < critratio * testgap):
                    print("Stopping because of no progress!")
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
            i = i - 1
            pass

        finally:
            if i == np.ceil(inext) or i < 5 or done:
                inext = i + iinc
                for m in range(numL):
                    obj_previous[m] = obj[m]
                    obj[m] = objective(x[m])
                    record = False
                    if obj[m] < obj_best:
                        record = True
                        m_best = m
                        x_best = copy(x[m])
                        i_best = i
                        obj_best = copy(obj[m])
                    
                    print("Iteration: {i}, m = {m}, lambda({m}) = {l}, val({m})={val}, obj({m})={obj} {star}".format(
                        m=m,
                        i=i,
                        obj=obj[m],
                        val=val[m],
                        l = delta_lambdas[m] + lambda0,
                        star='*' if record else ''
                    ))
            
            if i > imult*i_best and i >= miniters and all(obj >= obj_previous):
                done=True

#UPDATES---------------------
        if not done:
            v_i = copy(v_ip1)
            v_iU = copy(v_ip1U)
            D_im1_im1 = copy(D_i_i)
            L_i_im1 = copy(L_ip1_i)
            D0_im1_im1 = copy(D0_i_i)
            L0_i_im1 = copy(L0_ip1_i)
            rho_im1 = copy(rho_i)
            c_im1 = copy(c_i)
            i += 1

    x_last = x[m_best]

    return  x_best, i_best, m_best, x_last, obj_best

#CG_ITER and KS -----------------------------------------

class Convergence(Exception):
    def __init__(self):
        pass


#Runs a single iteration of CG
#Raises Convergence if p = 0
def cg_iter( apply_A, p, rcg, y, Mdiag, tolerance=1e-10 ):
    if all (abs(p) < tolerance):
        raise Convergence()
    Cdiag = np.sqrt(Mdiag)

    Ap = apply_A(p)
    pAp = p.dot(Ap)
    assert pAp > 0, "Curvature is negative" 

    ry = rcg.dot(y)
    alpha = ry / pAp

    rcg_new = rcg + alpha*Ap
    y_new = rcg_new / Mdiag

    beta = rcg_new.dot(y_new) / rcg.dot(y)

    p_new = -y_new + beta*p

    v = rcg_new / Cdiag
    return alpha, normalize(v), norm(v) / (np.sqrt(ry)*alpha), p_new, rcg_new, y_new

#Computes the KS subspace of dimension k (or smaller if there is degeneracy) starting from x0
def KS( apply_A, x0, k, Mdiag ):
    n = len(x0)
    U = np.zeros((n, k+1))
    Tu = np.zeros((k+1, k))

    Cdiag = np.sqrt(Mdiag)
    U[:,0] = normalize(x0 * Cdiag)

    rcg = -x0*Mdiag
    y = rcg / Mdiag
    p = -y

    D0_im1_im1 = 0
    L0_i_im1 = 0

    for i in range(k):

        try:
            alpha, u, Tu[i+1,i], p, rcg, y = cg_iter(apply_A, p, rcg, y, Mdiag)
        except Convergence:
            #In this case the KS subspace collapses to dimension i, so we set k=i and return what we have so far
            k = i
            return U[:,:k+1], Tu[:k+1,:k], k


        U[:,i+1] = (-1)**(i) * u

        if i < k-1:
            Tu[i,i+1] = Tu[i+1,i]

        D0_i_i = 1 / alpha
        Tu[i,i] = D0_i_i + L0_i_im1**2 * D0_im1_im1
        L0_ip1_i = Tu[i+1,i] / D0_i_i

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

def normalize(v):
    return v / norm(v) if norm(v) > 1e-10 else v

def export_to_matlab(d):
    from scipy import io
    mdict = { key : value for key, value in d.items() if type(value) in [np.ndarray, np.float64, int, float] }
    if 'b' in d and 'apply_damped_A' in d :
        n = len(d['b'])
        mdict['A'] = reconstruct_A(d['apply_damped_A'], n)
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
