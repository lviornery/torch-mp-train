import numpy as np
import numpy.linalg as npl
import scipy as sp

# NOTE - all matrix/vector operations are called using np.matvel/np.subtract, etc. Only scalars should multiply ndarrays.

def finite_diff(x):
    """take the finite difference of the input vector. Performs no validation.
    :param np.ndarray(x): data to take the finite difference of -- 1-dimensional array of length n

    :return: diff(x) -- 1 dimensional array of length n-1
    """
    return np.subtract(x[1:],x[:-1])

def matrix_diff(n,dx=1.):
    """return the matrix representations of the first-order central finite first difference, first-order central second
    finite difference, and simple finite difference for a specified size and delta x increment.
    :param n: the size of the requested matrix
    :param dx: the time step

    :return:    diff -- first order central finite first difference matrix of size n with free boundary conditions
                diff2 -- first order central finite second difference matrix of size n with free boundary conditions
                d -- simple finite differnceing matrix (nxn-1)
    """
    # first central difference
    mul_step = 1/(2*dx)
    c = mul_step*np.ones(n-1)
    diff = sp.sparse.diags_array([-c,c],offsets = [-1,1]).toarray()
    diff[0,0:3] = mul_step*np.asarray([-3,4,-1])
    diff[-1,-3:] = mul_step*np.asarray([1,-4,3])

    #second central difference
    mul_step = 1/(dx**2)
    c = mul_step*np.ones(n)
    diff2 = sp.sparse.diags_array([c[0:-1],-2*c,c[0:-1]],offsets = [-1,0,1]).toarray()
    diff2[0,0:4] = mul_step*np.asarray([2, -5, 4, -1])
    diff2[-1,-4:] = mul_step*np.asarray([-1,4,-5,2])

    #first order differencing matrix
    mul_step= 1/dx
    c = mul_step*np.ones(n)
    d = sp.sparse.diags_array([-c,c],offsets=[0,1],shape=[n-1,n]).toarray()

    return diff,diff2,d

def matrix_anti_diff(n,dx=1.):
    """return the matrix representation of the antidifferentiation operator for an input of size n
    :param n: the size of the requested matrix
    :param dx: the time step

    :return:    a -- cumulative trapezoidal integration matrix that when multiplied by a vector yields the cumulative integral (without the first index) (nxn-1)
                a2 -- cumulative trapezoidal double integration matrix that when multiplied by a vector yields the cumulative double integral (without the first index) (nxn-1)
    """
    c = 0.5*dx*np.ones(n)
    a = np.add(dx*np.tri(n,k=-1),np.diag(c))
    a[:,0] = 0.5*dx
    a[0,0] = 0

    a2 = np.matmul(a,a)

    a = a[1:,:]
    a2 = a2[1:,:]

    return a,a2

def reg_diff(raw_data,alpha,alpha2=None,dx=1,u0=None,min_iter=5,max_iter=100000,delta_cost_tol=1e-6,delta_norm_tol=1e-6,ep=1e-8,diag=False):
    """differentation function that takes a vector of position data and returns the first and second derivatives, evaluated using TV-regularized differentiation.
    :param data: the input data
    :param alpha: the regularization coefficient for the first derivative
    :param alpha2: the regularization coeffiient for the second derivative. If left blank only the first derivative will be calculated.
    :param dx: the time step for the data
    :param u0: an initial guess for the derivative(s) as an ndarray or list of ndarrays
    :param min_iter: minimal number of iterations to calculate before returning. Default value 5.
    :param max_iter: maximum number of iterations to calculate before returning. Default value is 1000000.
    :param delta_cost_tol: relative delta cost to reach iteratively before returning (applies to both derivatives). Default value is 1e-6.
    :param delta_norm_tol: relative delta norm of step to reach iteratively before returning (applies to both derivatives). Default value is 1e-6.
    :param ep: epsilon to use to smooth the absolute value function. Devault value is 1e-8
    :param diag: if True, prints diagnostic information during execution. Default False.

    :return:    u -- the first derivative
                v -- the second derivative
    """
    data = np.asarray(raw_data,copy=True)
    n = len(data)
    data_mean = data.mean()
    data_st_dev = data.std()
    data = (data - data_mean)/data_st_dev

    if alpha2:
        second_deriv = True
    else:
        second_deriv = False

    diff,diff2,d = matrix_diff(n)
    d_tp = np.transpose(d)

    a,a2 = matrix_anti_diff(n)
    a_tp = np.transpose(a)
    a_tp_a = np.matmul(a_tp,a)
    a_tp_a2 = np.matmul(a_tp,a2)

    #bunch of code for handling an input u0
    u = None
    v = None
    if u0 is not None:
        if isinstance(u0,list):
            if len(u0) == n:
                u = np.asarray(u0,copy=True)*dx/data_st_dev
            elif len(u0) == 2*n:
                u = np.asarray(u0[:n],copy=True)*dx/data_st_dev
                v = np.asarray(u0[n:],copy=True)*(dx**2)/data_st_dev
            elif len(u0) == 2 and len(u0[0]) == n and len(u0[1]) == n:
                u = np.asarray(u0[0],copy=True)*dx/data_st_dev
                v = np.asarray(u0[1],copy=True)*(dx**2)/data_st_dev
        elif isinstance(u0,np.ndarray):
            if u0.shape[0] == 2 and u0.shape[1] == n:
                    u = np.asarray(u0[0,:],copy=True)*dx/data_st_dev
                    v = np.asarray(u0[1,:],copy=True)*(dx**2)/data_st_dev
            elif u0.shape[1] == 2 and u0.shape[0] == n:
                    u = np.asarray(u0[:,0],copy=True)*dx/data_st_dev
                    v = np.asarray(u0[:,1],copy=True)*(dx**2)/data_st_dev
    if not isinstance(u,np.ndarray):
        u = np.matvec(diff,data)
    if not isinstance(v,np.ndarray) and second_deriv:
        v = np.matvec(diff2,data)

    #start calculating u
    ofst_1 = data[0]
    data_comp = data[1:] - ofst_1
    a_tp_b = np.matvec(a_tp,data_comp)
    if diag:
        u_cost = [npl.norm(np.subtract(sp.integrate.cumulative_trapezoid(u),data_comp)) + alpha*npl.norm(finite_diff(u),ord=1)]
    prev_cost = None
    for iter in range(max_iter):
        c = np.reciprocal(np.sqrt(np.power(finite_diff(u),2)+ep))
        En = np.diag(c)
        Ln = np.matmul(np.matmul(d_tp,En),d)
        gn = np.add(np.subtract(np.matvec(a_tp_a,u),a_tp_b),alpha*np.matvec(Ln,u))
        Hn = np.add(a_tp_a,alpha*Ln)
        s = npl.solve(Hn,gn)
        u = np.subtract(u,s)
        cost_n = npl.norm(np.subtract(sp.integrate.cumulative_trapezoid(u),data_comp)) + alpha*npl.norm(finite_diff(u),ord=1)
        if diag:
            print("u, iteration {0}: cost: {1}, relative change: {2}".format(
                iter+1,cost_n,npl.norm(s)/npl.norm(u)
            ))
            u_cost.append(cost_n)
        if iter >= min_iter and npl.norm(s)/npl.norm(u) < delta_norm_tol:
            if diag:
                print("u, delta norm tolerance reached")
            break
        if prev_cost and iter >= min_iter and (prev_cost - cost_n)/prev_cost < delta_cost_tol:
            if diag:
                print("u, relative cost tolerance reached")
            break
        prev_cost = cost_n

    if second_deriv:
        ofst_2 = u[0]
        data_comp = np.subtract(data[1:] - ofst_1,ofst_2*np.arange(1,n))
        a_tp_b = np.matvec(a_tp,data_comp)
        if diag:
            v_cost = [npl.norm(np.subtract(sp.integrate.cumulative_trapezoid(sp.integrate.cumulative_trapezoid(v,initial=0)),data_comp)) + alpha2*npl.norm(finite_diff(v),ord=1)]
        prev_cost = None
        for iter in range(max_iter):
            c = np.reciprocal(np.sqrt(np.power(finite_diff(v), 2) + ep))
            En = np.diag(c)
            Ln = np.matmul(np.matmul(d_tp, En), d)
            gn = np.add(np.subtract(np.matvec(a_tp_a2, v), a_tp_b), alpha2 * np.matvec(Ln, v))
            Hn = np.add(a_tp_a2, alpha2 * Ln)
            s = npl.solve(Hn, gn)
            v = np.subtract(v, s)
            cost_n = npl.norm(np.subtract(sp.integrate.cumulative_trapezoid(sp.integrate.cumulative_trapezoid(v,initial=0)),data_comp)) + alpha2*npl.norm(finite_diff(v),ord=1)
            if diag:
                print("v, iteration {0}: cost: {1}, relative change: {2}".format(
                    iter+1, cost_n, npl.norm(s) / npl.norm(v)
                ))
                v_cost.append(cost_n)
            if iter >= min_iter and npl.norm(s) / npl.norm(v) < delta_norm_tol:
                if diag:
                    print("v, delta norm tolerance reached")
                break
            if prev_cost and iter >= min_iter and (prev_cost - cost_n) / prev_cost < delta_cost_tol:
                if diag:
                    print("v, relative cost tolerance reached")
                break
            prev_cost = cost_n
        v = v*data_st_dev/(dx**2)
    u = u*data_st_dev/dx
    return u, v