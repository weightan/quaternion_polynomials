import quaternion 
import numpy as np
import math
import time
import random
from tqdm import tqdm
from datetime import datetime
from numba import jit
from harold import *
from PIL import Image
import scipy

from matplotlib import cm

import matplotlib.pyplot as plt

from numpy import linalg as LA


def h_pol(root, fpol):

    a = root.real
    b = abs(root.imag)

    n = len(fpol)
    for h in range(n):
            fpol[h] = (quaternion.as_float_array(fpol[h])) 

    bigA = np.array( [0.0, 0.0, 0.0, 0.0] )
    bigB = np.array( [0.0, 0.0, 0.0, 0.0] )

    for i in range(n):

        if i%2 ==0:
            yt  = (-1)**(i/2)
            
            bigA += yt*(fpol[i] * b  + a*np.array([1,1,1,1]))
        else:
            yt  = (-1)**(i//2)
            bigB += yt*(fpol[i] * b  + a*np.array([1,1,1,1]) )

    bigB = (-1)*(np.quaternion(bigB[0],bigB[1],bigB[2],bigB[3]))
    #print("B ", bigB)
    #print("A ", bigA)
    C = inverse(bigA)*bigB

    return a + b * C 

def companion_matrix_for_qpol(arrq):

    #make companion matrix for left quaternion polynomial

    n = len(arrq)
    arrq = arrq[::-1]

    #chtk type of input and converting if neaded
    if type(arrq) != quaternion:
        try:
            qs = quaternion.as_quat_array(a)
        except Exception as exc:
            print(exc)
            return 0

    #arrays for  submatrices
    A1 = np.zeros((n), dtype = np.complex128)
    A2 = np.zeros((n), dtype = np.complex128)

    A1bar = np.zeros((n), dtype = np.complex128)
    A2bar = np.zeros((n), dtype = np.complex128)

    for i in range(n):
        temp = quaternion.as_float_array(arrq[i])*(-1) 
        
        A1[i] = (temp[0] + temp[1]*1j)
        A2[i] = (temp[2] + temp[3]*1j)

        A1bar[i] = (temp[0] - temp[1]*1j)
        A2bar[i] = (temp[2] - temp[3]*1j) * (-1)

    #making chunks of main matrix
    MA1 = np.diag( np.ones((n-1)), k=1).astype(complex)
    MA1[n-1:] = A1

    MA2 = np.diag( np.ones((n-1)), k=1).astype(complex)
    MA2[n-1:] = A2

    MA1b = np.diag( np.ones((n-1)), k=1).astype(complex)
    MA1b[n-1:] = A1bar

    MA2b = np.diag( np.ones((n-1)) * (-1), k=1).astype(complex) 
    MA2b[n-1:] = A2bar

    #print(MA2b)
    M = np.block([[MA1, MA2],
                  [MA2b, MA1b]])    

    return M


def inverse(q):
    if type(q) != quaternion:
            try:
                q = quaternion.as_quat_array(q)
            except Exception as exc:
                print(exc)

    return np.conj(q) / ( np.norm(q)**2 )


def make_poly_q(power, coeficients = [np.array([1,0,0,0]), np.array([0,1,0,0]),np.array([0,0,1,0]), np.array([0,0,0,1])] ):

    qpoly = np.array([random.choice(coeficients) for i in range(power)])

    #qpoly = np.array([random.choice(coeficients)*random.random() for i in range(power)]) #test

    #qpoly[0] = np.array([1,1,1,1])#/math.sqrt(4)

    return qpoly


def haroldgcd(*args):
    """
    Takes 1D numpy arrays and computes the numerical greatest common
    divisor polynomial. The polynomials are assumed to be in decreasing
    powers, e.g. :math:`s^2 + 5` should be given as ``numpy.array([1,0,5])``.

    Returns a numpy array holding the polynomial coefficients
    of GCD. The GCD does not cancel scalars but returns only monic roots.
    In other words, the GCD of polynomials :math:`2` and :math:`2s+4` is
    still computed as :math:`1`.

    Parameters
    ----------
    args : iterable
        A collection of 1D array_likes.

    Returns
    --------
    gcdpoly : ndarray
        Computed GCD of args.

    Examples
    --------
    >>> a = haroldgcd(*map(haroldpoly,([-1,-1,-2,-1j,1j],
                                       [-2,-3,-4,-5],
                                       [-2]*10)))
    >>> a
    array([ 1.,  2.])

    .. warning:: It uses the LU factorization of the Sylvester matrix.
                 Use responsibly. It does not check any certificate of
                 success by any means (maybe it will in the future).
                 I have played around with ERES method but probably due
                 to my implementation, couldn't get satisfactory results.
                 I am still interested in better methods.
    """
    raw_arr_args = [np.atleast_1d(np.squeeze(x)) for x in args]
    arr_args = [np.trim_zeros(x, 'f') for x in raw_arr_args if x.size > 0]
    dimension_list = [x.ndim for x in arr_args]

    # do we have 2d elements?
    if max(dimension_list) > 1:
        raise ValueError('Input arrays must be 1D arrays, rows, or columns')

    degree_list = np.array([x.size-1 for x in arr_args])
    max_degree = np.max(degree_list)
    max_degree_index = np.argmax(degree_list)

    try:
        # There are polynomials of lesser degree
        second_max_degree = np.max(degree_list[degree_list < max_degree])
    except ValueError:
        # all degrees are the same
        second_max_degree = max_degree

    n, p, h = max_degree, second_max_degree, len(arr_args) - 1

    # If a single item is passed then return it back
    if h == 0:
        return arr_args[0]

    if n == 0:
        return np.array([1])

    if n > 0 and p == 0:
        return arr_args.pop(max_degree_index)

    # pop out the max degree polynomial and zero pad
    # such that we have n+m columns
    S = np.array([np.hstack((
            arr_args.pop(max_degree_index),
            np.zeros((1, p-1)).squeeze()
            ))]*p)

    # Shift rows to the left
    for rows in range(S.shape[0]):
        S[rows] = np.roll(S[rows], rows)

    # do the same to the remaining ones inside the regular_args
    for item in arr_args:
        _ = np.array([np.hstack((item, [0]*(n+p-item.size)))]*(
                      n+p-item.size+1))
        for rows in range(_.shape[0]):
            _[rows] = np.roll(_[rows], rows)
        S = np.r_[S, _]

    rank_of_sylmat = np.linalg.matrix_rank(S)

    if rank_of_sylmat == min(S.shape):
        return np.array([1])
    else:
        p, l, u = scipy.linalg.lu(S)

    u[abs(u) < 1e-8] = 0
    for rows in range(u.shape[0]-1, 0, -1):
        if not any(u[rows, :]):
            u = np.delete(u, rows, 0)
        else:
            break

    gcdpoly = np.real(np.trim_zeros(u[-1, :], 'f'))
    # make it monic
    gcdpoly /= gcdpoly[0]

    return gcdpoly


class qpol:
    qzero = np.quaternion(0, 0, 0, 0)
    r_c =  [np.array([1,0,0,0]), np.array([0,1,0,0]),np.array([0,0,1,0]), np.array([0,0,0,1])] 

    def __init__(self, arrq):

        if not isinstance(arrq[0], quaternion.quaternion):
            try:
                arrq = quaternion.as_quat_array(arrq)
            except Exception as exc:
                print(exc)

        self.coef = arrq
        self.length = len(arrq)


    def __add__(self, new):

        if new.length >= self.length:
            temp1 = np.copy (new.coef )
            for i in range( self.length):
                temp1[i + new.length - self.length ] += self.coef[i]
            return qpol(temp1)
        else :
            temp2 = np.copy  (self.coef)
            for i in range( new.length):
                temp2[i + self.length - new.length ] += new.coef[i]
            return qpol(temp2)


    def __mul__(self, new):
        
        
        m = [self.qzero for i in range(new.length + self.length - 1)]

        for o1, i1 in enumerate(self.coef):
            for o2, i2 in enumerate(new.coef):
                m[o1+o2] = m[o1+o2] + (i1*i2)


        return qpol(m)
        

    def __repr__(self):
        return str(self.coef)

    def height_qp(self, mod = 'H'):
        if mod == 'H':
            arrNormCoefficients = [ np.norm( self.coef[i]/self.coef[0] ) for i in range(1, self.length) ]

            arrNormCoefficients.append(1)
            return max(arrNormCoefficients)

    def eval_qp(self, val):
        s = self.qzero
        
        if not isinstance(val, quaternion.quaternion):
            try:
                val = quaternion.as_quat_array(val)
            except Exception as exc:
                print(exc)
        
        for i in range(self.length):

            s += self.coef[i]*(val**(self.length - i - 1))

        return s

    def is_root(self, val, rtol=1e-05, atol=1e-08):

        return np.isclose(self.eval_qp(val), self.qzero, rtol=1e-05, atol=1e-08)

    def companion_matrix(self):

        return companion_matrix_for_qpol(self.coef)

    def has_spher_roots(self):
        pass

    def four_arr_from_qp(self):
        # f (t) = a(t) + ib(t) + j(c(t) − id(t))
        # returns polinoms a, b, c, d
        temparr = []
        for i in range(4):
            temp = [ self.coef[j, i] for j in range(self.length)]
            
            #while temp and temp[-1] == 0:
            #    temp.pop(-1)

            temparr.append(np.trim_zeros(np.array(temp)))

        return temparr

    def real_poly_extract(self):

        tp = [quaternion.as_float_array(self.coef[i])[0] for i in range(self.length)]
        tp = np.trim_zeros(np.array(tp))

        return tp

    def normalise(self):
        temp = [self.coef[i] for i in range(self.length)]
        while temp and temp[-1] == self.qzero:
                temp.pop(-1)

        self.coef = quaternion.as_float_array(temp)

    def make_monic(self):

        temp = [inverse(self.coef[0])*self.coef[i] for i in range(self.length)]

        self.coef = quaternion.as_float_array(temp)

    def make_ffconj(self):
        #f*conj(f), f = qpol
        fconj = [np.conjugate(self.coef[i]) for i in range(self.length)]
        t = (self * qpol(fconj))
        return t.real_poly_extract()



    def pq_roots(self, mode = 'TS'):

        q_roots = []   

        if mode == 'TS': #based on Takis Sakkalis paper

            pol_a  = self.make_four_arr_from_qp()

            gcd =  haroldgcd(pol_a[0], pol_a[1], pol_a[2], pol_a[3]) 

            ffc = self.make_ffconj()

            roots = np.sort_complex( np.roots(gcd) ) 

            roots_ffc = np.sort_complex( np.roots(ffc) ) 
            roots_by_one = []

            for i in roots_ffc:
                if not (np.conjugate(i) in roots_by_one)  and not (i in roots_by_one):
                    roots_by_one.append(i)

            for item in roots_by_one:
                q_roots.append(h_pol(item, self.coef))

        return q_roots

    def make_four_arr_from_qp(self):
        temparr = []
        qarr = [0]*self.length

        for h in range(self.length):
            qarr[h] = list(quaternion.as_float_array(self.coef[h]))

        print (qarr )

        for i in range(4):
            
            temp = [ qarr[j][i] for j in range(self.length)]
                
            #while temp and temp[-1] == 0:
            #    temp.pop(-1)

            temparr.append(np.trim_zeros(np.array(temp)))

        return temparr



def test():
    a = qpol([[1,1,1,1], [1,1,1,10]])

    b = qpol([[1,0,0,0], [0,1,0,1], [1,0,0,1]])

    c = qpol([[1,-1,0,0], [0,0,1,0]])

    #print(c.eval_qp([0,1,0,0]))
    #print(c.is_root([0,1,0,0]))
    #print(a.height_qp())
    #t = np.quaternion(1,1,1,2)

    #print(inverse(t))
    #print(np.quaternion(1,0,0,0)/t)

    #print(b * a)
    #print(a * b)
    # d = c.make_ffconj()

    print(c.pq_roots()) 


def test2():
    pass
    #print(haroldgcd([1, 0, 0, 0, -1], [1, 1]))


if __name__ == "__main__":
    test()
    #print(type(np.quaternion(0, 0, 0, 0)))
    