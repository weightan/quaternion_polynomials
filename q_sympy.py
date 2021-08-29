from sympy.algebras.quaternion import Quaternion
from sympy import gcd_list
from sympy import Poly, Symbol

import numpy as np
import math
import time
import random
from tqdm import tqdm
from datetime import datetime


import itertools

def test_poly_3(root):

    root_1 = np.quaternion(0.0288237420701812, 0.0671329249043109, 0.544110244443226, 0.386948956748993)
    root_2 = np.quaternion(0.283796939082491, -0.792369984966505, -0.346661052571272, -1.32022604623824)
    root_3 = np.quaternion(-1.31262068115267, -0.105283665870052, -0.841276723700539, -0.379737031029315)

    roots = [root_1,root_2,root_3]

    P = lambda x: (np.quaternion(1, 0, 0, 0))  * (x**3 ) + (np.quaternion(1, 1, 1, 1))  * (x**2 ) +(np.quaternion(0, 1, 0, 1))     * x  + (np.quaternion(1, 1, 1, 0))

    return  P(root)

def test_poly_4(root):

    root_1 = np.quaternion(0.441280301959769, -0.732324944990305, 0.458025225813866, 0.169887996457338)
    root_2 = np.quaternion(-0.57748871019493, 0.275483416614018, 0.35235181837567, 0.0478157093575365)
    root_3 = np.quaternion(0.647435400221118, -1.02360532052823, 0.0629605174330259, -1.22188441077845)
    root_4 = np.quaternion(-0.511227015048595, -0.847149525894921, -0.707648090220202, -0.684329963550703)

    roots = [root_1,root_2,root_3,root_4]

    P = lambda x: (np.quaternion(1, 0, 0, 0))*(x**4 ) + (np.quaternion(0, 1, 0, 1))*(x**3 ) + (np.quaternion(1, 0, 0, 1))*(x**2 ) +(np.quaternion(0, 1, 1, 1))*x  + (np.quaternion(1, 1, 1, 0))

    return P(root)


class qpoly:
    q_zero = Quaternion(0, 0, 0, 0)
    q_one = Quaternion(1, 0, 0, 0)

    def __init__(self, arr, mode = 'Q', degree = 3):

        if mode == 'Q':
            self.coef = arr

        elif mode == 'L':
            self.coef = [Quaternion(*i) for i in arr]

        elif mode == 'R':
            if not isinstance(arr[1], Quaternion ) :
                self.coef = [Quaternion(*random.choice(arr)) for i in range(degree + 1)]
            else :
                self.coef = [random.choice(arr) for i in arr]

    def __str__(self):
        list_r = ['('+str(item)+')*x**'+ str(len(self.coef)-n-1) + '\n' for n,item in enumerate(self.coef)]
        return ''.join(list_r)

    def __add__ (self, other):
        if  isinstance(other, Quaternion ) :
            temp_list = other.coef.copy()

            temp_list[-1] = temp_list[-1] + other

            return qpoly(temp_list)

        elif  isinstance(other, qpoly) :
            len1 = len(self.coef) 
            len2 = len(other.coef) 

            if len1 >= len2:
                temp_list = self.coef.copy()

                for i in range(1, len1  + 1):
                    
                    temp_list[len1 - i] = temp_list[len1 - i] + other.coef[len2 - i]

                return qpoly(temp_list)

            else:
                temp_list = other.coef.copy()

                for i in range(1, len2 + 1):
                    
                    temp_list[len2 - i] = temp_list[len2 - i] + self.coef[len1 - i]

                return qpoly(temp_list)

    def __sub__ (self, other):
        if  isinstance(other, Quaternion ) :
            temp_list = other.coef.copy()

            temp_list[-1] = temp_list[-1] - other

            return qpoly(temp_list)

        elif  isinstance(other, qpoly) :
            len1 = len(self.coef) 
            len2 = len(other.coef) 

            if len1 >= len2:
                temp_list = self.coef.copy()

                for i in range(1, len1  + 1):
                    
                    temp_list[len1 - i] = temp_list[len1 - i] - other.coef[len2 - i]

                return qpoly(temp_list)

            else:
                temp_list = other.coef.copy()

                for i in range(1, len2 + 1):
                    
                    temp_list[len2 - i] =  self.coef[len1 - i] -  temp_list[len2 - i] 

                return qpoly(temp_list)

    def __mul__ (self, other): 

        if  isinstance(other, Quaternion ) :

            temp_list = [ item * other for item in self.coef]

            return qpoly(temp_list)

        elif  isinstance(other, qpoly) :
            len1 = len(self.coef) 
            len2 = len(other.coef) 

            temp_list = [Quaternion(0, 0, 0, 0)]*(len1+len2 - 1)

            for t in range(len1+ len2 - 2, -1, -1):
                c = Quaternion(0, 0, 0, 0)

                for i in range(0, t+1):
                    if i < len1 and t - i < len2:
                        c  = c + self.coef[i] * other.coef[t - i]

                temp_list[t] = c

            return qpoly(temp_list)

    def eval_at(self, x):
        c = Quaternion(0, 0, 0, 0)

        for n, q in enumerate(self.coef):

            power = len(self.coef) - n - 1
            #print(power)
            c = c + q*(x**( power))

        return c

    def has_no_spherical_roots(self):

        return self.gcd_of_four_pol() == 1

    def extract_four_pol(self):

        fc = [item._a for item in self.coef]
        fi = [item._b for item in self.coef]
        fj = [item._c for item in self.coef]
        fk = [item._d for item in self.coef]

        #print(fk)

        return [fc, fi, fj, fk]

    def gcd_of_four_pol(self):

        four_pol = self.extract_four_pol()

        x = Symbol('x')

        arr = [Poly(item, x).as_expr() for item in four_pol]

        #print(arr)

        return gcd_list(arr)

    def conjugate(self):

        return [item._eval_conjugate() for item in self.coef]

    def F_characteristic_poly(self):

        pass



def test():

    #print(isinstance(Quaternion(1,1,1,1), Quaternion) )

    # print((Quaternion(1,1,1,1)  )

    a = qpoly([[1,1,1,1], [0,0,0,0]], 'R', 4)
    b = qpoly([[1,1,1,1], [0,0,0,0]], 'R', 2)

    # print( a , '\n')
    # print( b , '\n' )

    # print( a - b  , '\n')

    # print( b - a )

    # print(a * Quaternion(2,0,0,0))

    #print(a * b)

    # x = Symbol('x')
    # print(Poly([1,2,3,4,5,6], x ))

    c = 0
    n = 100
    arr_res = []

    p = list(itertools.product([0,1],[0,1],[0,1],[0,1]))

    for p1, p2 in itertools.combinations(p, 2):
        c = 0

        for i in range(n):
            a = qpoly([p1, p2], 'R', 20)

            if a.gcd_of_four_pol() != 1 :
                c += 1

        arr_res.append([p1, p2, c, c/n ])

    arr_res = sorted(arr_res, key = lambda x: x[3])

    for i in arr_res:
        print(i)

def test2():
    a = qpoly([[1,1,1,1], [0,0,0,0]], 'R', 3)

    xp = a.eval_at( Quaternion(1,1,1,1) )

    print(xp)




if __name__ == "__main__":

    
    test()





