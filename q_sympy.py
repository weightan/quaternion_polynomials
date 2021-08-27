from sympy.algebras.quaternion import Quaternion

import numpy as np
import math
import time
import random
from tqdm import tqdm
from datetime import datetime




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
        list_r = ['('+str(item)+')*x^'+ str(len(self.coef)-n-1) + '\n' for n,item in enumerate(self.coef)]
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





    


if __name__ == "__main__":

    #print(isinstance(Quaternion(1,1,1,1), Quaternion) )

    # print((Quaternion(1,1,1,1)  )

    a = qpoly([[1,1,1,1], [0,0,0,0]], 'R', 3)
    b = qpoly([[1,1,1,1], [0,0,0,0]], 'R', 2)

    print( a , '\n')
    print( b , '\n' )

    # print( a - b  , '\n')

    # print( b - a )

    # print(a * Quaternion(2,0,0,0))

    print(a * b)






