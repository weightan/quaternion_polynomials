import quaternion 
import numpy as np
import math
import time
import random
from pol_div import *
import cmath
from tqdm import tqdm
import matplotlib.pyplot as plt

from numpy import linalg as LA

iterations = 1_0_000


maxV = 280
N = 2000
cmap = 'hot'

scal = 6
dt = 0.02
size = 10

rootsPr = []

def log_dansity_map(val, max_count):

    brightness = math.log(val) / math.log(max_count)
    gamma = 2.2
    brightness = math.pow(brightness, 1/gamma)

    return brightness

def makeCompM(arrq, n):
    
    for i in range(n):
        temp = quaternion.as_float_array(arrq[i])
        qval = temp[0] + math.sqrt(temp[1]**2 +temp[2]**2 +temp[3]**2)*1j
        arrq[i] = qval


    M = np.diag( np.ones((n-1)), k=1).astype(complex)
    M[n-1:] = arrq

    return M

def makeBetterCompM(arrq, n):
    
    arrq = arrq[::-1]
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

def test():
    start_main = time.time()

    for i in range(1_0_000):

        ex = np.quaternion(random.randrange(10), random.randrange(10), random.randrange(10), random.randrange(10))
        ex2 =  np.quaternion(-random.randrange(10), -random.randrange(10), -random.randrange(10), -random.randrange(10))

        a1 = ex/ex2
        a2 = ex2/ex
        a3 = a1*a2*a2*a1

    end_main = time.time()

    secs = end_main - start_main
    print("Main took ", secs)

def run():
    coefficints = [np.quaternion(0, 0, 0, 0),
                   np.quaternion(1, 1, 1, 1 )]
    nm = 10

    coef = np.zeros((N, N, N), dtype = np.int32)

    for i in range(iterations):#tqdm(
        roots = []
        num = [  random.choice(coefficints) for i in range(nm) ]
        M = makeCompM(num, nm)
        val, wec = LA.eig(M)

        trace = np.array([(np.conj(val[i]) + val[i]).real for i in range(len(val))])
        norm = np.array([(np.conj(val[i]) * val[i]).real for i in range(len(val))])

        for j in range(nm):
            
            den = [np.quaternion(1,0,0,0), np.quaternion(-trace[j],0,0,0), np.quaternion(norm[j],0,0,0) ]
            
            print(trace[j], norm[j])

            quot, rem = poly_divmod(num, den)
            
            if type(rem[0]) is quaternion and rem[0] == np.quaternion(0, 0, 0, 0):

                r0 = trace[j]/2 + math.sqrt(norm[j] - 0.25*trace[j]**2)*1j
                root = [ r0.real, r0.imag]
                try:
                    x = round(root[0] * N/8.8 + N/2)
                    y = round(root[1] * N/8.8 + N/2)
                    z = round(root[2] * N/8.8 + N/2)

                except Exception as inst:
                    pass

                if x < N and x > 0 and y < N and y > 0 and z < N and z > 0 and coef[x, y, z] < maxV:
                    coef[x, y, z] += 1
            elif len(rem) >= 2:

                f = rem[0]
                g = rem[1]
                #print(g)
                x = -1*g / f

                #print(x)
                root = quaternion.as_float_array(x)

                try:
                    x = round(root[0] * N/8.8 + N/2)
                    y = round(root[1] * N/8.8 + N/2)
                    z = round(root[2] * N/8.8 + N/2)
                    #print(root)
                except Exception as inst:
                    pass
                if x < N and x > 0 and y < N and y > 0 and z < N and z > 0 and coef[x, y, z] < maxV:
                    coef[x, y, z] += 1
        """   
        for j in roots:
            #if type(j) is list:
            #print(j)
            try:
                x = round(j[0] * N/8.8 + N/2)
                y = round(j[1] * N/8.8 + N/2)
            except Exception as inst:
                pass


            if x < N and x > 0 and y < N and y > 0 and coef[x, y] < maxV:
                coef[x, y] += 1
                """ 
        
    filenameArr = f'N_{N}_cmap_{cmap}_maxV_{maxV}_{random.randrange(10000000, 100000000)}'
    np.save(filenameArr, coef)

    ####
    """
    for i in range(N):
        for j in range(N):
            if coef[i, j]:
                coef[i, j] += 700 
    ####
    

    plt.figure(num = None, figsize=(size, size), dpi=300)

    plt.axis('off')

    plot = plt.imshow(coef[], cmap = cmap, interpolation='lanczos' )

    ####

    filenameImage = f'N_{N}_cmap_{cmap}_maxV_{maxV}_{random.randrange(10000000, 100000000)}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight', pad_inches=0.0)

    ####

    #plt.show()
    plt.close()
    """

def runTestForAll():
    rtr = 0
    coefficints = [np.quaternion(1, 0, dt, 0),
                   np.quaternion(-1, -dt, -dt, -1)]
    nm = 27

    coef = np.zeros((N, N), dtype = np.int32)

    for i in tqdm(range(iterations)):
        roots = []
        num = [  random.choice(coefficints) for i in range(nm) ]
        
        M = makeBetterCompM(num, nm)

        num.insert(0, np.quaternion(1, 0, 0, 0))

        val = np.linalg.eigvals(M)

        val = np.sort_complex(val) [::2]

        trace = np.array([(np.conj(val[i]) + val[i]).real for i in range(len(val))])
        norm = np.array([(np.conj(val[i]) * val[i]).real for i in range(len(val))])

        for j in range(nm):
            
            den = [np.quaternion(1,0,0,0), np.quaternion(-trace[j],0,0,0), np.quaternion(norm[j],0,0,0) ]
            quot, rem = poly_divmod(num, den)
            
            if type(rem[0]) is quaternion and rem[0] == np.quaternion(0, 0, 0, 0):
                pass
                """
                r0 = trace[j]/2 + math.sqrt(norm[j] - 0.25*trace[j]**2)*1j
                root = [ r0.real, r0.imag]
                try:
                    x = round(root[0] * N/scal + N/2)
                    y = round(root[1] * N/scal + N/2)
                    if x < N and x > 0 and y < N and y > 0:
                        coef[x, y] += 1

                except Exception as inst:
                    pass
                """
                

            elif len(rem) >= 2:

                f = rem[0]
                g = rem[1]
                
                x = -1* g*(1 / f) 

                
                root = quaternion.as_float_array(x)
                """
                if(rtr < 10) and (root[2] !=0):
                    rtr +=1
                    rootsPr.append(root)
                """

                try:
                    x = round(root[0] * N/scal + N/2)
                    y = round(root[1] * N/scal + N/2)
                    if x < N and x > 0 and y < N and y > 0 :
                        coef[x, y] += 1
                    
                except Exception as inst:
                    pass
                
        """   
        for j in roots:
            #if type(j) is list:
            #print(j)
            try:
                x = round(j[0] * N/8.8 + N/2)
                y = round(j[1] * N/8.8 + N/2)
            except Exception as inst:
                pass


            if x < N and x > 0 and y < N and y > 0 and coef[x, y] < maxV:
                coef[x, y] += 1
        """ 
        
    #filenameArr = f'N_{N}_cmap_{cmap}_maxV_{maxV}_{random.randrange(10000000, 100000000)}'
    #np.save(filenameArr, coef)

    ####
    max_count = np.max(coef)
    for i in range(N):
        for j in range(N):
            if coef[i, j]:
                coef[i, j] = 256*log_dansity_map(coef[i, j], max_count) 
                #coef[i, j] += 100
    ####
    

    plt.figure(num = None, figsize=(size, size), dpi=300)

    plt.axis('off')

    plot = plt.imshow(coef, cmap = cmap )

    ####

    filenameImage = f'N_{N}_cmap_{cmap}_maxV_{maxV}_{random.randrange(10_000_000, 100_000_000)}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight', pad_inches=0.0)

    ####
    
    #plt.show()
    plt.close()


def testCompM():
    P = [np.quaternion(1, 0, 0, 0),
         np.quaternion(1, 2, -4, 0),
         np.quaternion(0, -3.1, 0, -1),
         np.quaternion(0, 0, 2.5, 2.1),
         np.quaternion(3, -1, 0, 0),

         np.quaternion(-1.7, 0, 0, 0),
         np.quaternion(0, -1, -1, 0),
         np.quaternion(-7.2, 0, 0, 0),
         np.quaternion(0, 0, 0, 0),
         np.quaternion(0, 0, -1, 0),

         np.quaternion(-4, 0, 2.9, -2.9)]

    #P = P[1:]
    #P = P[::-1]
    print(P, '\n')

    P2 = P[1:]


    M = makeBetterCompM(P2, len(P2))
    #M = makeBetterCompM(P2, 3)
    #print(M)
    val, wec = LA.eig(M)
    val = sorted(val, key = lambda item: item.real) [::2]

    trace = np.array([(np.conj(val[i]) + val[i]).real for i in range(len(val))])
    norm = np.array([(np.conj(val[i]) * val[i]).real for i in range(len(val))])

    den = [np.quaternion(1,0,0,0), np.quaternion(-trace[1],0,0,0), np.quaternion(norm[1],0,0,0) ]
    quot, rem = poly_divmod(P, den)

    f = rem[0]
    g = rem[1]
                
    x = -1* g * (1 / f) 



    print(val, '\n')
    print(trace, '\n')
    print(norm, '\n')

    print(x,  '\n')
    #print(val)






if __name__ == "__main__":
    #run()
    #testCompM()
    runTestForAll()


    















