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

from matplotlib import cm

import matplotlib.pyplot as plt

from numpy import linalg as LA

#val = np.linalg.eigvals(M)

#TODO test from article, Newton method, 3d display

power = 26
coeficients = [np.array([1,0,0,0]), np.array([0,1,0,0]),np.array([0,0,1,0]), np.array([0,0,0,1])]

#inverse of a quaternion
def inverse(q):
    a = np.quaternion(q[0],q[1],q[2],q[3])
    return np.conj(a) / ( np.norm(a)**2 )


#returns array of arrays length 4
def make_poly_q(power, coeficients = [np.array([1,0,0,0]), np.array([0,1,0,0]),np.array([0,0,1,0]), np.array([0,0,0,1])] ):

    qpoly = np.array([random.choice(coeficients) for i in range(power)])

    #qpoly = np.array([random.choice(coeficients)*random.random() for i in range(power)]) #test

    #qpoly[0] = np.array([1,1,1,1])#/math.sqrt(4)

    return qpoly


# f (t) = a(t) + ib(t) + j(c(t) − id(t))
# returns polinoms a, b, c, d
def make_four_arr_from_qp(quatp):
    temparr = []
    for i in range(4):
        temp = [ quatp[j, i] for j in range(len(quatp))]
        
        while temp and temp[-1] == 0:
            temp.pop(-1)

        temparr.append(np.array(temp))

    return temparr


# returns f (t) conjugate
def make_ffconj(pol_arr):
    t = np.poly1d([0])
    #print(list(t))

    for item in pol_arr:
        p = np.poly1d(item)
        p2 = np.polymul(p,p)
        t += p2

    return t



def make_ffconj_q():
    #f*conj(f), f = qpol
    fconj = [np.conjugate(self.coef[i]) for i in range(self.length)]
    t = (self * qpol(fconj))

    return t.real_poly_extract()    



# returns coresponding root of f (t)
def h_pol(root, fpol):

    a = root.real
    b = abs(root.imag)

    n = len(fpol)

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
    



def calc_roots_of_q_poly(quat_poly_orig):

    pol_a  = make_four_arr_from_qp(quat_poly_orig)

    gcd =  haroldgcd(pol_a[0], pol_a[1], pol_a[2], pol_a[3]) 

    ffc = make_ffconj(pol_a)

    roots = np.sort_complex( np.roots(gcd) ) 

    

    roots_ffc = np.sort_complex( np.roots(ffc) ) 
    roots_by_one = []

    for i in roots_ffc:
        if not (np.conjugate(i) in roots_by_one)  and not (i in roots_by_one):
            roots_by_one.append(i)

    q_roots = []        

    for item in roots_by_one:
        q_roots.append(h_pol(item, quat_poly_orig))

    return q_roots


def eval_qp(polq, val):
    s = np.quaternion(0, 0, 0, 0)
    polq = quaternion.as_quat_array(polq)

    #print('\n', polq)

    if not isinstance(val, quaternion.quaternion):
        try:
            val = quaternion.as_quat_array(val)
        except Exception as exc:
            print(exc)
        
    for i in range(len(polq)):

        s = s + polq[i]*(val**(len(polq) - i - 1))

    return s


@jit
def log_dansity_map(val, max_count):
    max_count +=1
    brightness = math.log(val) / math.log(max_count)
    gamma = 2.2
    brightness = math.pow(brightness, 1/gamma)

    return brightness

def run(display):

    N = 4096 #4096
    scal = N * 4  #/3.7
    iters = 10_000
    #coeficients = [np.array([1,0,0,0]), np.array([0,1,0,0]),np.array([0,0,1,0]), np.array([0,0,0,1])]

    coeficients = [np.array([1,1,0,1])/math.sqrt(3), np.array([1,0,0,1])/math.sqrt(2)]
    sum = []

    grid = np.zeros((N, N), dtype = np.int32)

    for i in tqdm(range(iters)):
        a  = make_poly_q(power, coeficients)
        sum += calc_roots_of_q_poly(a)
    
    for root in sum:
        root = quaternion.as_float_array(root)
        if not ( math.isnan(root[0]) or math.isnan(root[1]) ):
            x = round(root[0] * scal + N/2)
            y = round(root[1] * scal + N/2)

            if x < N and x > 0 and y < N and y > 0 :
                grid[x, y] += 1

    display(grid, N, "0_1", coeficients)

    for root in sum:
        root = quaternion.as_float_array(root)
        if not ( math.isnan(root[0]) or math.isnan(root[2]) ):
            x = round(root[0] * scal + N/2)
            y = round(root[2] * scal + N/2)

            if x < N and x > 0 and y < N and y > 0 :
                grid[x, y] += 1

    display(grid, N, "0_2", coeficients)

    for root in sum:
        root = quaternion.as_float_array(root)
        if not ( math.isnan(root[0]) or math.isnan(root[3]) ):
            x = round(root[0] * scal + N/2)
            y = round(root[3] * scal + N/2)

            if x < N and x > 0 and y < N and y > 0 :
                grid[x, y] += 1

    display(grid, N, "0_3", coeficients)

    for root in sum:
        root = quaternion.as_float_array(root)
        if not ( math.isnan(root[1]) or math.isnan(root[2]) ):
            x = round(root[1] * scal + N/2)
            y = round(root[2] * scal + N/2)

            if x < N and x > 0 and y < N and y > 0 :
                grid[x, y] += 1

    display(grid, N, "1_2", coeficients)

    for root in sum:
        root = quaternion.as_float_array(root)
        if not ( math.isnan(root[1]) or math.isnan(root[3]) ):
            x = round(root[1] * scal + N/2)
            y = round(root[3] * scal + N/2)

            if x < N and x > 0 and y < N and y > 0 :
                grid[x, y] += 1

    display(grid, N, "1_3", coeficients)

    for root in sum:
        root = quaternion.as_float_array(root)
        if not ( math.isnan(root[3]) or math.isnan(root[3]) ):
            x = round(root[2] * scal + N/2)
            y = round(root[3] * scal + N/2)

            if x < N and x > 0 and y < N and y > 0 :
                grid[x, y] += 1

    display(grid, N, "2_3", coeficients)

def display_with_plt(grid, N, spec):

    cmap = "hot"
    size = 10

    max_count = np.max(grid)

    for i in range(N):
        for j in range(N):
            if grid[i, j]:
                grid[i, j] = 256*log_dansity_map(grid[i, j], max_count)

    plt.figure(num = None, figsize=(size, size), dpi=300)

    plt.axis('off')

    plot = plt.imshow(grid, cmap = cmap )

    ####
    now =  str(datetime.now()).replace(" ", '_')
    now =  now.replace(":", '_')

    filenameImage = f'N_{N}_cmap_{cmap}_projection_{spec}_{now}.png'

    plt.savefig(filenameImage, bbox_inches = 'tight', pad_inches=0.0)

    ####
    
    #plt.show()
    plt.close()

def display_with_pil(grid, N, spec, coeficients):
    cmap_name = "hot"
    cmap = cm.get_cmap(cmap_name)

    coef = [list(i) for i in coeficients]
    coef = [[round(j, 2) for j in i] for i in coeficients]

    aspect_ratio = 1
    width = N
    height = int(width * 1 / aspect_ratio)

    max_count = np.max(grid)
    print("Max count:", max_count)

    paint_zero_lvl = False
    mono_coloring = False

    im_arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Color image depending on density of eiginvalues for each pixel
    print("Coloring final image...")
    for y in tqdm(range(height)):
        for x in range(width):
            if grid[y, x] != 0:
                if mono_coloring:
                    im_arr[y, x] = 255
                else: 
                    rgba = cmap( log_dansity_map(grid[y, x], max_count) )
                    im_arr[y, x, 0] = int(255 * rgba[0])
                    im_arr[y, x, 1] = int(255 * rgba[1])
                    im_arr[y, x, 2] = int(255 * rgba[2])
            elif paint_zero_lvl:
                rgba = cmap( 0 )
                im_arr[y, x, 0] = int(255 * rgba[0])
                im_arr[y, x, 1] = int(255 * rgba[1])
                im_arr[y, x, 2] = int(255 * rgba[2])

    im = Image.fromarray(im_arr)
    print(f"Saving image...{width}_{height}_spec_{spec}")

    now =  str(datetime.now()).replace(" ", '_')
    now =  now.replace(":", '_')

    im.save(f'N_{N}_cmap_{cmap_name}_{coef}_projection_{spec}_{now}.png')

def test():
    
    a  = make_poly_q(power)
    b  = make_poly_q(power)

    #print(a, '\n\n', b, '\n\n')

    pol_a  = make_four_arr_from_qp(a)
    print(pol_a)

    gcd =  haroldgcd(pol_a[0], pol_a[1], pol_a[2], pol_a[3]) 

    ffc = make_ffconj(pol_a)

    roots = np.sort_complex( np.roots(gcd) ) 

    #print(ffc)

    roots_ffc = np.sort_complex( np.roots(ffc) ) 
    #print(roots_ffc)

    for item in (roots_ffc[::2]):
        print(h_pol(item, a))


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

def random_root_finding():
    arr = []

    for t in tqdm( range(1_00)):

        p_min = 10
        q_min = np.quaternion(0, 0, 0, 0)

        for i in range(1_00_000):

            x =  (1 - 2*np.random.rand(4)) * p_min

            x = q_min + np.quaternion(*x)

            
            p =  abs(test_poly_4(x)) 

            if p < p_min:
                p_min = p
                q_min = x

        if [q_min, p_min] not in arr:
            arr.append( [q_min, p_min])

    for i in arr :
        print(i)


def test2():

    a  = make_poly_q(3)
    print(a)

    r = calc_roots_of_q_poly(a)
    print(r)

    print(eval_qp(a, np.quaternion(0, 0, 1, 0)))

    #print(eval_qp(a, r[1]))


if __name__ == "__main__":

    #run(display_with_pil)

    #test()
    #test2()
    #print(test_poly_4(1.0))

    pole = quaternion.from_float_array(make_poly_q(10))




  