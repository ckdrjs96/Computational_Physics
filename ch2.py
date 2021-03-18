



## module LUdecomp
''' 
    a = LUdecomp(a)
    LUdecomposition: [L][U] = [a]
    x = LUsolve(a,b)
    Solution phase: solves [L][U]{x} = {b}
'''
import numpy as np
def LUdecomp(a):
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a [i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a

def LUsolve(a,b):
    n = len(a)
    for k in range(1,n):
        b[k] = b[k] - np.dot(a[k,0:k],b[0:k])
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    
    return b




### module LUdecomp3
''' 
    c,d,e = LUdecomp3(c,d,e).
    LU decomposition of tridiagonal matrix [c\d\e]. On output
    {c},{d} and {e} are the diagonals of the decomposed matrix.
    x = LUsolve(c,d,e,b).
    Solves [c\d\e]{x} = {b}, where {c}, {d} and {e} are the
    vectors returned from LUdecomp3.
'''

def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    
    return b




### module LUdecomp5
'''
    d,e,f = LUdecomp5(d,e,f).
    LU decomposition of symmetric pentadiagonal matrix [a], where
    {f}, {e} and {d} are the diagonals of [a]. On output
    {d},{e} and {f} are the diagonals of the decomposed matrix.
    x = LUsolve5(d,e,f,b).
    Solves [a]{x} = {b}, where {d}, {e} and {f} are the vectors
    returned from LUdecomp5.
'''
def LUdecomp5(d,e,f):
    n = len(d)
    for k in range(n-2):
        lam = e[k]/d[k]
        d[k+1] = d[k+1] - lam*e[k]
        e[k+1] = e[k+1] - lam*f[k]
        e[k] = lam
        lam = f[k]/d[k]
        d[k+2] = d[k+2] - lam*f[k]
        f[k] = lam
    lam = e[n-2]/d[n-2]
    d[n-1] = d[n-1] - lam*e[n-2]
    e[n-2] = lam
    return d,e,f

def LUsolve5(d,e,f,b):
    n = len(d)
    b[1] = b[1] - e[0]*b[0]
    for k in range(2,n):
        b[k] = b[k] - e[k-1]*b[k-1] - f[k-2]*b[k-2]
        b[n-1] = b[n-1]/d[n-1]
        b[n-2] = b[n-2]/d[n-2] - e[n-2]*b[n-1]
    for k in range(n-3,-1,-1):
        b[k] = b[k]/d[k] - e[k]*b[k+1] - f[k]*b[k+2]
    return b




## module swap
'''
    swapRows(v,i,j).
    Swaps rows i and j of a vector or matrix [v].
    swapCols(v,i,j).
    Swaps columns of matrix [v]
'''
def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]



import sys
def err(string):
    print(string)
    input('Press return to exit') 
    sys.exit()
   
    
    
    
    
    
### module choleski
''' 
    L = choleski(a)
    Choleski decomposition: [L][L]transpose = [a]
    x = choleskiSol(L,b)
    Solution phase of Choleski’s decomposition method
'''
import numpy as np
import math
#import error
def choleski(a):
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = math.sqrt(a[k,k] - np.dot(a[k,0:k],a[k,0:k]))
        except ValueError:
            error.err('Matrix is not positive definite')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): a[0:k,k] = 0.0
    
    return a


def choleskiSol(L,b):
    n = len(b)
  # Solution of [L]{y} = {b}
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k,0:k],b[0:k]))/L[k,k]
  # Solution of [L_transpose]{x} = {y}
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    
    return b



    
    

## module LUpivot
'''
    a,seq = LUdecomp(a,tol=1.0e-9).
    LU decomposition of matrix [a] using scaled row pivoting.
    The returned matrix [a] = contains [U] in the upper
    triangle and the nondiagonal terms of [L] in the lower triangle.
    Note that [L][U] is a row-wise permutation of the original [a];
    the permutations are recorded in the vector {seq}.
    x = LUsolve(a,b,seq).
    Solves [L][U]{x} = {b}, where the matrix [a] = and the
    permutation vector {seq} are returned from LUdecomp.
'''
import numpy as np
#import swap
#import error
def LUdecompp(a,tol=1.0e-9):
    n = len(a)
    seq = np.array(range(n))

# Set up scale factors
    s = np.zeros((n))
    for i in range(n):
        s[i] = max(abs(a[i,:]))
    for k in range(0,n-1):
# Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) <  tol: error.err('Matrix is singular')
        if p != k:
            swapRows(s,k,p)
            swapRows(a,k,p)
            swapRows(seq,k,p)
# Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    
    return a, seq




## module LUdecomp 
''' a = LUdecomp(a)
LUdecomposition: [L][U] = [a]
x = LUsolve(a,b)
Solution phase: solves [L][U]{x} = {b} 
'''
import numpy as np
def LUdecomp(a):
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a [i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n] 
                a[i,k] = lam
    return a



def LUsolve(a,b):
    n = len(a)
# Rearrange constant vector; store it in [x]
    #x = b.copy()
    #for i in range(n):
    #    x[i] = b[seq[i]]
# Solution
    for k in range(1,n):
        b[k] = b[k] - np.dot(a[k,0:k],b[0:k])
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
        
    return b



def LUsolvee(a,b,seq):
    n = len(a)
# Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]
# Solution
    for k in range(1,n):
        x[k] = x[k] - np.dot(a[k,0:k],x[0:k])
    x[n-1] = x[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        x[k] = (x[k] - np.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
        
    return x



## module conjGrad
'''
    x, numIter = conjGrad(Av,x,b,tol=1.0e-9)
    Conjugate gradient method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply
    the function Av(v) that returns the vector [A]{v}.
'''

import numpy as np
import math

def conjGrad(Av,x,b,tol=1.0e-9):
    n = len(b)
    r = b - Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = np.dot(s,r)/np.dot(s,u)
        x = x + alpha*s
        r = b - Av(x)
        if(math.sqrt(np.dot(r,r))) < tol:
            break 
        else:
            beta = -np.dot(r,u)/np.dot(s,u)
            s = r + beta*s
    
    return x,i





## module gaussElimin
'''
    x = gaussElimin(a,b).
    Solves [a]{b} = {x} by Gauss elimination.
'''

import numpy as np

def gaussElimin(a, b):
    n = len(b)
    # Elimination Phase
    for k in range(0, n-1):
        for i in range(k+1, n):
            if a[i, k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i, k+1:n] = a[i, k+1:n] - lam * a[k, k+1:n]
                b[i] = b[i] - lam * b[k]

    # Back substitution
    for k in range(n-1, -1, -1):
        b[k] = (b[k] - np.dot(a[k,k+1:n], b[k+1:n]))/a[k,k]
    
    return b




## module gaussPivot
'''
    x = gaussPivot(a,b,tol=1.0e-12).
    Solves [a]{x} = {b} by Gauss elimination with
    scaled row pivoting
'''
import numpy as np
#import swap
#import error
def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)
  # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))
    for k in range(0,n-1):
  # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol: error.err('Matrix is singular')
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)
      # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    
    if abs(a[n-1,n-1]) < tol: error.err('Matrix is singular')
  # Back substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    
    return b


## module gaussSeidel
'''
    x,numIter,omega = gaussSeidel(iterEqs,x,tol = 1.0e-9)
    Gauss-Seidel method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply the
    function iterEqs(x,omega) that returns the improved {x},
    given the current {x} (’omega’ is the relaxation factor).
'''
import numpy as np
import math

def gaussSeidel(iterEqs,x,tol = 1.0e-9):
    omega = 1.0
    k = 10
    p=1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(np.dot(x-xOld,x-xOld))
        if dx < tol: return x,i,omega
# Compute relaxation factor after k+p iterations
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))
            
    print('Gauss-Seidel failed to converge')
