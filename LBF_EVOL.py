# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:54:06 2018

@author: Administrator
"""

import numpy as np
import scipy.stats as st
from scipy import signal
import scipy 

def lbf(u0, Img, Ksigma, KI, KONE, nu, timestep, mu, lambda1, lambda2, epsilon):
    u = u0
    u = NeumannBoundCond(u)
    K = curvature_central(u)
    DrcU= (epsilon/np.pi)/(epsilon**2+u**2)
    
    [f1, f2] = localBinaryFit(Img, u, KI, KONE, Ksigma, epsilon)
    
    s1 = lambda1*(f1**2) - lambda2*(f2**2)                 #compute lambda1*e1-lambda2*e2 in the 1st term in eq. (15) in IEEE TIP 08
    s2 = lambda1*f1 - lambda2*f2
    
    dataForce = (lambda1-lambda2)*KONE*Img*Img+signal.convolve2d(s1,Ksigma,'same')-2*Img*signal.convolve2d(s2,Ksigma,'same')
    
    A = -DrcU*dataForce
    P = mu*(4*del2(u)-K)
    L = nu*DrcU*K
    u = u + timestep*(L+P+A)
    
    return u

def localBinaryFit(Img, u, KI, KONE, Ksigma, epsilon):#局部二值拟合
    Hu = 0.5*(1+(2/np.pi)*np.arctan(u/epsilon))#heaviside
    
    I = Img*Hu
    c1 = signal.convolve2d(Hu, Ksigma, 'same')
    c2 = signal.convolve2d(I, Ksigma, 'same')
    f1 = (c2)/(c1)                                           #compute f1 according to eq.(14) for i = 1
    f2 = (KI-c2)/(KONE-c1)
    
    return f1, f2


def NeumannBoundCond(f):#诺依曼边界条件
    g = f
    g[0,0] = g[2,2]
    g[0,-1] = g[2,-3]
    g[-1, 0] = g[-3, 2]
    g[-1,-1] = g[-3,-3]
    g[0][1:-1] = g[2][1:-1]
    g[-1][1:-1] = g[-3][1:-1]
    
    g[0][1:-1] = g[2][1:-1]
    g[-1][1:-1] = g[-3][1:-1]
    
    g[1:-1,0] = g[1:-1,2]
    g[1:-1,-1] = g[1:-1,-3]
    
    return g

def curvature_central(u):       #中心差分求kappa
    ep = 1e-10
    [fx, fy]= np.gradient(u)
    
    ax = fx/np.sqrt(fx**2 + fy**2 + ep)
    ay = fy/np.sqrt(fx**2 + fy**2 + ep)
    
    [axx, axy] = np.gradient(ax)   #central difference
    [ayx, ayy] = np.gradient(ay)
    K = axx + ayy
    
    return K


def del2(M):
    return scipy.ndimage.filters.laplace(M)/4
#def del2(M):    #LAPLACE模拟matlab  del2()
#    dx = 1
#    dy = 1
#    rows, cols = M.shape
#    dx = dx * np.ones ((1, cols - 1))
#    dy = dy * np.ones ((rows-1, 1))
#
#    mr, mc = M.shape
#    D = np.zeros ((mr, mc))
#
#    if (mr >= 3):
#        ## x direction
#        ## left and right boundary
#        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
#        D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1]) \
#            / (dx[:,mc - 3] * dx[:,mc - 2])
#
#        ## interior points
#        tmp1 = D[:, 1:mc - 1] 
#        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
#        tmp3 = np.kron (dx[:,0:mc -2] * dx[:,1:mc - 1], np.ones ((mr, 1)))
#        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3
#
#    if (mr >= 3):
#        ## y direction
#        ## top and bottom boundary
#        D[0, :] = D[0,:]  + \
#            (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])
#
#        D[mr-1, :] = D[mr-1, :] \
#            + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :]) \
#            / (dy[mr-3,:] * dx[:,mr-2])
#
#        ## interior points
#        tmp1 = D[1:mr-1, :] 
#        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
#        tmp3 = np.kron (dy[0:mr-2,:] * dy[1:mr-1,:], np.ones ((1, mc)))
#        D[1:mr-1, :] = tmp1 + tmp2 / tmp3
#
#    return D / 4



def gaussian_kern(nsig):
    kernlen=np.around(nsig*2)*2+2
    #Returns a 2D Gaussian kernel array.
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel