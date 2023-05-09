import os 
import sys
import numpy as np
import scipy as sp
import scipy.sparse as scysparse
import scipy.sparse.linalg as scysparselinalg  # sparse linear algebra
import pylab as plt
from pdb import set_trace as keyboard

def LeftRightNoSlip(vMat):
    # left wall no-slip
    vMat[:,0] = -vMat[:,1]
    # right wall no-slip
    vMat[:,-1] = -vMat[:,-2]

    return vMat

def TopBotNoSlip(uMat):
    # bottom wall no-slip
    uMat[0,:] = -uMat[1,:]
    # top wall no-slip
    uMat[-1,:] = - uMat[-2,:]

    return uMat

def Temp_BC(TMat):
    #TOP
    TMat[0,:] = TMat[1,:]  ## adiabatic top edge
    #BOTTOM
    TMat[-1,:] = TMat[-2,:] ## adiabatic bottom edge

    return TMat

def initializeLowMachCom(uMat, vMat, pMat, rMat,P0Mat, TMat, Uinflow, Density_ambient, amb_temp, amb_pressure):
    # uMat[:] = 0.0 
    # vMat[:] = 0.0
    pMat[:] = 1.0
    rMat[:] = Density_ambient
    TMat[:] = amb_temp
    P0Mat[:] = amb_pressure

    # Left and right ghost points x-velocity
    # uMat[1:-1,1:-1] = Uinflow 
    # uMat[:,-1] = 0.0
    # Left and right boundary x-velocity
    uMat[1:-1,:] = Uinflow
    # uMat[:,-2] = 0.0

    # top bottom ghost y-velocity
    vMat[:,1:-1] = 0.0
    # vMat[-1,:] = 0.0
    # top bottom boundary y-velocity
    # vMat[1,:] = 0.0; vMat[-2,:] = 0.0

    vMat = LeftRightNoSlip(vMat)
    uMat = TopBotNoSlip(uMat)

    rMat[:,1] = Density_ambient; rMat[:,-2] = Density_ambient

    # top bottom ghost y-velocity
    rMat[0,:] = Density_ambient; rMat[-1,:] = Density_ambient
    # top bottom boundary y-velocity
    rMat[1,:] = Density_ambient; rMat[-2,:] = Density_ambient

    TMat[:,1] = amb_temp; TMat[:,-2] = amb_temp

    TMat = Temp_BC(TMat)
    # top bottom ghost y-velocity
    TMat[0,:] = amb_temp; TMat[-1,:] = amb_temp
    # top bottom boundary y-velocity
    TMat[1,:] = amb_temp; TMat[-2,:] = amb_temp

    P0Mat[:,1] = amb_pressure; P0Mat[:,-2] = amb_pressure


    # top bottom ghost y-velocity
    P0Mat[0,:] = amb_pressure; P0Mat[-1,:] = amb_pressure
    # top bottom boundary y-velocity
    P0Mat[1,:] = amb_pressure; P0Mat[-2,:] = amb_pressure


    return uMat, vMat, pMat,rMat,TMat,P0Mat

def uConvection(uMat, vMat, r_mat, Dxu, Dyu,Xc, Xu, Yv, Yu, Xv):
    u2e = 0.25*(((uMat[1:-1, 2:-2] + uMat[1:-1, 3:-1])**2)*r_mat[1:-1,2:-1])
    u2w = 0.25*(((uMat[1:-1, 2:-2] + uMat[1:-1, 1:-3])**2)*r_mat[1:-1,1:-2])
    du2dx = (u2e - u2w)/Dxu[1:-1, 1:-1]
    d1 = (r_mat[1:-1,1:-2] + r_mat[1:-1,2:-1])/2
    d2 = (r_mat[2:,1:-2] + r_mat[2:,2:-1])/2
    d3 = (r_mat[0:-2,1:-2] + r_mat[0:-2,2:-1])/2
    uvn = 0.25*(uMat[1:-1, 2:-2] + uMat[2:, 2:-2])*(vMat[2:-1, 1:-2] + vMat[2:-1,2:-1])*(0.5*(d1+d2))
    uvs = 0.25*(uMat[1:-1, 2:-2] + uMat[:-2, 2:-2])*(vMat[1:-2,1:-2] + vMat[1:-2,2:-1])*(0.5*(d1+d3))
    duvdy = (uvn - uvs)/Dyu[1:-1,1:-1]
    return du2dx + duvdy

def vConvection(uMat, vMat, r_mat, Dxv, Dyv,Yc, Yv, Xu,Xv, Yu):
    v2n = 0.25*(((vMat[2:-2, 1:-1] + vMat[3:-1,1:-1]))**2)*r_mat[2:-1,1:-1]
    v2s = 0.25*(((vMat[2:-2, 1:-1] + vMat[1:-3, 1:-1]))**2)* r_mat[1:-2,1:-1]
    d2vdy = (v2n - v2s)/Dyv[1:-1,1:-1]
    d1 = (r_mat[2:-1,2:] + r_mat[1:-2,2:])/2
    d2 = (r_mat[2:-1,1:-1] + r_mat[1:-2,1:-1])/2
    d3 = (r_mat[2:-1,0:-2] + r_mat[1:-2,0:-2])/2
    uve = 0.25*(uMat[1:-2 ,2:-1] + uMat[2:-1,2:-1])*(vMat[2:-2,1:-1] + vMat[2:-2,2:])*(0.5*(d1+d2))
    uvw = 0.25*(uMat[1:-2, 1:-2] + uMat[2:-1,1:-2])*(vMat[2:-2,1:-1] + vMat[2:-2,:-2])*(0.5*(d2+d3))
    duvdx = (uve - uvw)/Dxv[1:-1,1:-1]

    return duvdx + d2vdy

def uDiffusion(uMat, vMat, Xu, Yu, Dxu, Dyu, nu):
    nududx = nu*(uMat[1:-1, 2:-1] - uMat[1:-1, 1:-2])/(Xu[1:-1, 2:-1] - Xu[1:-1, 1:-2])
    nududy = nu*(uMat[1:,2:-2] - uMat[:-1,2:-2])/(Yu[1:,2:-2] - Yu[:-1,2:-2])

    diffx = (nududx[:,1:] - nududx[:,:-1])/Dxu[1:-1,1:-1]
    diffy = (nududy[1:,:] - nududy[:-1,:])/Dyu[1:-1,1:-1]

    return diffx + diffy

def vDiffusion(uMat, vMat, Xv, Yv, Dxv, Dyv, nu):
    nudvdx = nu*(vMat[2:-2, 1:] - vMat[2:-2, :-1])/(Xv[2:-2, 1:] - Xv[2:-2, :-1])   
    nudvdy = nu*(vMat[2:-1,1:-1] - vMat[1:-2, 1:-1])/(Yv[2:-1,1:-1] - Yv[1:-2, 1:-1])

    diffx = (nudvdx[:,1:] - nudvdx[:,:-1])/Dxv[1:-1,1:-1]
    diffy = (nudvdy[1:,:] - nudvdy[:-1,:])/Dyv[1:-1,1:-1]

    return diffx + diffy

def Energy_temp(uMat, vMat, rMat, tMat, k, cp, dt, divX, divY, divGrad): ## this spatial_operators will work as the temperature is defined analogous to pressure at the cell centre
    tMat = Temp_BC(tMat)
    t_ulike=(tMat[1:-1,:-1]+tMat[1:-1,1:])/2
    t_vlike=(tMat[:-1,1:-1]+tMat[1:,1:-1])/2
    
    qVec = ((divX.dot(t_ulike.flatten()*uMat[1:-1,1:-1].flatten()) + \
            divY.dot(t_vlike.flatten()*vMat[1:-1,1:-1].flatten())) - (tMat[1:-1,1:-1].flatten()) / dt) * rMat[1:-1,1:-1].flatten()*cp #/dt
    
    #keyboard()
    a=cp*rMat[1:-1,1:-1].flatten()
    b=(((k*divGrad.dot(tMat[1:-1,1:-1].flatten()))-qVec)*dt)
    
    T = b/a
    #keyboard()
    # T = (((k*divGrad.dot(tMat[1:-1,1:-1].flatten()))-qVec)*dt)/(cp*rMat[1:-1,1:-1].flatten())

    return T

def ThermoPressure_p0(p0Mat, uMat, dt, Xu, gamma):
    # keyboard()
    q=(gamma*p0Mat[1:-1,1:-1]*(uMat[1:-1,1:-2]-uMat[1:-1,2:-1]))/(Xu[1:-1,2:-1]-Xu[1:-1,1:-2])
    #keyboard()

    return (q*dt)+p0Mat[1:-1,1:-1]

def RHS(uMat, vMat,r_mat, mesh_info, nu):
    [Xc, Yc, Xu, Yu, Xv, Yv, \
    Dxc, Dyc, Dxu, Dyu, Dxv, Dyv, \
    Np, Nu, Nv, \
    jp, ip,jt,it,jd,i_d,jp0,ip0, ju, iu, jv, iv,\
    pressureIndices, uIndices, vIndices, Xn, Yn] = mesh_info

    Ruconv = uConvection(uMat, vMat, r_mat, Dxu, Dyu,Xc, Xu, Yv, Yu, Xv)

    Rvconv = vConvection(uMat, vMat, r_mat, Dxv, Dyv,Yc, Yv, Xu,Xv, Yu)

    Rudiff = uDiffusion(uMat, vMat, Xu, Yu, Dxu, Dyu, nu)

    Rvdiff = vDiffusion(uMat, vMat, Xv, Yv, Dxv, Dyv, nu)

    return Rudiff - Ruconv, Rvdiff - Rvconv


def fractionalStepP1EE(uMat, vMat, pMat,p0Mat, r_mat, T_mat,dt, \
    divX, divY, gradX, gradY, divGrad,Xu, nu, R, k, Cp, gamma, mesh_info):

    jp, ip = mesh_info[15:17]
    jt, it = mesh_info[17:19]
    jd, i_d = mesh_info[19:21]
    jp0,ip0 =mesh_info[21:23]
    ju, iu = mesh_info[23:25]
    jv, iv = mesh_info[25:27]

    u0 = uMat; v0 = vMat
    uRK = u0; vRK = v0

    d1 = (r_mat[1:-1,1:-2] + r_mat[1:-1,2:-1])/2
    d2 = (r_mat[2:-1,1:-1] + r_mat[1:-2,1:-1])/2

    #STEP 1
    Ru, Rv = RHS(uMat, vMat,r_mat, mesh_info, nu)
    uRK[1:-1,2:-2] = u0[1:-1,2:-2]*d1 + Ru*dt
    vRK[2:-2,1:-1] = v0[2:-2,1:-1]*d2 + Rv*dt
    
    uRK = TopBotNoSlip(uRK)
    # vRK = LeftRightNoSlip(vRK)

    uMat[1:-1, 2:-2] += Ru*dt
    vMat[2:-2,1:-1] += Rv*dt
    # we have U_star
    divGrad[:,0] = 0.0
    divGrad[0,0] = 1.0

    #STEP 2
    T_mat = Temp_BC(T_mat)

    T_mat[jt,it] = Energy_temp(uMat, vMat, r_mat, T_mat, k, Cp, dt, divX, divY, divGrad)
    #keyboard()
    p0Mat[1:-1,1:-1] = ThermoPressure_p0(p0Mat, uMat, dt, Xu, gamma)

    r_mat_new =(p0Mat/T_mat) / R

    dense = (r_mat_new[1:-1,1:-1] - r_mat[1:-1,1:-1])/ (dt**2)
    
    r_ulike=(r_mat[1:-1,:-1]+r_mat[1:-1,1:])/2
    r_vlike=(r_mat[:-1,1:-1]+r_mat[1:,1:-1])/2
    
    # (pMat) = scysparselinalg.spsolve(divGrad,(dense + ((divX.dot(r_ulike*(uMat[1:-1,1:-1])) + divY.dot(r_vlike*(vMat[1:-1,1:-1])))/dt)))
    qVec = ( (divX.dot((r_ulike * uMat[1:-1,1:-1]).flatten()))/dt + (divY.dot((r_vlike * vMat[1:-1,1:-1]).flatten()))/dt )#+ dense.flatten()) #*0.00001

    pMat[jp, ip] = scysparselinalg.spsolve(divGrad, qVec) #* 0.000001
    #keyboard()
    ## dynamic pressure

    # STEP 3
    '''
    qVec = (divX.dot(uMat[1:-1,1:-1].flatten()) + \
    divY.dot(vMat[1:-1,1:-1].flatten()))/dt

    # divGrad[:,0] = 0.0
    # divGrad[0,0] = 1.0

    pMat[jp, ip] = scysparselinalg.spsolve(divGrad, qVec)
    '''
    uMat = TopBotNoSlip(uMat)
    uMat[ju, iu] -= dt*gradX.dot(pMat[jp,ip])/r_ulike.flatten()
    vMat[jv, iv] -= dt*gradY.dot(pMat[jp, ip])/r_vlike.flatten()
    #vMat[jv, iv] -= dt*gradY.dot(pMat[jp, ip])
    
    

    #vMat = LeftRightNoSlip(vMat)
    #r_mat_new = r_mat
    #keyboard()
    return uMat, vMat, pMat, p0Mat,r_mat_new,T_mat