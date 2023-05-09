import os 
import sys
import numpy as np
import scipy as sp
import scipy.sparse as scysparse
import pylab as plt
from pdb import set_trace as keyboard
import mesh
import NS2D2_
from PyPDF2 import PdfMerger

Nxc = 64
Nyc = 64
Lx = 2
Ly = 2
Uwall = 1.0
nu = 1.0
CFL = 0.5
FOU = 0.5
Uinflow=10.0
Density_ambient=1.025
amb_temp=280
amb_pressure=101.325
R=0.287
k=0.025
Cp=1.005
gamma=1.4


mesh_info = mesh.setup(Nxc, Nyc, Lx, Ly)
# np.savez_compressed("./meshInfo", mesh_info)

[Xc, Yc, Xu, Yu, Xv, Yv, \
    Dxc, Dyc, Dxu, Dyu, Dxv, Dyv, \
    Np, Nu, Nv, \
    jp, ip,jt,it,jd,i_d,jp0,ip0, ju, iu, jv, iv,\
    pressureIndices, uIndices, vIndices, Xn, Yn] = mesh_info

dx_min = np.sqrt(np.min(Xc)*np.min(Yc))

divX, divY, gradX, gradY = mesh.getDivGrad_Neumann(mesh_info)
divGrad = divX.dot(gradX) + divY.dot(gradY)

u_mat = np.empty(Xu.shape, dtype=np.float64)
v_mat = np.empty(Xv.shape, dtype=np.float64)
p_mat = np.empty(Xc.shape, dtype=np.float64)
r_mat = np.empty(Xc.shape, dtype=np.float64)
t_mat = np.empty(Xc.shape, dtype=np.float64)
pt_mat = np.empty(Xc.shape, dtype=np.float64)

u_mat, v_mat, p_mat, r_mat, t_mat, pt_mat = NS2D2_.initializeLowMachCom(u_mat, v_mat, p_mat, r_mat,pt_mat, t_mat, Uinflow, Density_ambient, amb_temp, amb_pressure)

t_final = 0.2
t = 0.0
dt = 1.0/(Uwall/(0.5*dx_min) + nu/(0.5*(dx_min**2)))
t_iter = 0
max_iter = t_final/dt
frame_iter = 0
plt.ion()

fig, axs = plt.subplots(2, 2, figsize=(8, 4))
pdf_merger = PdfMerger()

while(t < t_final) & (t_iter < max_iter):
    u_mat, v_mat, p_mat, pt_mat,r_mat, t_mat = NS2D2_.fractionalStepP1EE(u_mat, v_mat, p_mat,pt_mat, r_mat, t_mat,dt, \
    divX, divY, gradX, gradY, divGrad,Xu, nu, R, k, Cp, gamma, mesh_info)
    t+=dt 
    t_iter+=1
    if(t_iter%100 == 0):
        print("saved file at t=",t," and iter=",t_iter )
        axs[0,0].contourf(Xv, Yv, v_mat, levels=np.linspace(np.min(v_mat), np.max(v_mat), 50),cmap='viridis')
        axs[0,0].set_title(f'V_velocity at t = {t}')
        axs[0,1].contourf(Xu, Yu, u_mat, levels=np.linspace(np.min(u_mat), np.max(u_mat), 50),cmap='coolwarm')
        axs[0,1].set_title(f'U_velocity at t = {t}')
        axs[1,0].contourf(Xc, Yc, p_mat, levels=np.linspace(np.min(p_mat), np.max(p_mat), 50),cmap='magma')
        axs[1,0].set_title(f'Pressure at t = {t}')
        axs[1,1].contourf(Xc, Yc, t_mat, levels=np.linspace(np.min(t_mat), np.max(t_mat), 50),cmap='magma')
        axs[1,1].set_title(f'Pressure at t = {t}')
        plt.draw()
        plt.pause(0.000001)
        plt.show()
        frame_iter+=1
        plt.savefig('frame{}.pdf'.format(frame_iter))
        with open('frame{}.pdf'.format(frame_iter), 'rb') as f:
            pdf_merger.append(f)
        os.remove('frame{}.pdf'.format(frame_iter))
with open('output.pdf', 'wb') as f:
    pdf_merger.write(f)
plt.ioff()
print(u_mat)
print(t_mat)