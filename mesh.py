import os 
import sys
import numpy as np
import scipy as sp
import scipy.sparse as scysparse
import pylab as plt
from pdb import set_trace as keyboard
import spatial_operator


figwidth       = 10
figheight      = 10
lineWidth      = 4
textFontSize   = 28
gcafontSize    = 30

def setup(Nxc, Nyc, Lx, Ly):

	xsi_u = np.linspace(0.0, 1.0, Nxc+1)
	xsi_v = np.linspace(0.0, 1.0, Nyc+1)

	xu = xsi_u*Lx
	yv = xsi_v*Ly
	dxu = xu[1:] - xu[:-1]
	dyv= yv[1: ] - yv[:-1]

	xu = np.concatenate([[xu[0] - dxu[0]], xu, [xu[-1] + dxu[-1]]])
	yv = np.concatenate([[yv[0] - dyv[0]], yv, [yv[-1] + dyv[-1]]])
	# x coordinates of centers
	xc = 0.5*(xu[1:] + xu[:-1])
	yc = 0.5*(yv[1:] + yv[:-1])

	dxc = xu[1:]-xu[:-1]
	dyc = yv[1:] - yv[:-1]
	dxu = np.diff(xc)
	dyv = np.diff(yc)
	
	[Dxc, Dyc] = np.meshgrid(dxc, dyc)
	[Dxu,Dyu] = np.meshgrid(dxu,dyc)
	[Dxv,Dyv] = np.meshgrid(dxc,dyv)

	[Xc, Yc] = np.meshgrid(xc, yc)
	[Xu, Yu] = np.meshgrid(xu, yc)
	[Xv, Yv] = np.meshgrid(xc, yv)
	[Xn, Yn] = np.meshgrid(xu, yv)

	cellsMask = np.zeros(Xc.shape)
	cellsMask[1:-1, 1:-1] = True

	uMask = np.zeros(Xu.shape)
	uMask[1:-1, 1:-1] = True
	vMask = np.zeros(Xv.shape)
	vMask[1:-1, 1:-1] = True

	pressureIndices = -np.ones(Xc.shape, dtype=np.int32)
	jp,ip = np.where(cellsMask==True)
	Np = len(jp)
	pressureIndices[jp,ip] = range(0, Np)

	temperatureIndices = -np.ones(Xc.shape, dtype=np.int32)
	jt,it = np.where(cellsMask==True)
	Nt = len(jt)
	temperatureIndices[jt,it] = range(0, Nt)

	densityIndices = -np.ones(Xc.shape, dtype=np.int32)
	jd,i_d = np.where(cellsMask==True)
	Nd = len(jd)
	densityIndices[jd,i_d] = range(0, Nd)

	p0Indices = -np.ones(Xc.shape, dtype=np.int32)
	jp0,ip0 = np.where(cellsMask==True)
	Np0 = len(jp0)
	densityIndices[jp0,ip0] = range(0, Np0)	

	uIndices = -np.ones(Xu.shape, dtype=np.int32)
	ju,iu = np.where(uMask==True)
	Nu = len(iu)
	uIndices[ju,iu] = range(0, Nu)

	vIndices = -np.ones(Xv.shape, dtype=np.int32)
	jv,iv = np.where(vMask==True)
	Nv = len(iv)
	vIndices[jv,iv] = range(0, Nv)

	plot_mesh([Xc, Yc], [Xu, Yu], [Xv, Yv], [Xn, Yn],\
	 [jp, ip], [ju, iu], [jv, iv])

	mesh_info = [Xc, Yc, Xu, Yu, Xv, Yv, \
	Dxc, Dyc, Dxu, Dyu, Dxv, Dyv, \
	Np, Nu, Nv, \
	jp, ip,jt,it,jd,i_d,jp0,ip0, ju, iu, jv, iv, 
	pressureIndices, uIndices, vIndices, Xn, Yn]

	return mesh_info
def getDivGrad_Neumann(mesh_info):
	[Xc, Yc, Xu, Yu, Xv, Yv, \
    Dxc, Dyc, Dxu, Dyu, Dxv, Dyv, \
    Np, Nu, Nv, \
    jp, ip,jt,it,jd,i_d,jp0,ip0, ju, iu, jv, iv,\
    pressureIndices, uIndices, vIndices, Xn, Yn] = mesh_info


	gradX, gradY = spatial_operator.gradient_operator(Xc, Yc, \
		pressureIndices,uIndices, vIndices, jp, ip, ju, iu, jv,iv,  Np, Nu, Nv)

	divX, divY = spatial_operator.divergence_operator(Xu, Yu, Xv, Yv, \
							 pressureIndices, jp, ip, Np,\
							  uIndices, Nu,vIndices, Nv, Dxc, Dyc)

	return divX, divY, gradX, gradY
	
def plot_mesh(XYc, XYu, XYv, XYn, ijc, iju, ijv):
	fig = plt.figure(0, figsize=(figwidth,figheight))
	ax  = fig.add_axes([0.15,0.15,0.8,0.8])
	plt.axes(ax)

	plt.plot(XYc[0],XYc[1], 'ko',mfc="None")
	plt.plot(XYc[0][ijc[0],ijc[1]], XYc[1][ijc[0],ijc[1]],'ko')


	plt.plot(XYn[0],XYn[1], 'k-')
	plt.plot(XYn[0].T,XYn[1].T, 'k-')

	plt.plot(XYu[0], XYu[1], 'r>', mfc="None", markersize=3)
	plt.plot(XYu[0][iju[0],iju[1]], XYu[1][iju[0],iju[1]], 'r>', markersize=3)

	plt.plot(XYv[0], XYv[1], 'b^', mfc="None", markersize=3)
	plt.plot(XYv[0][ijv[0],ijv[1]], XYv[1][ijv[0], ijv[1]], 'b^', markersize=3)	
	
	plt.savefig("./2DNS_mesh.pdf")
	plt.close()	

# divGrad = divX.dot(gradX) + divY.dot(gradY)

# f = np.sin(2.0*np.pi*Xc)*np.cos(2.0*np.pi*Yc)
# Lap_f_a = -8.0*(np.pi*np.pi)*f


# dfdx_a = 2.0*np.pi*np.cos(2.0*np.pi*Xc)*np.cos(2.0*np.pi*Yc)
# dfdx_n = divX.dot(f.flatten())
# dfdx_mat = np.empty(Xc.shape)*np.nan
# dfdx_mat[jp, ip] = dfdx_n

# Lap_f_n = np.empty(Xc.shape)
# Lap_f_n[jp, ip] = divGrad.dot(f[1:-1,1:-1].flatten())

# # eps = np.linalg.norm(dfdx_mat[1:-1,1:-1] - dfdx_a[1:-1,1:-1])
# # print(eps)

# plt.contourf(Xc, Yc, Lap_f_a, 21)
# plt.colorbar()
# plt.savefig('./laplacian_test.pdf')
# plt.close()
# plt.contourf(Xc[2:-2,2:-2], Yc[2:-2,2:-2], Lap_f_n[2:-2,2:-2], 21)
# plt.colorbar()
# plt.show()

# dfdx_a = 2.0*np.pi*np.cos(2.0*np.pi*Xc)*np.cos(2.0*np.pi*Yc)
# dfdx_n = gradX.dot(f[jp,ip])
# dfdx_n_mat = np.empty(Xu.shape)

# dfdx_n_mat[np.where(uMask==True)] = dfdx_n
# # plt.contourf(Xc, Yc, dfdx_a, 50)
# plt.contourf(Xu, Yu, dfdx_n_mat, 50)
# plt.show()
# keyboard()
# # plt.plot(Xc, Yc, 'ko')
# # plt.plot(Xu, Yu, 'r>')
# # plt.plot(Xv, Yv, 'b^')
# # plt.show()