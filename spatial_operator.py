import numpy as np
import scipy.sparse as scysparse
import sys
from pdb import set_trace as keyboard
import scipy.sparse as scysparse
import scipy.sparse.linalg as spysparselinalg
import scipy.linalg as scylinalg 

def gradient_operator(Xc, Yc, pressureIndices,uIndices, vIndices, jp, ip, ju, iu, jv, iv,Np, Nu, Nv):
	leftPressureCells = pressureIndices[1:-1,:-1].flatten()
	rightPressureCells = pressureIndices[1:-1,1:].flatten()
	botPressureCells = pressureIndices[:-1,1:-1].flatten()
	topPressureCells = pressureIndices[1:,1:-1].flatten()

	inv_DxE = 1.0/(Xc[jp, ip+1] - Xc[jp,ip])
	inv_DxN = 1.0/(Yc[jp+1, ip] - Yc[jp,ip])
	iE = pressureIndices[jp, ip+1]
	iN = pressureIndices[jp+1, ip]
	gradX = scysparse.csr_matrix((Nu, Np), dtype=np.float64)
	gradY = scysparse.csr_matrix((Nv, Np), dtype=np.float64)

	boundaryUMask = (leftPressureCells!=-1)&(rightPressureCells!=-1)
	boundaryVMask = (botPressureCells!=-1)&(topPressureCells!=-1)

	gradX[uIndices[ju, iu][boundaryUMask], leftPressureCells[boundaryUMask]] = -inv_DxE[iE!=-1]
	gradX[uIndices[ju, iu][boundaryUMask], rightPressureCells[boundaryUMask]] = inv_DxE[iE!=-1]

	gradY[vIndices[jv, iv][boundaryVMask], botPressureCells[boundaryVMask]] = -inv_DxN[iN!=-1]
	gradY[vIndices[jv, iv][boundaryVMask], topPressureCells[boundaryVMask]] = inv_DxN[iN!=-1]

	return gradX, gradY

def divergence_operator(Xu, Yu, Xv, Yv, \
						 pressureIndices, jjp, iip, Np,\
						  uIndices, Nu,vIndices, Nv, Dxc, Dyc):

	leftUNodes = uIndices[:,:-1].flatten()
	rightUNodes = uIndices[:,1:].flatten()

	topVNodes = vIndices[1:,:].flatten()
	botVNodes = vIndices[:-1,:].flatten()

	# inv_Dxc = 1.0/(Xu[1:-1,2:-1].flatten() - Xu[1:-1,1:-2].flatten())
	# inv_Dyc = 1.0/(Yv[2:-1,1:-1].flatten() - Yv[1:-2,1:-1].flatten())
	
	inv_Dxc = 1.0/Dxc[jjp, iip]
	inv_Dyc = 1.0/Dyc[jjp, iip]
	
	divX = scysparse.csr_matrix((Np, Nu), dtype=np.float64)
	divY = scysparse.csr_matrix((Np, Nv), dtype=np.float64)

	iC = pressureIndices[jjp,iip]
	boundaryUMask = (leftUNodes!=-1)&(rightUNodes!=-1)
	boundaryVMask = (topVNodes!=-1)&(botVNodes!=-1)

	divX[iC, leftUNodes[boundaryUMask]] = -inv_Dxc
	divX[iC, rightUNodes[boundaryUMask]] = inv_Dxc

	divY[iC, botVNodes[boundaryVMask]] = -inv_Dyc
	divY[iC, topVNodes[boundaryVMask]] = inv_Dyc

	return divX, divY