from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian,quantum_LinearOperator
import scipy.sparse as sp
from scipy.sparse.linalg import bicg
import numexpr,cProfile
import numpy as np
import matplotlib.pyplot as plt




class LHS(sp.linalg.LinearOperator):
	# LinearOperator that generates the left hand side of the equation.
	def __init__(self,H,omega,eta,E0,kwargs={}):
		self._H = H
		self._S = omega +1j*eta + E0 
		self._kwargs = kwargs

	@property
	def shape(self):
		return (self._H.Ns,self._H.Ns)
	
	@property
	def dtype(self):
		return np.dtype(self._H.dtype)
	
	def _matvec(self,v):
		return self._S * v - self._H.dot(v,**self._kwargs)

	def _rmatvec(self,v):
		return self._S.conj() * v - self._H.dot(v,**self._kwargs)

on_the_fly = False
L = 16


s = np.arange(L)
T = (s+1)%L
Z = -(s+1)


basis0 = spin_basis_general(L,m=0,pauli=False,kblock=(T,0),zblock=(Z,0))


J_list = [[1.0,i,(i+1)%L] for i in range(L)]
static = [["xx",J_list],["yy",J_list],["zz",J_list],]

if on_the_fly:
	H0 = quantum_LinearOperator(static,basis=basis0,dtype=np.float64)
else:
	H0 = hamiltonian(static,[],basis=basis0,dtype=np.float64)




[E0],psi0 = H0.eigsh(k=1,which="SA")
psi0 = psi0.ravel()
psi0 = psi0.ravel()

qs = np.arange(-L//2+1,L//2,1)
omegas = np.arange(0,6,0.05)
eta = 0.1
G = np.zeros(omegas.shape+qs.shape,dtype=np.complex128)

for j,q in enumerate(qs):
	print(q)
	if q == 0:
		continue
	elif 2*q == L: # pi-momentum, operator is odd under parity so |A> is odd as well
		block = dict(kblock=(T,q),zblock=(Z,1))
	else: 
		block = dict(kblock=(T,q),zblock=(Z,1))


	f = lambda i:np.exp(-2j*np.pi*q*i/L)/np.sqrt(L)

	Op_list = [["z",[i],f(i)] for i in range(L)]


	basisq = spin_basis_general(L,m=0,pauli=False,**block)

	if on_the_fly:
		Hq = quantum_LinearOperator(static,basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	else:
		Hq = hamiltonian(static,[],basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)

	psiA = basisq.Op_shift_sector(basis0,Op_list,psi0.ravel())

	# use vector correction method:
	# solve (z-H)|x> = |A> solve for |x> 
	# using iterative solver
	for i,omega in enumerate(omegas):
		lhs = LHS(Hq,omega,eta,E0)
		x,*_ = bicg(lhs,psiA)
		G[i,j] = -np.vdot(psiA,x)/np.pi

qs = np.array(list(qs)+[qs[-1]+1])

qs = 2*np.pi*qs/L

plt.pcolormesh(qs,omegas,G.imag)
plt.xlabel("$k$")
plt.ylabel("$\omega$")
plt.show()



