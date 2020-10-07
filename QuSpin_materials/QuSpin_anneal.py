import matplotlib
matplotlib.rcParams['text.usetex'] = True
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting results
#
##### define model parameters #####
L=20 # system size
T = 100.0
npoints=101
def fzz(t,T): # z driving
	return np.sin(np.pi*t/(2*T))**2
def fx(t,T): # x driving
	return 1-np.sin(np.pi*t/(2*T))**2
##### construct basis
basis=spin_basis_1d(L,S="1/2",pauli=True,kblock=0,pblock=1,zblock=1)
# define PBC site-coupling lists for operators
x_field=[[-1.0,i] for i in range(L)]
J_nn=[[-1.0,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[]
dynamic = [["zz",J_nn,fzz,(T,)],["x",x_field,fx,(T,)]]
m2_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]
###### construct Hamiltonian
H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
M2=hamiltonian([["zz",m2_list]],[],dtype=np.float64,basis=basis)
##### calculate ground state at t=0 #####
E0,psi0 = H.eigsh(time=0,k=1,which="SA")
psi0 = psi0.reshape((-1,))
##### evolve state #####
times = np.linspace(0,T,npoints)
M2_eq = np.zeros_like(times)
E_eq = np.zeros_like(times)

for i,t in enumerate(times):
	[E],psi = H.eigsh(time=t,k=1,which="SA",maxiter=1e4)
	M2_eq[i] = M2.expt_value(psi.ravel())
	E_eq[i] = E

psi_t = H.evolve(psi0,0,times,verbose=True,atol=1e-15,rtol=1e-15)
M2_t=M2.expt_value(psi_t).real
E_t = H.expt_value(psi_t,time=times).real

# plotting data
f,(ax1,ax2) = plt.subplots(2,1,figsize=(6,12))
ax1.plot(fzz(times,T),M2_t,linestyle="--",linewidth=2,label="dynamics")
ax1.plot(fzz(times,T),M2_eq,linewidth=2,label="ground state")
ax2.plot(fzz(times,T),E_t,linestyle="--",linewidth=2)
ax2.plot(fzz(times,T),E_eq,linewidth=2)
ax2.set_xlabel("s",fontsize=16)
ax1.set_ylabel(r"$\langle m^2\rangle$",fontsize=16)
ax2.set_ylabel(r"$\langle E\rangle$",fontsize=16)
ax1.legend()
f.tight_layout()
f.subplots_adjust(hspace=0.02)
plt.show()