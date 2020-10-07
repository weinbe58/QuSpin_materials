import matplotlib
matplotlib.rcParams['text.usetex'] = True
from quspin.operators import hamiltonian,quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting results
#
##### define model parameters #####
L=16 # system size
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
operator_dict=dict(Jzz=[["zz",J_nn]],hx=[["x",x_field]])
m2_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]
###### construct Hamiltonian
H = quantum_operator(operator_dict,basis=basis,dtype=np.float64)
M2=hamiltonian([["zz",m2_list]],[],dtype=np.float64,basis=basis)
##### calculate ground state at t=0 #####
E0,psi0 = H.eigsh(k=1,which="SA",pars=dict(Jzz=0,hx=1))
psi0 = psi0.reshape((-1,))
##### evolve state #####

times = np.linspace(0,1,101)
M2_eq = np.zeros(101)
E_eq = np.zeros(101)

# calculating ground state expectation values
for i,t in enumerate(times):
	s = fzz(t,1)
	[E],psi = H.eigsh(k=1,which="SA",maxiter=1e4,pars=dict(Jzz=s,hx=1-s))
	M2_eq[i] = M2.expt_value(psi.ravel())
	E_eq[i] = E


f,(ax1,ax2) = plt.subplots(2,1,figsize=(6,12))
ax1.plot(fzz(times,1),M2_eq,linewidth=2,label=r"$T=\infty$")
ax2.plot(fzz(times,1),E_eq,linewidth=2)

# calculate dynamics
for T in [1,2,4,8,16,32,64,128]:
	H_t = H.tohamiltonian(dict(Jzz=(fzz,(T,)),hx=(fx,(T,))))
	psi_t = H_t.evolve(psi0,0,T*times,verbose=True,atol=1e-8,rtol=1e-8)
	M2_t=M2.expt_value(psi_t).real
	E_t = H_t.expt_value(psi_t,time=T*times).real


	ln, = ax1.plot(fzz(times,1),M2_t,linestyle="--",linewidth=2,label=r"$T={}$".format(T))
	ax2.plot(fzz(times,1),E_t,linestyle="--",linewidth=2,color=ln.get_color())

ax1.set_xlabel("s",fontsize=16)
ax2.set_xlabel("s",fontsize=16)
ax1.set_ylabel(r"$\langle m^2\rangle$",fontsize=16)
ax2.set_ylabel(r"$\langle E\rangle$",fontsize=16)
ax1.legend()
f.tight_layout()
f.subplots_adjust(hspace=0.02)
plt.show()