from __future__ import print_function
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt
#
##### define model parameters #####
n_points = 21 # number of s points to calculate
s_list = np.linspace(0,1,n_points)
for L in [4,8,12,16,20,24]: # loop over system sizes
	
	basis=spin_basis_1d(L,kblock=0,pblock=1,zblock=1) # create basis
	print(L,basis.Ns)
	# creating operator lists
	x_field=[[-1,i] for i in range(L)]
	J_nn=[[-1,i,(i+1)%L] for i in range(L)] # PBC
	m2_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]
	# creating operators
	Hzz=hamiltonian([["zz",J_nn]],[],dtype=np.float64,basis=basis)
	Hx=hamiltonian([["x",x_field]],[],dtype=np.float64,basis=basis)
	M2=hamiltonian([["zz",m2_list]],[],dtype=np.float64,basis=basis)
	M4=M2**2
	# calculation
	Bind_values = np.zeros_like(s_list) # creates array of zeros shaped like s_list. 
	gaps = np.zeros_like(s_list) # creates array of zeros shaped like s_list. 
	for i,s in enumerate(s_list):
		E0,psi0 = (s*Hzz+(1-s)*Hx).eigsh(k=2,which="SA")
		psi0 = psi0[:,0].ravel() # flatten array. 
		Bind_values[i] = M2.expt_value(psi0).real**2/M4.expt_value(psi0).real
		gaps[i] = E0[1]-E0[0]

	plt.plot(s_list,Bind_values,marker=".",label="$L={}$".format(L))

plt.xlabel(r"$s$",fontsize=16)
plt.ylabel(r"$\langle 0|m^2|0\rangle^2/\langle 0|m^4|0\rangle$",fontsize=16)
plt.legend(loc=0,fontsize=14)
plt.savefig("B.pdf")
