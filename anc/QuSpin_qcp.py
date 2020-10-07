from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting results
#
##### define model parameters #####
L=10 # system size
n_points = 21 # number of points to calculate
##### construct basis
basis=spin_basis_1d(L,S="1/2",pauli=True)
# define PBC site-coupling lists for operators
x_field=[[-1,i] for i in range(L)]
J_nn=[[-1,i,(i+1)%L] for i in range(L)] # PBC
m2_list = [[1.0/L**2,i,j] for i in range(L) for j in range(L)]
###### construct Hamiltonians
Hzz=hamiltonian([["zz",J_nn]],[],dtype=np.float64,basis=basis)
Hx=hamiltonian([["x",x_field]],[],dtype=np.float64,basis=basis)
M2=hamiltonian([["zz",m2_list]],[],dtype=np.float64,basis=basis)
###### calculate Magnetization as a function of s.
s_list = np.linspace(0,1,n_points)
M2_values = np.zeros_like(s_list) # creates array of zeros shaped like s_list. 
for i,s in enumerate(s_list): # loop over s-values
	print("calculating s={}".format(s))
	E0,psi0 = (s*Hzz+(1-s)*Hx).eigsh(k=1,which="SA")
	psi0 = psi0.reshape((-1,)) # flatten array. 
	M2_values[i] = M2.expt_value(psi0).real 
###### plotting results
plt.plot(s_list,M2_values,marker=".")
plt.show()