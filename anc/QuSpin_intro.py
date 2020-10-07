from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
#
##### define model parameters #####
L=4 # system size
s=0.4 
##### construct basis
basis=spin_basis_1d(L,S="1/2",pauli=True)
print(basis) # displays basis states. 
# define PBC site-coupling lists for operators
x_field=[[-(1-s),i] for i in range(L)]
J_nn=[[-s,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["zz",J_nn],["x",x_field]]
dynamic = []
###### construct Hamiltonian
H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
##### various exact diagonalisation routines #####
# calculate entire spectrum only
E=H.eigvalsh()
# calculate full eigensystem
E,V=H.eigh()
# calculate ground state
Emin,psi_0=H.eigsh(k=1,which="SA",maxiter=1E4)