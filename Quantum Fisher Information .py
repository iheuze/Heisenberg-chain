import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Given parameters
L = 10      # number of sites
J = 1       # coupling constant
tmax = 1.5  # maximum time
dt = 0.01   # time step
nsteps = int(tmax/dt)  # number of steps

# Initial state (Neel state)
up = np.array([1, 0])
down = np.array([0, 1])
psi0 = np.kron(up, down)
for i in range(2, L):
    if i % 2 == 0:
        psi0 = np.kron(psi0, up)
    else:
        psi0 = np.kron(psi0, down)


for i, val in enumerate(psi0):
    if abs(val) > 1e-6:
        print(f"Nonzero value {val} at index {i} in psi0")

# Hamiltonian
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
H = np.zeros((2**L, 2**L), dtype=complex)
for i in range(L-1):
    H += -J*np.kron(np.kron(np.eye(2**(i)), np.kron(sx, sx)), np.eye(2**(L-i-2))) \
    -J*np.kron(np.kron(np.eye(2**(i)), np.kron(sy, sy)), np.eye(2**(L-i-2))) \
    -0*np.kron(np.kron(np.eye(2**(i)), np.kron(sz, sz)), np.eye(2**(L-i-2))) # set to J for U=J

# Time evolution
t = np.linspace(0, tmax, nsteps)
S = np.zeros((nsteps,))
for j in range(nsteps):
  U = expm(-1j*dt*H)
    psi = np.reshape(psi0, (2**L, 1))
    for k in range(0, int(t[j]/dt)):
        psi = np.dot(U, psi)
    # Entanglement entropy
    n = int(L/2)  # number of sites in the left subsystem
    rhoA = np.reshape(psi, (2**n, 2**n))
    rhoB = np.reshape(np.conj(psi), (2**(L-n), 2**(L-n))).T
    C = np.dot(rhoA, rhoB)
    eig_vals = np.linalg.eigvalsh(C)
    eig_vals = eig_vals[eig_vals > 0]
    S[j] = -np.sum(eig_vals * np.log(eig_vals))
    FQ = np.zeros((nsteps,))
    for j in range(nsteps):
        U = expm(-1j*dt*H)
        psi = np.reshape(psi0, (2**L, 1))
        Fj = np.zeros(L)  # initialize Fisher information for all modes
        for k in range(0, int(t[j]/dt)):
            psi = np.dot(U, psi)
            for jj in range(L):
                if jj < L/2:
                    sj = 1
                elif jj >= L/2:
                    sj = -1
                else:
                    sj = 0
            for kk in range(L):
                if kk < L/2:
                    sk = 1
                elif kk >= L/2:
                    sk = -1
                else:
                    sk = 0
                skj = sj * sk
                fq = 0.
                fq1 = np.conj(psi) * skj * psi[kk]
                fq2 = np.conj(psi) * sj * psi[kk]
                fq += fq1
                fq -= (fq2)**2
                Fj[kk] += np.sum(np.real(fq))  # accumulate Fisher information for current mode
    FQ[j] = 5/32*np.sum(Fj)  # store total Fisher information for current time
    
# Plot results
plt.plot(t, S, color="darkviolet")
plt.plot(t, FQ, color="violet")
plt.xlabel('Jt/$\hbar$')
plt.ylabel('Entanglement entropy')
plt.show()
