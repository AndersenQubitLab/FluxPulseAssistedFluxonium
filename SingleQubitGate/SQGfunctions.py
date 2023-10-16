import numpy as np
import qutip as qtp
from tqdm.notebook import tqdm
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from qutip import *
import scipy.integrate as integrate
qutip.settings.auto_tidyup = False
from scipy.optimize import minimize
import cma
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import pandas as pd

N = 30 #Hilbert space size for qubit
M = 5 #Hilbert space size for resonator

#creation/annihilation operators for qubit
c = destroy(N)
cdag = create(N)

#creation/annihilation operators for resonator
a = destroy(M)
adag = create(M)

rnum = adag * a
resonator_num = tensor(qeye(N), rnum)

sweet_spot = 2*np.pi*0.5
Ej = 2*np.pi*4.75 #Josephson energy, (2piGHz)
Ec = 2*np.pi*1.25 #charging energy, (2piGHz)
El = 2*np.pi*1.5 #inductive energy, (2piGHz)

#resonator frequency
w = 2*np.pi*7.0 #2piGHz
H_lc = w * (adag * a + 1/2)

#qubit-resonator coupling
g = 2*np.pi*0.05
coupling1 = tensor(c, adag)
coupling2 = tensor(cdag, a)
H_i = g * (coupling1 + coupling2)

#reduced charge and flux operators
phi_naught = ((8 * Ec) / El)**(1/4)
n_op = (-1j / (math.sqrt(2) * phi_naught)) * (c - cdag)
phi_op = (phi_naught / math.sqrt(2)) * (c + cdag)
drive_op = tensor(n_op, qeye(M))

initialState = basis(8,0)
initRho = initialState #* initialState.dag()

expectState = basis(8, 2)
expectRho = expectState #* expectState.dag()

#state matrices
H00 = Qobj([[1, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H01 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H10 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H11 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H20 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H21 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H30 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],])
H31 = Qobj([[0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],])

def fid(finalState, expectedState):
    arg1 = (finalState.sqrtm() * expectedState * finalState.sqrtm())
    arg2 = arg1.sqrtm()
    fidelity = arg2.tr()**2
    return fidelity

def fluxonium_hamiltonian(phi,Ej,Ec,El,n_op,phi_op):
    Cterm = 4 * Ec * (n_op)**2
    Lterm = (1/2) * El * phi_op**2
    Jterm = -Ej * ((1/2) * ((1j * (phi_op - phi)).expm()) + (1/2) * ((-1j * (phi_op.dag() - phi)).expm()))
    H_flux = Jterm + Cterm + Lterm
    return H_flux
    
def total_hamiltonian(phi,Ej,Ec,El,n_op,phi_op,H_lc,H_i):
    H_flux = fluxonium_hamiltonian(phi,Ej,Ec,El,n_op,phi_op)
    H_sys = tensor(H_flux, qeye(M)) + tensor(qeye(N), H_lc) + H_i
    return H_sys

def fluxonium_system(phi,Ej,Ec,El,n_op,phi_op,H_lc,H_i):
    H_sys = total_hamiltonian(phi,Ej,Ec,El,n_op,phi_op,H_lc,H_i)
    evals, evecs = H_sys.eigenstates()
    evals_sort = np.argsort(evals)
    evals = evals[evals_sort]
    evecs = evecs[evals_sort]
    return H_sys,evals,evecs

def qubit_energies(H_sys,evals,evecs):  
    expvals = expect(resonator_num, evecs)
    expvals_index = np.argwhere(expvals < 1.2)
    expvals_index_res0 = np.argwhere(expvals < 0.6)
    expvals_index_res1 = np.argwhere(expvals > 0.6)
    expvals_index_res0 = np.array([i for i in expvals_index if (i in expvals_index_res0)] )[:,0]
    expvals_index_res1 = np.array([i for i in expvals_index if (i in expvals_index_res1)] )[:,0] 
    state0 = evecs[expvals_index_res0[0]] #qubit 0 resonator 0
    state1 = evecs[expvals_index_res1[0]] #qubit 0 resonator 1
    state2 = evecs[expvals_index_res0[1]] #qubit 1 resonator 0
    state3 = evecs[expvals_index_res1[1]] #qubit 1 resonator 1
    state4 = evecs[expvals_index_res0[2]] #qubit 2 resonator 0
    state5 = evecs[expvals_index_res1[2]] #qubit 2 resonator 1
    state6 = evecs[expvals_index_res0[3]] #qubit 3 resonator 0
    state7 = evecs[expvals_index_res1[3]] #qubit 3 resonator 1
    Eq0r0 = H_sys.matrix_element(state0.dag(), state0) #energies
    Eq0r1 = H_sys.matrix_element(state1.dag(), state1)
    Eq1r0 = H_sys.matrix_element(state2.dag(), state2)
    Eq1r1 = H_sys.matrix_element(state3.dag(), state3)
    Eq2r0 = H_sys.matrix_element(state4.dag(), state4)
    Eq2r1 = H_sys.matrix_element(state5.dag(), state5)
    Eq3r0 = H_sys.matrix_element(state6.dag(), state6)
    Eq3r1 = H_sys.matrix_element(state7.dag(), state7)

    return Eq0r0, Eq0r1, Eq1r0, Eq1r1, Eq2r0, Eq2r1, Eq3r0, Eq3r1

def matrix_element(drive_op,H_sys,evals,evecs):
    expvals = expect(resonator_num, evecs)
    expvals_index = np.argwhere(expvals < 1.2)
    expvals_index_res0 = np.argwhere(expvals < 0.6)
    expvals_index_res1 = np.argwhere(expvals > 0.6)
    expvals_index_res0 = np.array([i for i in expvals_index if (i in expvals_index_res0)] )[:,0]
    expvals_index_res1 = np.array([i for i in expvals_index if (i in expvals_index_res1)] )[:,0] 
    state0 = evecs[expvals_index_res0[0]] #qubit 0 resonator 0
    state1 = evecs[expvals_index_res1[0]] #qubit 0 resonator 1
    state2 = evecs[expvals_index_res0[1]] #qubit 1 resonator 0
    state3 = evecs[expvals_index_res1[1]] #qubit 1 resonator 1
    state4 = evecs[expvals_index_res0[2]] #qubit 2 resonator 0
    state5 = evecs[expvals_index_res1[2]] #qubit 2 resonator 1
    state6 = evecs[expvals_index_res0[3]] #qubit 3 resonator 0
    state7 = evecs[expvals_index_res1[3]] #qubit 3 resonator 1
    
    trunc_drive_op = Qobj([[drive_op.matrix_element(state0.dag(),state0), drive_op.matrix_element(state0.dag(),state1), drive_op.matrix_element(state0.dag(),state2), drive_op.matrix_element(state0.dag(),state3), drive_op.matrix_element(state0.dag(),state4), drive_op.matrix_element(state0.dag(),state5), drive_op.matrix_element(state0.dag(),state6), drive_op.matrix_element(state0.dag(),state7)],
                                [drive_op.matrix_element(state1.dag(),state0), drive_op.matrix_element(state1.dag(),state1), drive_op.matrix_element(state1.dag(),state2), drive_op.matrix_element(state1.dag(),state3), drive_op.matrix_element(state1.dag(),state4), drive_op.matrix_element(state1.dag(),state5), drive_op.matrix_element(state1.dag(),state6), drive_op.matrix_element(state1.dag(),state7)],
                                [drive_op.matrix_element(state2.dag(),state0), drive_op.matrix_element(state2.dag(),state1), drive_op.matrix_element(state2.dag(),state2), drive_op.matrix_element(state2.dag(),state3), drive_op.matrix_element(state2.dag(),state4), drive_op.matrix_element(state2.dag(),state5), drive_op.matrix_element(state2.dag(),state6), drive_op.matrix_element(state2.dag(),state7)],
                                [drive_op.matrix_element(state3.dag(),state0), drive_op.matrix_element(state3.dag(),state1), drive_op.matrix_element(state3.dag(),state2), drive_op.matrix_element(state3.dag(),state3), drive_op.matrix_element(state3.dag(),state4), drive_op.matrix_element(state3.dag(),state5), drive_op.matrix_element(state3.dag(),state6), drive_op.matrix_element(state3.dag(),state7)],
                                [drive_op.matrix_element(state4.dag(),state0), drive_op.matrix_element(state4.dag(),state1), drive_op.matrix_element(state4.dag(),state2), drive_op.matrix_element(state4.dag(),state3), drive_op.matrix_element(state4.dag(),state4), drive_op.matrix_element(state4.dag(),state5), drive_op.matrix_element(state4.dag(),state6), drive_op.matrix_element(state4.dag(),state7)],
                                [drive_op.matrix_element(state5.dag(),state0), drive_op.matrix_element(state5.dag(),state1), drive_op.matrix_element(state5.dag(),state2), drive_op.matrix_element(state5.dag(),state3), drive_op.matrix_element(state5.dag(),state4), drive_op.matrix_element(state5.dag(),state5), drive_op.matrix_element(state5.dag(),state6), drive_op.matrix_element(state5.dag(),state7)],
                                [drive_op.matrix_element(state6.dag(),state0), drive_op.matrix_element(state6.dag(),state1), drive_op.matrix_element(state6.dag(),state2), drive_op.matrix_element(state6.dag(),state3), drive_op.matrix_element(state6.dag(),state4), drive_op.matrix_element(state6.dag(),state5), drive_op.matrix_element(state6.dag(),state6), drive_op.matrix_element(state6.dag(),state7)],
                                [drive_op.matrix_element(state7.dag(),state0), drive_op.matrix_element(state7.dag(),state1), drive_op.matrix_element(state7.dag(),state2), drive_op.matrix_element(state7.dag(),state3), drive_op.matrix_element(state7.dag(),state4), drive_op.matrix_element(state7.dag(),state5), drive_op.matrix_element(state7.dag(),state6), drive_op.matrix_element(state7.dag(),state7)]])
    trunc_drive_op = trunc_drive_op*np.sign(np.imag(trunc_drive_op[0,2]))
    
    return drive_op.matrix_element(state0.dag(), state2), trunc_drive_op

#no_fp suffix indicates for case with 0 flux shifting
def H_drive_TD_no_fp(t, args):
    gate_time = args['gate_time']
    omega_drive = args['omega_drive']
    lamb = args.get('lamb',0)
    alpha = args.get('alpha',0.5)
    return (((1 - math.cos((2*np.pi*t)/gate_time)) * 
           (math.sin(omega_drive * t))) +
           (((lamb/alpha)*(1/gate_time)*2*np.pi*math.sin(2*np.pi*t/gate_time)) * 
           (math.cos(omega_drive * t)))
           )

def time_evolve_no_fp(H_total, psi, time, args=None):
    result = mesolve(H_total, psi, tlist=time, args=args)
    return result.states

def gate_pulse_no_fp(lamb,Ed,gatetime,drivefrequency,psi):
    time = np.linspace(0, gatetime, 101)
    H_sys,evals,evecs = fluxonium_system(sweet_spot,Ej,Ec,El,n_op,phi_op,H_lc,H_i)
    Es = np.real(qubit_energies(H_sys,evals,evecs))
    alpha = (Es[4] - Es[2]) - (Es[2] - Es[0])
    drive_me, trunc_drive_op = matrix_element(drive_op,H_sys,evals,evecs)
    tgate = gatetime
    drive_freq = drivefrequency
    H_total = [Qobj(np.diag(Es)), [Ed*trunc_drive_op, H_drive_TD_no_fp]]
    states = time_evolve_no_fp(H_total, initialState, time, args={'gate_time': tgate, 'lamb': lamb,
                                                    'omega_drive': drive_freq, 
                                                    'alpha': alpha})[1:]
    fidelity = np.abs((expectRho.dag()*states[-1]).full()[0,0])**2
    print('')
    print('lambda',lamb)
    print('drive',Ed)
    print('fidelity',fidelity)
    return fidelity, states