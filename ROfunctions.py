import numpy as np
import qutip as qtp
import math
from qutip import *
from functools import cmp_to_key
import pandas as pd
import cmath
from scipy import interpolate
from scipy.integrate import solve_ivp
from scipy.integrate import quad

#used in DispersiveShift.ipynb and ResonatorDynamics.ipynb
def truncate_disshift(hamiltonian, expectoperator):
    evals, evecs = hamiltonian.eigenstates()
    evals_sort = np.argsort(evals)
    evals = evals[evals_sort]
    evecs = evecs[evals_sort]
    expvals = expect(expectoperator, evecs)
    expvals_index = np.argwhere(expvals < 1.2) #returns indices where expval is less than 1.1
    expvals_index_res0 = np.argwhere(expvals < 0.5)
    expvals_index_res1 = np.argwhere(expvals > 0.55)
    expvals_index_res0 = np.array([i for i in expvals_index if (i in expvals_index_res0)] )[:,0]
    expvals_index_res1 = np.array([i for i in expvals_index if (i in expvals_index_res1)] )[:,0]
    
    state0 = evecs[expvals_index_res0[0]] #qubit 0 resonator 0
    state1 = evecs[expvals_index_res1[0]] #qubit 0 resonator 1
    state2 = evecs[expvals_index_res0[1]] #qubit 1 resonator 0
    state3 = evecs[expvals_index_res1[1]] #qubit 1 resonator 1
    states = [state0, state1, state2, state3]
    
    Eq0r0 = hamiltonian.matrix_element(state0.dag(), state0) #energies
    Eq1r0 = hamiltonian.matrix_element(state2.dag(), state2)
    Eq0r1 = hamiltonian.matrix_element(state1.dag(), state1)
    Eq1r1 = hamiltonian.matrix_element(state3.dag(), state3)
    energies = [Eq0r0, Eq0r1, Eq1r0, Eq1r1]
    
    qEdiff_r0 = Eq1r0 - Eq0r0
    qEdiff_r1 = Eq1r1 - Eq0r1
    rEdiff_q0 = Eq0r1 - Eq0r0
    rEdiff_q1 = Eq1r1 - Eq1r0
    frequencies = [qEdiff_r0, qEdiff_r1, rEdiff_q0, rEdiff_q1]
    
    chi_value = (rEdiff_q1 - rEdiff_q0) / 2
    
    truncated_H  = Qobj([[hamiltonian.matrix_element(state0.dag(),state0), hamiltonian.matrix_element(state0.dag(),state1), hamiltonian.matrix_element(state0.dag(),state2), hamiltonian.matrix_element(state0.dag(),state3)],
                        [hamiltonian.matrix_element(state1.dag(),state0), hamiltonian.matrix_element(state1.dag(),state1), hamiltonian.matrix_element(state1.dag(),state2), hamiltonian.matrix_element(state1.dag(),state3)],
                        [hamiltonian.matrix_element(state2.dag(),state0), hamiltonian.matrix_element(state2.dag(),state1), hamiltonian.matrix_element(state2.dag(),state2), hamiltonian.matrix_element(state2.dag(),state3)],
                        [hamiltonian.matrix_element(state3.dag(),state0), hamiltonian.matrix_element(state3.dag(),state1), hamiltonian.matrix_element(state3.dag(),state2), hamiltonian.matrix_element(state3.dag(),state3)]])
    
    return states, energies, frequencies, chi_value, truncated_H

#used in Detuning.ipynb
def truncate_detuning(hamiltonian, expectoperator):
    evals, evecs = hamiltonian.eigenstates()
    evals_sort = np.argsort(evals)
    evals = evals[evals_sort]
    evecs = evecs[evals_sort]
    expvals = expect(expectoperator, evecs)
    expvals_index = np.argwhere(expvals < 1.2) #returns indices where expval is less than 1.1
    expvals_index_res0 = np.argwhere(expvals < 0.5)
    expvals_index_res1 = np.argwhere(expvals > 0.55)
    expvals_index_res0 = np.array([i for i in expvals_index if (i in expvals_index_res0)] )[:,0]
    expvals_index_res1 = np.array([i for i in expvals_index if (i in expvals_index_res1)] )[:,0]
    
    state00 = evecs[expvals_index_res0[0]]
    state01 = evecs[expvals_index_res1[0]]
    state10 = evecs[expvals_index_res0[1]]
    state11 = evecs[expvals_index_res1[1]]
    state20 = evecs[expvals_index_res0[2]]
    state30 = evecs[expvals_index_res0[3]]
    states = [state00, state01, state10, state11, state20, state30]

    Eq0r0 = hamiltonian.matrix_element(state00.dag(), state00)
    Eq0r1 = hamiltonian.matrix_element(state01.dag(), state01)
    Eq1r0 = hamiltonian.matrix_element(state10.dag(), state10)
    Eq1r1 = hamiltonian.matrix_element(state11.dag(), state11)
    Eq2r0 = hamiltonian.matrix_element(state20.dag(), state20)
    Eq3r0 = hamiltonian.matrix_element(state30.dag(), state30)
    energies = [Eq0r0, Eq0r1, Eq1r0, Eq1r1, Eq2r0, Eq3r0]
    
    res_trans = Eq0r1 - Eq0r0
    qubit10_trans = Eq1r0 - Eq0r0
    qubit20_trans = Eq2r0 - Eq0r0
    qubit21_trans = Eq2r0 - Eq1r0
    qubit30_trans = Eq3r0 - Eq0r0
    qubit31_trans = Eq3r0 - Eq1r0
    frequencies = [qubit10_trans, qubit20_trans, qubit21_trans, qubit30_trans, qubit31_trans, res_trans]
    
    chi_value = ((Eq1r1 - Eq1r0) - (Eq0r1 - Eq0r0)) / 2
    
    return states, energies, frequencies, chi_value

def SNR(n_readout, kappa, chi, time):
    epsilon = math.sqrt(n_readout*(chi**2 + (kappa)**2 / 4))
    phi_qb = 2 * math.atan(2 * chi / kappa)
    scaledSNR = []
    rawSNR = []
    
    for t in time:
        M = []
        tps = np.linspace(0,t,1001)
        for tp in tps:
            alpha_plus = epsilon/np.sqrt(kappa)*np.exp(-1j*phi_qb)*(1-2*np.cos(0.5*phi_qb)*np.exp(-1j*chi*tp - 0.5*kappa*tp + 0.5*1j*phi_qb))
            alpha_minus = epsilon/np.sqrt(kappa)*np.exp(1j*phi_qb)*(1-2*np.cos(0.5*phi_qb)*np.exp(1j*chi*tp - 0.5*kappa*tp - 0.5*1j*phi_qb))
            M.append(alpha_plus-alpha_minus)
    
        SNRnum = np.sqrt(kappa)*abs(np.trapz(M))*np.diff(tps)[0]
        SNRdenom = math.sqrt(2 * kappa * t)
        SNR = SNRnum / SNRdenom
        rawSNR.append(SNR)
        scaledSNR.append(SNR / math.sqrt(n_readout))
        
    return scaledSNR, rawSNR