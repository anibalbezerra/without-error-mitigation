#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pennylane.utils
from matplotlib import pyplot as plt

import pandas as pd
import scipy as scp

from pennylane import numpy as np
import matplotlib.pyplot as plt
from math import pi

import os
os.environ["OMP_NUM_THREADS"] = "6"
from pennylane import numpy as np
import pennylane as qml
from pennylane.templates import RandomLayers
from pennylane import broadcast
from pennylane import expval, var, device

from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random as folding
from mitiq.zne.inference import AdaExpFactory
from pennylane.transforms import mitigate_with_zne
from mitiq.zne import execute_with_zne
from mitiq.zne.scaling import fold_gates_at_random as folding
from qiskit.providers.fake_provider import FakeLima
from qiskit.providers.aer.noise import NoiseModel

from ase.build import bulk
from ase.lattice import FCC
from pennylane import numpy as np
from qiskit.providers.fake_provider import FakeLima
from multiprocessing import pool

from functools import wraps
from time import time

from qiskit import Aer
from optimparallel import minimize_parallel
from qiskit import IBMQ


# In[3]:


class KP:
    def __init__(self, ep0=1.5292, delta=0.340, a=5.653, Ep=22.71, verbose=1, npoints_path=15):

        self.RY =1
        self.ep0= 1.5292
        self.delta = 0.341
        self.a=a
        self.verbose = verbose
        self.npoints_path = npoints_path
        self.Ep = 22.71
        self.gamma1 = 7.03
        self.gamma2 = 2.33
        self.gamma3 = 3.03
        self.mass = 0.0665
        self.P = 9.75
        
    def lattice(self,atom = 'Si',structure = 'diamond', path = 'XGL'):
        self.atom =atom
        self.structure = structure
        self.pathPoints = path
        
        struc = bulk(self.atom,self.structure, a= self.a)
        self.lat = struc.cell.get_bravais_lattice()
        
        self.path = struc.cell.bandpath(self.pathPoints, npoints=self.npoints_path)
        pt = FCC(self.a).bandpath(self.pathPoints, npoints = self.npoints_path)
       
        self.p_axis = pt.get_linear_kpoint_axis(eps = 1e-5)[0]
        self.p_labels_vert = pt.get_linear_kpoint_axis(eps = 1e-5)[1]
        self.p_labels_stg = pt.get_linear_kpoint_axis(eps = 1e-5)[2]
        
        if self.verbose == 1:
            print('Symmetry points =  ',list(self.lat.get_special_points()))
            print('k-vector-path = ', self.p_axis)
            print('Path vertices = ', self.p_labels_vert)
            print('Path labels = ', self.p_labels_stg)

        #creating a dataframe to stroe the path
        self.data = pd.DataFrame(self.path.kpts,columns=['kx','ky','kz'])
        self.data['k'] = np.sqrt((self.data['kx'])**2+(self.data['kx'])**2+(self.data['kx'])**2)
        
    def Hamilton_GaAs(self, index = 0, nqbits=3, custom_lattice = False)->np.array:
        if not custom_lattice:
            self.lattice()
        else:
            print("define a custom lattice")
        
        
        k = self.data.iloc[index].k
        kx = self.data.iloc[index].kx
        ky = self.data.iloc[index].ky
        kz = self.data.iloc[index].kz

        kmais = kx + 1j * ky
        kmenos = kx - 1j * ky

        e = 1 / self.mass - (self.Ep/3) * ((2 / self.ep0) + 1 / (self.ep0 + self.delta))
        P = 9.75
        
        Q = - ((self.gamma1 + self.gamma2)*(kx**2 - ky**2) - (self.gamma1 - 2*self.gamma2)*kz**2 )
        R = -np.sqrt(3)*(self.gamma2*(kx**2 - ky**2) + 2*1j*self.gamma3*kx*ky)
        Ec = self.ep0 + e*(kx**2 + ky**2 + kz**2)
        Pz = self.P * kz

        T = - ((self.gamma1 - self.gamma2)*(kx**2 + ky**2) + (self.gamma1 + 2*self.gamma2)*kz**2 )
        S = 1j * (2*np.sqrt(3) * self.gamma3 * kz * kmenos)
        Pmais = self.P * kmais / np.sqrt(2)
        Pmenos = self.P * kmenos / np.sqrt(2)

        H = np.zeros((4,4), dtype=np.complex64)               
        
        H[0,0] = T/2
        H[0,1] = - S
        H[0,2] = - 1j * (T-Q) / np.sqrt(2)
        H[0,3] = - 1j * np.sqrt(2/3) * Pz
        H[1,1] = Q/2
        H[1,2] = -1j * np.conjugate(S) / np.sqrt(2)
        H[1,3] = - Pmais

        H[2,2] = ((Q+T)/2 - self.delta) / 2
        H[2,3] = -Pz  / np.sqrt(3)

        H[3,3] = Ec/2

        self.H = H + np.transpose(np.conjugate(H))
        return self.H[:2**nqbits,:2**nqbits]

       
kp_instance = KP()


kp_instance.lattice()


hamiltonian_matrix = kp_instance.Hamilton_GaAs()


print("Hamiltonian matrix for GaAs:")
print(hamiltonian_matrix)

print("Path data:")
print(kp_instance.data)



# In[21]:


class SSVQE:
    
    def __init__(self, index=0, ep0=1.5292, delta=0.752, a=5.653, num_qubits=2, verbose=1, npoints_path=15, machine=None, num_layers=2, num_excited=0):

        
        
        self.time = []
        self.num_qubits= num_qubits
        self.index=index
        self.verbose=verbose
        self.npoints_path= npoints_path
        self.ep0=ep0
        self.delta=delta
        self.a=a
        self.HKP= KP(ep0=self.ep0, delta=self.delta, a=self.a, verbose=self.verbose, npoints_path= self.npoints_path)
        self.HH = self.HKP.Hamilton_GaAs(index=self.index,nqbits =self.num_qubits)
        self.data = self.HKP.data
        self.machine= machine
        self.num_layers= num_layers
        self.num_excited= num_excited
        
        
        
        
        self.dev = qml.device("default.qubit", wires = self.num_qubits)
       
            
        if self.verbose ==1:
            print('Hamiltonian:')
            print(self.HH)
            print('Number of Qubits:', self.num_qubits)
            print('Running on ', machine)
            self.epoch2print =50
        elif self.verbose == 0:
            self.epoch2print = 200
            
            
            
    def states(self, nq=None)->list:
        if not nq:
            nq = self.num_qubits
        st = []
        for i in range(2 ** nq):
            formStr = '0' + repr(nq) + 'b'
            st.append(format(i, formStr))
        return st
    
    
    
    def ansatz(self, params, wires, state_idx=0)-> None:
        state = self.states(nq =self.num_qubits)
        
        for ind_qb, qb in enumerate(list(state[state_idx])):
            if qb =='1':
                qml.PauliX(wires=wires[ind_qb])
                
                
        qml.templates.StronglyEntanglingLayers(params, wires = range(self.num_qubits))
        
    def total_cost(self,params, **kwargs):
        single_cost = kwargs.get('single_cost')
        w = kwargs.get('w')
        cost = 0
        for state_idx in range(2**self.num_qubits):
            cost +=w[state_idx]*single_cost(params, state_idx=state_idx)
        return cost
    
    
    def run(self, params, max_iterations =1000, conv_tol = 1e-5, n_excited = 1, optimizer = 'Adam', opt_rate = 0.1,method= 'simple',processes=None):
        self.n_excited = n_excited
        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.params = params
        
        coeffs, obs_list = qml.pauli.conversion._generalized_pauli_decompose(self.HH)
        
        
        obs_groupings = qml.pauli.group_observables(observables=obs_list, grouping_type="qwc", method="rlf")
        print("Number of required measurements after optimization", len(obs_groupings))
        
        
        H = qml.Hamiltonian(coeffs, obs_list)
        
        
        
        eigen,_ = np.linalg.eigh(self.HH)
        energies = np.zeros(2)
        
        
        if self.machine == "PennyLane_statevector":
            self.dev = qml.device("default.qubit", wires = self.num_qubits)
        elif self.machine == "PennyLane_simulator":
            self.dev = qml.device("default.mixed", wires = self.num_qubits, shots= 1024)
        elif self.machine == "qasm_Statevector":
            self.dev = qml.device('qiskit.aer', wires=self.num_qubits, backened='aer_simulator_Statevector')
            
            
            
        self.nonoiseDev = qml.device("default.qubit", wires = self.num_qubits)
        self.nonoisesingle_cost = qml.ExpvalCost(self.ansatz, H , self.nonoiseDev, optimize = True)
        
        
        
        
        
        
        self.single_cost = qml.ExpvalCost(self.ansatz, H, self.dev, optimize = True)
        
        
        self.w = np.arange(2**self.num_qubits, 0, -1)
        energy_optimized, ind, energies_during_opt = self.optimize(
            params, optm=optimizer, opt_rate=opt_rate, conv_tol=conv_tol)

        results = {
            'energy_optimized': energy_optimized,
            'param_optimized': params,
            'all_energies': energies_during_opt,
            'number_of_cycles': ind,
            'convergence_tolerance': self.conv_tol
        }
        return results
    def timing(f):
        @wraps(f)
        def wrap(self,*args,**kw):
            ts=time()
            result = f(self,*args,**kw)
            te = time()
            dt = te-ts
            print('function_time'%(f.__name__,args,kw,te-ts))
            self.time.append({'seconds': dt})
            return result
        return wrap
    @timing
    def optimize(self,params,optm = 'Adam', opt_rate = 0.1, conv_tol = 1e-3):
        self.params = params
        opt = self.optimizers(optm,opt_rate)
        costs = []
        
        conv = 10
        KP_callback_energies = np.zeros((self.n_excited+1,self.max_iterations))
        low_energy = 1e3
        param_low_energy = params
        
        prev = 1e3
        for i in range(self.max_iterations):
            self.params,prev_energy = opt.step_and_cost(self.total_cost, self.params, single_cost = self.single_cost, w= self.w)
            energy = self.total_cost(self.params, single_cost = self.single_cost, w = self.w)
            conv = np.abs(prev_energy - energy)
            
            
            KP_callback_energies[0,i] = prev_energy
            
            
            if energy < low_energy:
                print('Energy lowered, Actual low energy = ', energy)
                low_energy = energy
                param_low_energy = self.params
                
                
            if i%1 == 0:
                print('iteration = ', i)
                print('conv = ', conv)
                print('energy = ', energy)
                   
            if(conv<= self.conv_tol):
                break
                
                
        print("optimization converged after ", i , "cycles1")
        
        if energy > low_energy:
            self.params = param_low_energy
            print('Energy non_converged after maximum iterations... ')
            
            
        energy_optimized = np.zeros(self.num_excited + 1)
        for exc in range(self.num_excited+1):
            energy_optimized[exc] = self.single_cost(self.params, state_idx = exc)
            
            
        return enrgy_optimized, i, KP_callback_energies
    
    def optimizers(self, optm, opt_rate, eps=1e-8):
        if optm == 'Adam':
            opt = qml.AdamOptimizer(stepsize=opt_rate, eps=eps)
        elif optm == 'Adagrad':
            opt = qml.AdagradOptimizer(stepsize=opt_rate, eps=eps)
        
        else:
            raise ValueError("Unsupported optimizer: {}".format(optm))
        return opt
    
        

            



        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


num_layers = 5
num_qubits = 2
point_path = 1
num_excited = num_qubits**2 - 1
params = np.random.uniform(low=0, high=2*np.pi, size=(num_layers, num_qubits, 3))

eVQE = []
eExact = []

for i in range(point_path):
    VQE = SSVQE(index=i, verbose=0, num_qubits=num_qubits,
               num_excited=num_excited, npoints_path=point_path, machine="default.qubit",
               num_layers=num_layers)
    results = VQE.run(params, max_iterations=2000,
                     conv_tol=1e-4, optimizer='Adam', n_excited=num_excited, opt_rate=0.01,
                     method='simple')
    params = results.get('param_optimized')
    newEnergs = results.get('energy_optimized')

    eVQE.append(newEnergs)
    eExact.append(results.get('energy_exact'))


# In[ ]:




