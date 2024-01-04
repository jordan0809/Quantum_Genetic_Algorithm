# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 17:43:11 2023

@author: Lai, Chia-Tso
"""

import numpy as np
from qiskit import *
from qiskit.circuit import ParameterVector
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
import time

from maxcut_genetic_algorithm import maxcut_fit
from maxcut_genetic_algorithm import selection
from maxcut_genetic_algorithm import crossover
from maxcut_genetic_algorithm import mutation
from maxcut_genetic_algorithm import evaluate_std
from maxcut_grover_enhanced_genetic_algorithm import quant_to_class_generation


#QAOA circuit
def QAOA_Maxcut(chromo_size,popu_size,gene_weight,layer_num):
    
    n = chromo_size
    Q = -gene_weight
    c = [np.sum(gene_weight[i,:]) for i in range(len(gene_weight))]
    
    gamma = ParameterVector("\u03B3", layer_num)
    beta = ParameterVector("\u03B2", layer_num)

    circuit = QuantumCircuit(n,n)

    #Initialize the circuit in |+> state
    circuit.h(range(n))

    #Evolution of the circuit with trotterization
    for i in range(layer_num):
        
        # Mixer Hamiltonian
        circuit.rx(2*beta[i],range(n))
    
        # Cost Hamiltonian Hc
    
        #Hc single Rz gate 
        for t in range(n):
            circuit.rz(gamma[i]*(c[t]+np.sum(Q[t,:])),t)
    
        #Hc Rzz gate 
        for j in range(n):
            for k in range(j+1,n):
                circuit.cx(j,k)
                circuit.p(0.5*Q[j,k]*gamma[i],k)
                circuit.cx(j,k)
            
        circuit.barrier()
        
    #assign parameters values to the ansatz and conduct measurement

    param_gamma = np.linspace(0,1,layer_num+1)[1:]  #ascending
    param_beta = np.ones(layer_num)-param_gamma     #descending
    param_dict_gamma = {gamma[i]:param_gamma[i] for i in range(layer_num)}
    param_dict_beta = {beta[i]:param_beta[i] for i in range(layer_num)}
    param_dict = {**param_dict_gamma,**param_dict_beta}   ##concatenate two dictionaries by unpacking each dictionary first

    assigned_circuit = circuit.assign_parameters(parameters=param_dict)
    assigned_circuit.measure(range(n),range(n))
    
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(assigned_circuit, backend = simulator, shots=popu_size, memory=True).result()
    memory = result.get_memory(assigned_circuit)
    
    plot_histogram(result.get_counts())
    plt.show()
    
    return quant_to_class_generation(memory)
    
    n = chromo_size
    Q = -gene_weight
    c = [np.sum(gene_weight[i,:]) for i in range(len(gene_weight))]
    
    gamma = ParameterVector("\u03B3", layer_num)
    beta = ParameterVector("\u03B2", layer_num)

    circuit = QuantumCircuit(n,n)

    #Initialize the circuit in |+> state
    circuit.h(range(n))

    #Evolution of the circuit with trotterization
    for i in range(layer_num):
        
        # Mixer Hamiltonian
        circuit.rx(2*beta[i],range(n))
    
        # Cost Hamiltonian Hc
    
        #Hc single Rz gate 
        for t in range(n):
            circuit.rz(gamma[i]*(c[t]+np.sum(Q[t,:])),t)
    
        #Hc Rzz gate 
        for j in range(n):
            for k in range(j+1,n):
                circuit.cx(j,k)
                circuit.p(0.5*Q[j,k]*gamma[i],k)
                circuit.cx(j,k)
            
        circuit.barrier()
        
    #assign parameters values to the ansatz and conduct measurement

    param_gamma = np.linspace(0,1,layer_num+1)[1:]  #ascending
    param_beta = np.ones(layer_num)-param_gamma     #descending
    param_dict_gamma = {gamma[i]:param_gamma[i] for i in range(layer_num)}
    param_dict_beta = {beta[i]:param_beta[i] for i in range(layer_num)}
    param_dict = {**param_dict_gamma,**param_dict_beta}   ##concatenate two dictionaries by unpacking each dictionary first

    assigned_circuit = circuit.assign_parameters(parameters=param_dict)
    assigned_circuit.measure(range(n),range(n))
    
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(assigned_circuit, backend = simulator, shots=popu_size, memory=True).result()
    memory = result.get_memory(assigned_circuit)
    
    plot_histogram(result.get_counts())
    plt.show()
    
    return quant_to_class_generation(memory)


#Combine QAOA and classical GA
def QAOA_quantum_genetic_algorithm(chromo_size,popu_size,gene_weight, layer_num,
                              selection_rate,mutation_prob,max_iteration):
    
    #Quantum Subroutine
    generation = QAOA_Maxcut(chromo_size,popu_size,gene_weight,layer_num)
    
    #Classical Genetic Algorithm
    best_individuals=[]
    mean_individuals=[]
    i=0
    while i<max_iteration:
        i += 1
        parents = selection(generation,selection_rate,maxcut_fit,gene_weight)
        offspring = crossover(parents,popu_size)
        offspring = mutation(offspring,mutation_prob)
        generation = offspring
        
        best = np.max([maxcut_fit(chrom,gene_weight) for chrom in generation])
        best_individuals.append(best)
        mean = np.mean([maxcut_fit(chrom,gene_weight) for chrom in generation])
        mean_individuals.append(mean)
    
        if evaluate_std(generation,maxcut_fit,gene_weight) < 0.01:
            break
            
    std = evaluate_std(generation,maxcut_fit,gene_weight)
    optimal = max([maxcut_fit(chrom,gene_weight) for chrom in generation])
    
    print(generation[:10])
    print("iteration:",i,"std:",std,"optimal solution:",optimal)
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes([0,0,1,1])
    ax.plot(range(i),best_individuals)
    ax.plot(range(i),mean_individuals)


print("------------------------------------------")
print("QAOA Enhanced Genetic Algorithm")
print("------------------------------------------")

#Example 1: 4-node fully connected maxcut problem
w = np.array([[0,1,2,3],
              [1,0,4,5],
              [2,4,0,6],
              [3,5,6,0]])

print("------------------------------------------")
print("4-node Maxcut Problem")
print("------------------------------------------")

start = time.time()
print(QAOA_quantum_genetic_algorithm(4,100,w,layer_num= 10,
                               selection_rate=0.3,mutation_prob=0.1,
                               max_iteration=5000))
end = time.time()
print("runtime:",end-start,"s")

#Example 2: 10-node fully connected maxcut problem
W = np.array([[ 0. ,  7.5,  5.5,  6. ,  8. ,  3.5,  5. ,  9.5, 12.5,  3.5],
              [ 7.5,  0. ,  8. , 14. , 12.5, 10. , 10.5,  9.5, 13. ,  1.5],
              [ 5.5,  8. ,  0. ,  4.5,  8.5, 14.5, 11.5,  9.5, 12. ,  4.5],
              [ 6. , 14. ,  4.5,  0. ,  5. , 14.5, 11. ,  9. , 17. , 13. ],
              [ 8. , 12.5,  8.5,  5. ,  0. ,  8.5,  5.5, 16.5, 16.5, 13. ],
              [ 3.5, 10. , 14.5, 14.5,  8.5,  0. ,  8. ,  6.5,  8.5,  5.5],
              [ 5. , 10.5, 11.5, 11. ,  5.5,  8. ,  0. , 12. ,  8.5, 13.5],
              [ 9.5,  9.5,  9.5,  9. , 16.5,  6.5, 12. ,  0. ,  9. , 11.5],
              [12.5, 13. , 12. , 17. , 16.5,  8.5,  8.5,  9. ,  0. , 14.5],
              [ 3.5,  1.5,  4.5, 13. , 13. ,  5.5, 13.5, 11.5, 14.5,  0. ]])

print("------------------------------------------")
print("10-node Maxcut Problem")
print("------------------------------------------")

start = time.time()
print(QAOA_quantum_genetic_algorithm(10,500,W,layer_num = 10,
                               selection_rate=0.3,mutation_prob=0.1,
                               max_iteration=5000))
end = time.time()
print("runtime:",end-start,"s")