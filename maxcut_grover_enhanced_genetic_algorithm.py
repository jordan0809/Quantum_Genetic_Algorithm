# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:25:53 2023

@author: Lai, Chia-Tso
"""
import numpy as np
from qiskit import *
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
import time
from copy import deepcopy

#Import the classical genetic algorithm
from maxcut_genetic_algorithm import maxcut_fit
from maxcut_genetic_algorithm import generate_popu
from maxcut_genetic_algorithm import selection
from maxcut_genetic_algorithm import crossover
from maxcut_genetic_algorithm import mutation
from maxcut_genetic_algorithm import evaluate_std


#Rescale the matrix so that its fitness score can be effectively stored with limited registers
#The non-zero elements would be mapped to the domain [1,2**(n+2)/n**2]
def rescale(weight):
    n = weight.shape[0]
    up_lim = np.max(weight)
    low_lim = np.min(weight[weight != np.min(weight)])
    for i in range(n):
        for j in range(n):
            if i != j:
                weight[i,j] = round(1+(weight[i,j]-low_lim)*(2**(n+2)/n**2-1)/(up_lim-low_lim))
    return weight.astype(int)

    
#Return the two complement of a btistring(for subtracting the fitness threshold)
def two_comp(bitstring):
    flip_bit=''
    for i in range(len(bitstring)):
        if bitstring[i] == "0":
            flip_bit += "1"
        else:
            flip_bit += "0"
    bit_int = int(flip_bit,2)
    bit_int += 1
    two_comp = "{0:b}".format(bit_int)
    return two_comp


#Quantum fitness model for maxcut problems
def maxcut_qfitness(n,gene_weight,best_fitness):
       
    circuit = QuantumCircuit(3*n+2)
    
    #initialize the first generation as uniform distribution
    circuit.h(range(n))
    
    #compute the fitness of a chromosone
    for i in range(gene_weight.shape[0]):
        for j in range(i+1,gene_weight.shape[0]):   #create a a pair of qubits(nodes)
            
            wstring = "{0:b}".format(gene_weight[i,j])
            if len(wstring) < n:
                wstring = "0"*(n-len(wstring))+wstring 
                
                for a in reversed(range(len(wstring))):   
                    if wstring[a] == "1":
                        
                        #compute the addition, but only if the qubit pair is 01 or 10
                        
                        #if the qubit pair is 10 this flipping brings them to 11
                        circuit.x(n-i-1)
                        
                        for b in reversed(range(a)):
                            circuit.mct([n-i-1,n-j-1]+list(range(2*n-1-a,2*n-a+b)),2*n-a+b)
                        circuit.ccx(n-i-1,n-j-1,2*n-a-1)
                    
                        circuit.x(n-i-1)  #flip the bit back
                        
                        #if the qubit pair is 10 this flipping brings them to 11
                        circuit.x(n-j-1)
                    
                        for b in reversed(range(a)):
                            circuit.mct([n-i-1,n-j-1]+list(range(2*n-1-a,2*n-a+b)),2*n-a+b)  
                        circuit.ccx(n-i-1,n-j-1,2*n-a-1)
                    
                        circuit.x(n-j-1)  #flip the bit back
    
    #Compute the two complement of best fitness
    limit = "{0:b}".format(best_fitness)
    if len(limit) < n+1:
        limit = "0"*(n+1-len(limit))+limit
    two_limit = two_comp(limit)
    
    #map the bits of two_limit to the circuit
    for i in range(len(two_limit)):
        if two_limit[i] == "1":
            circuit.x(3*n+1-i)
            
    #compute the summation
    for j in reversed(range(len(two_limit))):
        if two_limit[j] =="1":
            
            for k in reversed(range(j)):
                circuit.mct([3*n+1-j]+list(range(2*n-j,2*n-j+k+1)),2*n-j+k+1)
            circuit.cx(3*n+1-j,2*n-j)
        
    #The 2*n bit determines the sign of the sum (if 0 then greater than limit)
    circuit.x(2*n)
    
    gate = circuit.to_gate()
    gate.name="Maxcut_QFitness"
    return gate
    
    return gate


#Sx gate inverts the phases of states marked by the quantum fitness function
def phase_flip():
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.z(0)
    circuit.x(0)
    circuit.z(0)
    
    gate = circuit.to_gate()
    gate.name = "Sx"
    
    return gate


#Grover's operator for maxcut_qfitness function
def maxcut_Grover_operator(n,gene_weight,best_fitness):
    circuit = QuantumCircuit(3*n+2)
    
    circuit.append(phase_flip().control(1),[2*n,0])
    
    circuit.append(maxcut_qfitness(n,gene_weight,best_fitness).inverse(),range(3*n+2))
    
    circuit.x(range(n))
    circuit.h(n-1)
    circuit.mct(list(range(n-1)),n-1)
    circuit.h(n-1)
    circuit.x(range(n))
    
    circuit.append(maxcut_qfitness(n,gene_weight,best_fitness),range(3*n+2))
    
    gate = circuit.to_gate()
    gate.name = "Grover Operator"
    return gate


#Transform quantum measurements into the input population of the classical GA (list class) 
def quant_to_class_generation(memory):
    
    generation = []
    for i in memory:
        a = list(i)
        b = [int(j) for j in a]
        generation.append(b)
    return generation


#Amplitude Amplification 
def Maxcut_Genetic_Amplitude_Amplification(chromo_size,popu_size,gene_weight,best_fitness,steps):
    circuit = QuantumCircuit(3*chromo_size+2,chromo_size)

    #initialize the state with maxcut_qfitness
    circuit.append(maxcut_qfitness(chromo_size,gene_weight,best_fitness),range(3*chromo_size+2))

    #start the amplitude amplification
    for i in range(steps):
        circuit.append(maxcut_Grover_operator(chromo_size,gene_weight,best_fitness),range(3*chromo_size+2))

    circuit.measure(range(chromo_size),range(chromo_size))
    
    simulator = Aer.get_backend("qasm_simulator")
    result = execute(circuit, backend = simulator, shots=popu_size ,memory=True).result()
    memory = result.get_memory(circuit)
    
    plot_histogram(result.get_counts())
    plt.show()
    
    return quant_to_class_generation(memory)


def Grover_genetic_algorithm(chromo_size,popu_size,weight,
                              selection_rate,mutation_prob,max_iteration):
    
    rescaled_weight = rescale(deepcopy(weight))
    
    #Pick out the best fitness score from the first classical generation
    first_generation = generate_popu(chromo_size,popu_size)
    best_fitness = maxcut_fit(first_generation[0],rescaled_weight)
    for j in range(1,popu_size):
        if maxcut_fit(first_generation[j],rescaled_weight) > best_fitness:
            best_fitness = maxcut_fit(first_generation[j],rescaled_weight)
            
    #Quantum Subroutine
    steps=1    #one time amplification has the best performance
    generation = Maxcut_Genetic_Amplitude_Amplification(chromo_size,popu_size,rescaled_weight,best_fitness,steps)
    
    #Classical Genetic Algorithm
    best_individuals=[]
    mean_individuals=[]
    i=0
    while i<max_iteration:
        i += 1
        parents = selection(generation,selection_rate,maxcut_fit,weight)
        offspring = crossover(parents,popu_size)
        offspring = mutation(offspring,mutation_prob)
        generation = offspring
        
        best = np.max([maxcut_fit(chrom,weight) for chrom in generation])
        best_individuals.append(best)
        mean = np.mean([maxcut_fit(chrom,weight) for chrom in generation])
        mean_individuals.append(mean)
    
        if evaluate_std(generation,maxcut_fit,weight) < 0.01:
            break
            
    std = evaluate_std(generation,maxcut_fit,weight)
    optimal = max([maxcut_fit(chrom,weight) for chrom in generation])
    
    print(generation[:10])
    print("iteration:",i,"std:",std,"optimal fitness:",optimal)
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes([0,0,1,1])
    ax.plot(range(i),best_individuals)
    ax.plot(range(i),mean_individuals)
    

if __name__ == '__main__':
    
    print("------------------------------------------")
    print("Grover Enhanced Genetic Algorithm")
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
    print(Grover_genetic_algorithm(4,100,w,
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
    print(Grover_genetic_algorithm(10,500,W,
                                   selection_rate=0.3,mutation_prob=0.1,
                                   max_iteration=5000))
    end = time.time()
    print("runtime:",end-start,"s")

