# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 14:46:18 2023

@author: lai, Chia-Tso
"""

import numpy as np
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dimod import Binary
import networkx as nx
import time

from maxcut_genetic_algorithm import maxcut_fit
from maxcut_genetic_algorithm import selection
from maxcut_genetic_algorithm import crossover
from maxcut_genetic_algorithm import mutation
from maxcut_genetic_algorithm import evaluate_std


#Quantum annealer- optimized maxcut problem
def DWave_maxcut(chromo_size,popu_size,gene_weight):
    
    #generate the graph
    G = nx.Graph()
    for i in range(chromo_size):
        G.add_node(i)
    for j in range(chromo_size):
        for k in range(j+1,chromo_size):
            G.add_edge(j,k)

    for i in range(chromo_size):
        for j in range(i+1,chromo_size):
            G[i][j]["weight"] = gene_weight[i][j]
            
    #formulate the cost function
    maxcut = 0 
    for u,v in G.edges:
        maxcut += 2*gene_weight[u][v]*Binary(u)*Binary(v)
    for v in G.nodes:
        maxcut -= np.sum(gene_weight[v][:])*Binary(v)
    
    #Run the annealing process on the D Wave sampler    
    token = "API Token from Dwave Leap Account"
    qpu = DWaveSampler(token=token)
    sampler = EmbeddingComposite(qpu)
    sampleset = sampler.sample(maxcut,num_reads=popu_size)
    
    record = sampleset.record
    generation = []
    for i in range(len(record)):
        generation += [record[i][0].tolist() for a in range(record[i][2])]  
        #record[i][2] is the nubmer of occurence of the i result
    
    return generation


def DWave_quantum_genetic_algorithm(chromo_size,popu_size,gene_weight,
                              selection_rate,mutation_prob,max_iteration):
    
    #Quantum Subroutine
    generation = DWave_maxcut(chromo_size,popu_size,gene_weight)
    
    #Classical Genetic Algorithm
    i=0
    while i<max_iteration:
        i += 1
        parents = selection(generation,selection_rate,maxcut_fit,gene_weight)
        offspring = crossover(parents,popu_size)
        offspring = mutation(offspring,mutation_prob)
        generation = offspring
    
        if evaluate_std(generation,maxcut_fit,gene_weight) < 0.01:
            break
            
    std = evaluate_std(generation,maxcut_fit,gene_weight)
    #optimal = max([qubo_fit(chrom) for chrom in generation])
    
    print(generation[:10])
    print("iteration:",i,"std:",std)
    
    
print("------------------------------------------")
print("Quantum Annealer Enhanced Genetic Algorithm")
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
print(DWave_quantum_genetic_algorithm(4,100,w,
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
print(DWave_quantum_genetic_algorithm(10,500,W,
                               selection_rate=0.3,mutation_prob=0.1, max_iteration=5000))
                      
end = time.time()
print("runtime:",end-start,"s")


#Example3: 20-node fully connected maxcut problem ()
#Create a symmetric matrix 
weight = np.random.randint(0,10,size=(20,20))
weight = (weight+weight.transpose())/2

for k in range(len(weight)):
    weight[k][k] = 0


print("------------------------------------------")
print("20-node Maxcut Problem")
print("------------------------------------------") 

start = time.time()
print(DWave_quantum_genetic_algorithm(chromo_size=20,popu_size=200,gene_weight=weight,
                              selection_rate=0.3,mutation_prob=0.1,max_iteration=1000))
end = time.time()
print("runtime:",end-start,"s") 
