# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 16:23:56 2023

@author: Lai,Chia-Tso
"""


import numpy as np
from qiskit import *
import random
import time
import matplotlib.pyplot as plt


def quantum_state_tomography(circuit):
    
    n = circuit.num_qubits
    zcircuit = circuit.copy()
    zcircuit.measure(range(n),range(n))
    
    ycircuit = circuit.copy()
    ycircuit.s(range(n))
    ycircuit.h(range(n))
    ycircuit.measure(range(n),range(n))
    
    xcircuit = circuit.copy()
    xcircuit.h(range(n))
    xcircuit.measure(range(n),range(n))
    
    simulator = Aer.get_backend("qasm_simulator") 
    
    zresult = execute(zcircuit,backend=simulator,shots=5000).result()
    zcounts = zresult.get_counts()
    
    nz0=np.zeros(n)
    nz1=np.zeros(n)
    for key, item in zcounts.items():
        for i in range(n):
            if key[i] == "0":
                nz0[i] += item
            else:
                nz1[i] += item
                
    zbar = (nz0-nz1)/(nz0+nz1)
    
    yresult = execute(ycircuit,backend=simulator,shots=5000).result()
    ycounts = yresult.get_counts()
    
    ny0=np.zeros(n)
    ny1=np.zeros(n)
    for key, item in ycounts.items():
        for i in range(n):
            if key[i] == "0":
                ny0[i] += item
            else:
                ny1[i] += item
                
    ybar = (ny0-ny1)/(ny0+ny1)
    
    xresult = execute(xcircuit,backend=simulator,shots=5000).result()
    xcounts = xresult.get_counts()
    
    nx0=np.zeros(n)
    nx1=np.zeros(n)
    for key, item in xcounts.items():
        for i in range(n):
            if key[i] == "0":
                nx0[i] += item
            else:
                nx1[i] += item
                
    xbar = (nx0-nx1)/(nx0+nx1)
    
    
    for i in range(n):
        if np.sqrt(xbar**2+ybar**2+zbar**2)[i] > 1:
            target_vector = np.array([xbar[i],ybar[i],zbar[i]])
            norm_vector = target_vector/np.linalg.norm(target_vector)
            xbar[i] = norm_vector[0]
            ybar[i] = norm_vector[1]
            zbar[i] = norm_vector[2]
        
    r = list(np.sqrt(xbar**2+ybar**2+zbar**2))
    
    for i in range(n):
        if r[i]>1:
            r[i]=1
    
    phi = [0]*n
    for i,x in enumerate(xbar):
        if x == 0 :
            phi[i] = np.pi/2
        else:
            phi[i] = np.arctan(ybar[i]/xbar[i])
            
    theta = list(np.arccos(zbar/np.array(r)))
    
    for i in range(n):
        if phi[i] < 0:
            phi[i] = phi[i]+2*np.pi
            
    for i in range(n):
        if np.isnan(phi[i]):
            phi[i] = np.random.uniform(0,2*np.pi)
            
    for i in range(n):
        if np.isnan(theta[i]):
            theta[i] = np.random.uniform(0,np.pi)

    
    return theta[:-1]+phi[:-1]+r[:-1]



def encoding_mapping(theta,phi,r):
    
    if -1<=r<=1:        
        alpha = 2*np.arccos(r)
    else:
        alpha = 0
    if r<=1:
        beta = np.arcsin(r*np.sin(theta)*np.cos(phi)+np.sqrt(1-r**2)*np.sqrt(1-np.sin(theta)**2*np.cos(phi)**2))
    else:
        beta = np.arcsin(np.sin(theta)*np.cos(phi))
    
    gamma = -np.arctan(np.tan(phi))
    
    return (alpha,beta,gamma)



def theta_mapping(x,xmax,xmin):
    return np.pi/(xmax-xmin)*(x-xmin)
def phi_mapping(x,xmax,xmin):
    return (2*np.pi-0.00001)/(xmax-xmin)*(x-xmin)
def r_mapping(x,xmax,xmin):
    return 1/(xmax-xmin)*(x-xmin)



def reverse_theta_mapping(parameter,xmax,xmin):
    return xmin+(xmax-xmin)/np.pi*parameter
def reverse_phi_mapping(parameter,xmax,xmin):
    return xmin+(xmax-xmin)/(2*np.pi-0.00001)*parameter
def reverse_r_mapping(parameter,xmax,xmin):
    return xmin+(xmax-xmin)/1*parameter



def population_generation(variables,domain,mutation_prob):
    
    encoding_qubit = 0
    if len(variables)%3 == 0:
        encoding_qubit = len(variables)//3
    else:
        encoding_qubit = len(variables)//3+1
        variables += [0]*(3-len(variables)%3)
        
    variables_theta = variables[:encoding_qubit]
    variables_phi = variables[encoding_qubit:encoding_qubit*2]
    variables_r = variables[encoding_qubit*2:encoding_qubit*3]
    
    domain_theta = domain[:encoding_qubit]
    domain_phi = domain[encoding_qubit:encoding_qubit*2]
    domain_r = domain[encoding_qubit*2:encoding_qubit*3]
    
    theta = [theta_mapping(variables_theta[i],domain_theta[i][1],domain_theta[i][0]) for i in range(encoding_qubit)]
    phi = [phi_mapping(variables_phi[i],domain_phi[i][1],domain_phi[i][0]) for i in range(encoding_qubit)]
    r = [r_mapping(variables_r[i],domain_r[i][1],domain_r[i][0]) for i in range(encoding_qubit)]
    
    #map theta,phi,r to encoding angles
    param_array = np.transpose(np.array([theta,phi,r]))
    
    encoding_angles = [encoding_mapping(param_array[row,:][0],param_array[row,:][1],param_array[row,:][2]) for row in range(encoding_qubit)]
    alpha = [encoding_angles[i][0] for i in range(encoding_qubit)]
    beta = [encoding_angles[i][1] for i in range(encoding_qubit)]
    gamma = [encoding_angles[i][2] for i in range(encoding_qubit)]
    
    circuit = QuantumCircuit(encoding_qubit+1,encoding_qubit+1) #one ancilla qubit for entanglement

    #First qubit is the ancilla qubit
    circuit.h(0)
    for i in reversed(range(1,encoding_qubit+1)):  
        circuit.cry(alpha[encoding_qubit-i],0,i)
    for i in reversed(range(1,encoding_qubit+1)):
        circuit.ry(beta[encoding_qubit-i],i)
    for i in reversed(range(1,encoding_qubit+1)):
        circuit.rz(gamma[encoding_qubit-i],i)
        
    circuit.barrier()
    
    #mutation

    #This mutation changes theta
    dice = [np.random.random(1) for i in range(encoding_qubit)]
    for i,prob in enumerate(dice):
        if prob <= mutation_prob:
            mutation_angle = np.random.uniform(-np.pi,np.pi)
            circuit.ry(mutation_angle,i+1)
    
    #This mutation changes phi
    dice = [np.random.random(1) for i in range(encoding_qubit)]
    for i,prob in enumerate(dice):
        if prob <= mutation_prob:
            mutation_angle = np.random.uniform(0,2*np.pi)
            circuit.rz(mutation_angle,i+1)

    #This mutation changes r
    dice = [np.random.random(1) for i in range(encoding_qubit)]
    for i,prob in enumerate(dice):
        if prob <= mutation_prob:
            mutation_angle = np.random.uniform(-np.pi,np.pi)
            circuit.cry(mutation_angle,0,i+1)
            
    #self crossover
    if encoding_qubit >1:
        dice = np.random.random(1)
        if dice <= mutation_prob:
            swap_pair = random.sample(range(1,encoding_qubit+1),2)
            circuit.swap(swap_pair[0],swap_pair[1])
    
        
    return circuit



def selection(variables_popu,fitness,ratio):
    selection_size = round(len(variables_popu)*ratio)
    fitness_scores = [fitness(variables) for variables in variables_popu]
    selection_index = np.argsort(fitness_scores)[-selection_size:]
    parents = np.array(variables_popu)[selection_index]
    
    return parents.tolist()



def crossover(parents,popu_size):
    cross_point = np.random.randint(0,len(parents[0]))
    
    offspring = parents
    
    for i in range(popu_size-len(parents)):
        index = random.sample(range(len(parents)),2)    #take 2 random index to be picked out from the parents population
        offspring.append(parents[index[0]][:cross_point]+parents[index[1]][cross_point:])
        
    return offspring


def quantum_continous_variables_genetic_algorithm(popu_size,variables_domain,fitness,
                                                  selection_ratio,mutation_prob,max_iteration,cutoff_std,target_value):
    
    encoding_qubit = 0
    if len(variables_domain)%3 == 0:
        encoding_qubit = len(variables_domain)//3
    else:
        encoding_qubit = len(variables_domain)//3+1
        variables_domain += [(0,1)]*(3-len(variables_domain)%3)
    
    
    initial_variables =[np.linspace(domain[0],domain[1],popu_size) for domain in variables_domain]
        
    initial_variables = np.transpose(np.array(initial_variables)).tolist()
    
    variables_population = initial_variables
        
    best_individuals = []
    mean_individuals = []
    worst_individuals= []

    iteration=0
    while iteration<max_iteration:
        iteration+=1
        
        population_circ = [population_generation(v,variables_domain,mutation_prob) for v in variables_population]
        
        
        parameters_population=[quantum_state_tomography(circuit) for circuit in population_circ]
        
        
        theta = [parameters[:encoding_qubit] for parameters in parameters_population]
        phi = [parameters[encoding_qubit:2*encoding_qubit] for parameters in parameters_population]
        r = [parameters[2*encoding_qubit:3*encoding_qubit] for parameters in parameters_population]
        
        variables_population=[]
        for i in range(popu_size):
            variables = [reverse_theta_mapping(theta[i][j],variables_domain[j][0],variables_domain[j][1]) for j in range(encoding_qubit)]
            variables += [reverse_phi_mapping(phi[i][j],variables_domain[encoding_qubit+j][0],variables_domain[encoding_qubit+j][1]) for j in range(encoding_qubit)]
            variables += [reverse_r_mapping(r[i][j],variables_domain[2*encoding_qubit+j][0],variables_domain[2*encoding_qubit+j][1]) for j in range(encoding_qubit)]
            variables_population.append(variables)
            
        parents = selection(variables_population,fitness,selection_ratio)
        offspring = crossover(parents,popu_size)
        variables_population = offspring
        
        
        fitness_list = [fitness(variables) for variables in variables_population]
        maximum = np.max(fitness_list)
        best_individuals.append(maximum)
        minimum = np.min(fitness_list)
        worst_individuals.append(minimum)
        mean = np.mean(fitness_list)
        mean_individuals.append(mean)
        
        if maximum > target_value:
            break
            
        std = np.std(fitness_list)
        if std < cutoff_std:
            break
        
    fitness_scores = [fitness(variables) for variables in variables_population]
    optimal = np.max(fitness_scores)
    std = np.std(fitness_scores)
    best_index = np.argsort(fitness_scores)[-1:-11:-1]
    best_parameters = np.array(variables_population)[best_index]
        
    print("solution parameters:",best_parameters)
    print("optimal value:",optimal)
    print("std:",std)
    print("iteration",iteration)
    

    fig = plt.figure(figsize=(10,6))
    ax= fig.add_axes([0,0,1,1])
    ax.plot(range(iteration),worst_individuals)
    ax.plot(range(iteration),mean_individuals)
    ax.plot(range(iteration),best_individuals)


        
    return [best_parameters,optimal]

