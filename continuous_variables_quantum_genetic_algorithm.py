# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:57:18 2023

@author: lAI, CHIA-TSO
"""

import numpy as np
from qiskit import*
import random


#Single qubit quantum tomography for readout
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
        
    theta = list(np.arccos(zbar))
    phi = []
    for i in range(n):
        if np.sin(theta[i]) == 0:
            phi.append(np.random.uniform(0,2*np.pi))
        if xbar[i]/np.sin(theta[i])>1:
            phi.append(0.)
        if xbar[i]/np.sin(theta[i])<-1:
            phi.append(np.pi)
        else:
            phi.append(np.arccos(xbar[i]/np.sin(theta[i])))

    # in case there are still nan values in phi
    for i in range(n):
        if np.isnan(phi[i]):
            phi[i] = np.random.uniform(0,2*np.pi)
    
    return theta+phi


#Define mapping and reverse mapping
def theta_mapping(x,xmax,xmin):
    return np.pi/(xmax-xmin)*(x-xmin)
def phi_mapping(x,xmax,xmin):
    return (2*np.pi-0.00001)/(xmax-xmin)*(x-xmin)
def reverse_theta_mapping(parameter,xmax,xmin):
    return xmin+(xmax-xmin)/np.pi*parameter
def reverse_phi_mapping(parameter,xmax,xmin):
    return xmin+(xmax-xmin)/(2*np.pi-0.00001)*parameter


#Encode the variables in a quantum circuit
def population_generation(variables,domain,mutation_prob,
                     theta_mapping=theta_mapping,phi_mapping=phi_mapping):
    
    num_variables = len(variables)
    variables_theta = variables[:int(num_variables/2)]
    variables_phi = variables[int(num_variables/2):]
    domain_theta = domain[:int(num_variables/2)]
    domain_phi = domain[int(num_variables/2):]
    
    theta = [theta_mapping(variables_theta[i],domain_theta[i][1],domain_theta[i][0]) for i in range(int(num_variables/2))]
    phi = [phi_mapping(variables_phi[i],domain_phi[i][1],domain_phi[i][0]) for i in range(int(num_variables/2))]
    
    circuit = QuantumCircuit(int(num_variables/2),int(num_variables/2))
    
    for i in reversed(range(int(num_variables/2))):  
        circuit.ry(theta[int(num_variables/2)-1-i],i)
    for i in reversed(range(int(num_variables/2))):
        circuit.rz(phi[int(num_variables/2)-1-i],i)
        
    circuit.barrier()
    
    #mutation
    for i in range(int(num_variables/2)):
        dice = np.random.random(1)
        if dice <= 2*mutation_prob:
            mutation_angle = np.random.uniform(-np.pi,np.pi)
            circuit.ry(mutation_angle,i)
    for i in range(int(num_variables/2)):
        dice = np.random.random(1)
        if dice <= mutation_prob:
            mutation_angle = np.random.uniform(0,2*np.pi)
            circuit.rz(mutation_angle,i)
            
    return circuit


#Here are the selection and crossover routines of classical GA
def selection(variables_popu,fitness,ratio):
    
    selection_size = round(len(variables_popu)*ratio)
    fitness_scores = [fitness(variables) for variables in variables_popu]
    selection_index = np.argsort(fitness_scores)[-selection_size:]
    parents = np.array(variables_popu)[selection_index]
    
    return parents.tolist()


#Parents are included in the next generation
def crossover(parents,popu_size):
    
    cross_point = np.random.randint(0,len(parents[0]))
    
    offspring = parents
    
    for i in range(popu_size-len(parents)):
        index = random.sample(range(len(parents)),2)    
        offspring.append(parents[index[0]][:cross_point]+parents[index[1]][cross_point:])
        
    return offspring


#Quantum encoding/mutation + classical GA selection/crossover
def quantum_continous_variables_genetic_algorithm(popu_size,variables_domain,fitness,
                                                  selection_ratio,mutation_prob,max_iteration,cutoff_std,
                                                theta_mapping=theta_mapping,phi_mapping=phi_mapping):
    
    n=len(variables_domain)
    initial_variables=[]
    for domain in variables_domain:
        initial_variables.append(np.random.uniform(domain[0],domain[1]+1.e-6,popu_size))  
        
    initial_variables = np.transpose(np.array(initial_variables)).tolist()
    
    variables_population = initial_variables
        
        
    iteration=0
    while iteration<max_iteration:
        iteration+=1
        
        population_circ = []
        for variables in variables_population:
            population_circ.append(population_generation(variables,variables_domain,mutation_prob,
                     theta_mapping=theta_mapping,phi_mapping=phi_mapping))
        
        #read out theta and phi values
        parameters_population=[]
        for circuit in population_circ:
            parameters_population.append(quantum_state_tomography(circuit))
        
        #map theta and phi back to variables domains
        variables_population=[]
        for parameters in parameters_population:
            theta = parameters[:int(n/2)]
            phi = parameters[int(n/2):]
            variables = [reverse_theta_mapping(theta[i],variables_domain[i][0],variables_domain[i][1]) for i in range(int(n/2))]
            variables += [reverse_phi_mapping(phi[i],variables_domain[int(n/2)+i][0],variables_domain[int(n/2)+i][1]) for i in range(int(n/2))]
            variables_population.append(variables)
            
        parents = selection(variables_population,fitness,selection_ratio)
        offspring = crossover(parents,popu_size)
        variables_population = offspring
        
        std = np.std([fitness(variables) for variables in variables_population])
        if std < cutoff_std:
            break
        
    fitness_scores = [fitness(variables) for variables in variables_population]
    optimal = np.max(fitness_scores)
    best_index = np.argsort(fitness_scores)[-1:-11:-1]
    best_parameters = np.array(variables_population)[best_index]
        
    print("solution parameters:",best_parameters)
    print("optimal value:",optimal)
    print("std:",std)
    print("iteration",iteration)
        
        
    return [best_parameters,optimal]