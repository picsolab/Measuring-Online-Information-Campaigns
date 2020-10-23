import numpy as np
import networkx as nx
import copy

epsilon = 1e-6

'''
first = (1./num_edges) * np.sum(new_degrees[:, 0] * new_degrees[:, 1])
second = np.power((1./num_edges) * np.sum(0.5 * (new_degrees[:, 0] + new_degrees[:, 1])), 2)
third = (1./num_edges) * np.sum(0.5 * (np.power(new_degrees[:, 0], 2) + np.power(new_degrees[:, 1], 2)))
fourth = np.power(((1./num_edges) *  np.sum(0.5 * (new_degrees[:, 0] + new_degrees[:, 1]))), 2)
result = (first - second) / (third - fourth)
'''

def addNoise(mat):
    new_mat = copy.deepcopy(mat)
    new_mat = new_mat.astype('float64')
    if np.std(new_mat[:, 0]) == 0:
        noise = np.random.normal(0, 1, mat.shape[0])
        new_mat[:, 0] = (epsilon * noise) + (1 - epsilon) * new_mat[:, 0]
    if np.std(new_mat[:, 1]) == 0:
        noise = np.random.normal(0, 1, mat.shape[0])
        new_mat[:, 1] = (epsilon * noise) + (1 - epsilon) * new_mat[:, 1]
    return new_mat

def calculateGlobalEfficiency(G):
    n = len(G)
    denom = n*(n-1)
    nw_global_efficiency = 0
    for path_collection in nx.all_pairs_shortest_path_length(G):
        source = path_collection[0]
        for target in path_collection[1]:
            if target != source:
                nw_global_efficiency += 1./path_collection[1][target]
    
    return nw_global_efficiency/denom


def calculateDegreeAssortativity(G, type_1, type_2):
    num_edges = len(G.edges)
    if  num_edges == 0:
        return None
    source_degrees = None
    target_degrees = None
    if type_1 == 'in' and type_2 == 'in':
        source_degrees = G.in_degree
        target_degrees = G.in_degree
    elif type_1 == 'in' and type_2 == 'out':
        source_degrees = G.in_degree
        target_degrees = G.out_degree
    elif type_1 == 'out' and type_2 == 'in':
        source_degrees = G.out_degree
        target_degrees = G.in_degree
    elif type_1 == 'out' and type_2 == 'out':
        source_degrees = G.out_degree
        target_degrees = G.out_degree
    
    degrees = np.zeros(shape=(num_edges, 2))
    i = 0
    for (u, v, d) in G.edges(data=True):
        degrees[i, 0] = source_degrees[u]
        degrees[i, 1] = target_degrees[v]
        i+=1
    
    ## will add noise if necessary
    new_degrees = addNoise(degrees)
    
    ## calculate directed assortativity
    numerator = np.sum((new_degrees[:, 0] - np.mean(new_degrees[:, 0])) * (new_degrees[:, 1] - np.mean(new_degrees[:, 1])))
    denominator = np.sqrt(np.sum(np.power(new_degrees[:, 0] - np.mean(new_degrees[:, 0]), 2))) * np.sqrt(np.sum(np.power(new_degrees[:, 1] - np.mean(new_degrees[:, 1]), 2)))
    #print('numerator:', numerator, 'denominator:', denominator)
    
    if numerator == 0:
        return numerator
    
    return numerator / denominator

def calculateNumericAssortativity(G, attr_name):
    num_edges = len(G.edges)
    if  num_edges == 0:
        return None
    degrees = np.zeros(shape=(num_edges, 2))
    i = 0
    for (u, v, d) in G.edges(data=True):
        degrees[i, 0] = G.nodes[u][attr_name]
        degrees[i, 1] = G.nodes[v][attr_name]
        i+=1
    
    ## will add noise if necessary
    new_degrees = addNoise(degrees)
    
    ## calculate directed assortativity
    numerator = np.sum((new_degrees[:, 0] - np.mean(new_degrees[:, 0])) * (new_degrees[:, 1] - np.mean(new_degrees[:, 1])))
    denominator = np.sqrt(np.sum(np.power(new_degrees[:, 0] - np.mean(new_degrees[:, 0]), 2))) * np.sqrt(np.sum(np.power(new_degrees[:, 1] - np.mean(new_degrees[:, 1]), 2)))
    #print('numerator:', numerator, 'denominator:', denominator, 'result:', result)
    
    if numerator == 0:
        return numerator
    
    return numerator / denominator
