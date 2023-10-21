import torch
from networkx.algorithms.approximation import steinertree
from utils import find_all_neighbors_bynx, MaxMinNormalization
import random
import numpy as np
import time

def subgraph_density(candidate_score, avg_weight):
    weight_gain = (sum(candidate_score)-len(candidate_score)*avg_weight)/(len(candidate_score)**0.5)
    # weight_gain = sum(candidate_score)-len(candidate_score)*avg_weight
    # return weight_gain/(len(candidate_score)**0.1)
    return weight_gain

def subgraph_density_controled(candidate_score, graph_score):
    # modularity
    # weight_gain = sum(candidate_score)/(len(candidate_score)**1)-sum(graph_score)/(len(graph_score)**1)
    # weight_gain = sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1) + 1/(len(candidate_score)**1)
    # weight_gain = sum(candidate_score)/(len(candidate_score)**1)-sum(graph_score)/(len(graph_score)**1) + (len(candidate_score)**0.16)
    # weight_gain = sum(candidate_score)/len(graph_score)-(sum(graph_score)*len(candidate_score)/(len(graph_score)))**2
    
    # density modularity 
    # weight_gain = (sum(candidate_score)/(len(candidate_score)**1.0)-sum(graph_score)/(len(graph_score)**1.0))/(len(candidate_score)**0.1)
    
    # 可以improve的点是，这个average score，可以在一些hop之内的平均，而不是整个graph的平均
    weight_gain = (sum(candidate_score)-sum(graph_score)*(len(candidate_score)**1)/(len(graph_score)**1))/(len(candidate_score)**0.50)
    return weight_gain

# Given query nodes q index, all graph score. return the subgraph with maximum weight gain which contains the query.
# Expand from inner to outside.
# Early stopping (too be added): if the maximum weight of nodes in this branch is smaller than the average weight. drop this branch directly
# If all the nodes have been traversaled, terminated.


def mwg_subgraph_heuristic(query_index, graph_score, graph):

    candidates = query_index

    # find a connected component
    # if len(query_index) > 1:
    #     H = steinertree.steiner_tree(graph, query_index)
    #     candidates = list(H.nodes())

    selected_candidate = candidates
    max_density = -1000

    avg_weight = sum(graph_score)/len(graph_score)

    count = 0
    while True:
        start = time.time()
        neighbors = find_all_neighbors_bynx(candidates, graph)
        # print("neighbor obtain: {:.4f}".format(time.time()-start))
        if len(neighbors) == 0 or count>0.25*len(graph_score):
            break
        
        # select the index with the largest score.
        neighbor_score = [graph_score[i]for i in neighbors]
        i_index = neighbor_score.index(max(neighbor_score))
        # print("neighbor with maximum socre obtain: {:.4f}".format(time.time()-start))
        # i_index = neighbors[max_neighbor_index]
        # i_index = random.randint(0, len(neighbors)-1)
        candidates = candidates+[neighbors[i_index]]

        candidate_score = [graph_score[i]for i in candidates]
        candidates_density = subgraph_density(candidate_score, avg_weight)
        if candidates_density > max_density:
            max_density = candidates_density
            selected_candidate = candidates
        else:
            break
        
        # print("{:.4f}: total time needed: {:.4f}".format(count, time.time()-start))

        count += 1
    print(len(selected_candidate), len(graph_score))
    return selected_candidate



def mwg_subgraph_nocon(query_index, graph_score):

    # graph_score = MaxMinNormalization(graph_score, 0, 100)
    candidates = query_index
    selected_candidate = candidates

    start = time.time()
    graph_score=np.array(graph_score)
    max2min_index = np.argsort(-graph_score)
    end = time.time()
    print("sort time: {:.4f}".format(end-start))
    # avg_weight = sum(graph_score)/len(graph_score)
    max_density = -1000
    candidate_score = [graph_score[i] for i in candidates]

    start = time.time()
    for i in range(int(0.25*len(max2min_index))):
        # if max2min_index[i] not in query_index:
            candidates = candidates+[max2min_index[i]]
            # candidate_score = [graph_score[i] for i in candidates]
            candidate_score = candidate_score+[graph_score[max2min_index[i]]]
            candidates_density = subgraph_density_controled(candidate_score, graph_score)
            if candidates_density > max_density:
                max_density = candidates_density
                selected_candidate = candidates
    end = time.time()
    print("loop time: {:.4f}".format(end-start))
    # for i in range(len(max2min_index)):
    #     if max2min_index[i] not in query_index and graph_score[max2min_index[i]]>avg_weight:

    #         candidates = candidates+[max2min_index[i]]

    # selected_candidate = candidates

    # print(len(selected_candidate), len(graph_score))
    return selected_candidate, max_density



