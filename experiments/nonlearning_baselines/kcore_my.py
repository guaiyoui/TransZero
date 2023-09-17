import networkx as nx

def tolist(component):
    result = list(component)
    return [int(i) for i in result]

class kcore_cs_my():

    #########################################################################
    #   Initialize by an empty dictionry                                    #
    #########################################################################
    def __init__(self, edge_path):
        self.G = nx.read_edgelist(edge_path)

    def query(self, query_set):
        k = 3
        need_increase_k = True
        while need_increase_k:
            P = nx.k_core(self.G, k)
            need_increase_k = False
            components = nx.connected_components(P)
            for component in components:
                indicator = [str(node) in component for node in query_set]
                if all(indicator):
                    k = k+1
                    need_increase_k = True
        
        P = nx.k_core(self.G, k-1)
        components = nx.connected_components(P)
        for component in components:
            indicator = [str(node) in component for node in query_set]
            if all(indicator):
                return tolist(component)



# edge_path = "edge.txt"
# kcore_cs = kcore_cs_my(edge_path)
# component = kcore_cs.query(query_set = [1, 2])
# # component = kcore_cs.query(query_set = [1, 7])
# print(component)

# class kcore_cs():

#     #########################################################################
#     #   Initialize by an empty dictionry                                    #
#     #########################################################################
#     def __init__(self, edge_path):
#         self.G = nx.read_edgelist(edge_path)

#         self.core_number = nx.core_number(self.G).items()
#         self.core_number_list = list(self.core_number)
    
#     def query(self, query_set):
        
#         min_corenumber_in_query_set = min([self.core_number_list[node][1] for node in query_set])
#         # print(min_corenumber_in_query_set)
#         candidate_set = []
#         for i in range(len(self.G.nodes())):
#             if self.core_number_list[i][1] >= min_corenumber_in_query_set:
#                 candidate_set.append(str(i))
#         candidate_subgraph = self.G.subgraph(candidate_set)
#         components = nx.connected_components(candidate_subgraph)
#         components_candidate = []
#         for component in components:
#             indicator = [str(node) in component for node in query_set]
#             if all(indicator):
#                 # return component
#                 return tolist(component)
#             if any(indicator):
#                 components_candidate.append(component)
#         all_nodes = set()
#         for component in components_candidate:
#             all_nodes.update(component)
#         print("the input are not in a connected compoent with kcore constraints")
#         return tolist(all_nodes)
    
    

        
        

