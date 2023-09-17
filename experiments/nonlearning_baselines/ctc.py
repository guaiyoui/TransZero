import networkx as nx

def tolist(component):
    result = list(component)
    # return [int(i) for i in result]
    return [i for i in result]

def str2int(l):
    return [int(i) for i in l]

def check_set_connectivity(G, nodes):
    start_node = nodes[0]
    connected = all(nx.has_path(G, start_node, node) for node in nodes[1:])
    return connected

def int2str(l):
    return [str(i) for i in l]

def setdist(G, onenode, nodes):
    dist = 0
    for node in nodes:
        dist += nx.shortest_path_length(G, source=node, target=onenode)
    return dist

def maxdist(G, nodes):
    dist = {}
    for node in G.nodes():
        if node not in nodes:
            dist[node] = setdist(G, node, nodes)
    max_key = max(dist, key=dist.get)
    return max_key, dist[max_key]
    

class ctc_cs_my():

    #########################################################################
    #   Initialize by an empty dictionry                                    #
    #########################################################################
    def __init__(self, edge_path):
        self.G = nx.read_edgelist(edge_path)

    #########################################################################
    #   This is finding k-truss for different values of k until there is    #
    #   not a connected component that contains all query vertices          #
    #########################################################################
    
    def query(self, query_set):
        query_set = int2str(query_set)
        k = 3
        need_increase_k = True
        while need_increase_k:
            P = nx.k_truss(self.G, k)
            need_increase_k = False
            components = nx.connected_components(P)
            for component in components:
                indicator = [node in component for node in query_set]
                if all(indicator):
                    k = k+1
                    need_increase_k = True
        
        P = nx.k_truss(self.G, k-1)
        components = nx.connected_components(P)
        
        component_promising = False
        for component in components:
            indicator = [node in component for node in query_set]
            if all(indicator):
                candidate_set = tolist(component)
                component_promising = True
        # print(candidate_set)
        if not component_promising:
            return None
        subgraph = self.G.subgraph(candidate_set)
        result = []
        result_distance = []
        while check_set_connectivity(subgraph, query_set):
            result.append(candidate_set)
            index, dist = maxdist(subgraph, query_set)
            result_distance.append(dist)

            candidate_set = [x for x in candidate_set if x != index]
            subgraph = self.G.subgraph(candidate_set)
            P = nx.k_truss(subgraph, k-1)

            component_is_promissing = False
            for component in components:
                indicator = [node in component for node in query_set]
                if all(indicator):
                    candidate_set = tolist(component)
                    component_is_promissing = True
            if not component_is_promissing:
                break
            
            subgraph = self.G.subgraph(candidate_set)
        
        # print(result, result_distance)

        max_value = max(result_distance)
        max_index = result_distance.index(max_value)

        return str2int(result[max_index]) 







        # if nx.has_path(self.G, "4", "6"):
        #     print("There is a path between nodes", "4", "and", "6")
        # else:
        #     print("There is no path between nodes", "4", "and", "6")
        # print()
        # print(nx.shortest_path_length(self.G, source="6", target="4"))

        


# edge_path = "edge.txt"
# ctc_cs = ctc_cs_my(edge_path)
# component = ctc_cs.query(query_set = [1, 2])
# # component = ctc_cs.query(query_set = [1, 7])
# print(component)

# a = {}
# a["1"] = 1
# a["3"] = 2
# print(a)

# max_key = max(a, key=a.get)
# print(max_key)
# print(a[max_key])

# candidate_set = [1, 2, 4, 5]


