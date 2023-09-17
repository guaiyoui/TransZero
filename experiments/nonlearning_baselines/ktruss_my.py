import networkx as nx

def tolist(component):
    result = list(component)
    return [int(i) for i in result]

class ktruss_cs_my():

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
        k = 3
        need_increase_k = True
        while need_increase_k:
            P = nx.k_truss(self.G, k)
            need_increase_k = False
            components = nx.connected_components(P)
            for component in components:
                indicator = [str(node) in component for node in query_set]
                if all(indicator):
                    k = k+1
                    need_increase_k = True
        
        P = nx.k_truss(self.G, k-1)
        components = nx.connected_components(P)
        for component in components:
            indicator = [str(node) in component for node in query_set]
            if all(indicator):
                return tolist(component)
        


# edge_path = "edge.txt"
# ktruss_cs = ktruss_cs_my(edge_path)
# component = ktruss_cs.query(query_set = [1, 2])
# # component = ktruss_cs.query(query_set = [1, 7])
# print(component)