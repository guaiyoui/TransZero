import networkx as nx

def tolist(component):
    result = list(component)
    return [int(i) for i in result]

class kecc_cs_my():

    #########################################################################
    #   Initialize by an empty dictionry                                    #
    #########################################################################
    def __init__(self, edge_path):
        self.G = nx.read_edgelist(edge_path)

    #########################################################################
    #   This is finding k-ecc for different values of k until there is    #
    #   not a connected component that contains all query vertices          #
    #########################################################################
    
    def query(self, query_set):
        k = 1
        need_increase_k = True

        while need_increase_k:
            G0 = self.G.copy()
            print("=============")
            print(k)
            # components = nx.k_edge_components(self.G, k)
            components = nx.k_edge_subgraphs(self.G, k)
            print(need_increase_k, k)
            need_increase_k = False
            # components = nx.connected_components(P)
            for component in components:
                # print(k)
                if len(component) == 1:
                    continue

                # check if it still k-ecc in ths subgraph
                G1 = self.G.subgraph(component)
                component_is_promissing = False
                for comp in list(nx.k_edge_components(G1, k)):
                    indicator1 = [str(node) in comp for node in query_set]
                    if all(indicator1):
                        component_is_promissing = True
                
                indicator = [str(node) in component for node in query_set]
                if all(indicator) and component_is_promissing:
                # if all(indicator):
                    k = k+1
                    need_increase_k = True
                    break
                if k>10:
                    break
        
        components = nx.k_edge_components(self.G, k-1)
        # components = nx.connected_components(P)
        for component in components:
            indicator = [str(node) in component for node in query_set]
            if all(indicator):
                return tolist(component)

edge_path = "edge.txt"
kecc_cs = kecc_cs_my(edge_path)
component = kecc_cs.query(query_set = [1, 2])
# component = kecc_cs.query(query_set = [1, 7])
print(component)