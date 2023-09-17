#!/usr/bin/env python

#########################################################################
#   This code finds k-edge-connected component of a given graph using   #
#   random contraction method suggested by Akiba et al in CIKM paper.   #
#   It also queries vertices in kECC by finding kECC that contains the  #
#   query and its k is maximized.                                       #
#   Author: Mojtaba (Omid) Rezvani                                      #
#########################################################################

import sys
import random
from os.path import isfile, join
import copy

sys.setrecursionlimit(5000)


#########################################################################
#   Vertex class; We consider a dictionary to store the neighbours of   #
#   each vertex. It gives us the chance to access each neighbor in      #
#   constant amount of time, as required in this application.           #
#########################################################################
class Vertex:

    #########################################################################
    #   Initialize by an empty dictionry. Here, we also store an array      #
    #   that contains the list of vertices that have been contracted to     #
    #   this vertex. We also distinguish between number of neighbors and    #
    #   sum of weights of neighbors.                                        #
    #########################################################################
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        self.contracted = [node]
        self.num_neighbors = 0
        self.sum_weights = 0


    #########################################################################
    #   Let it print the neighbors of this vertex                           #
    #########################################################################
    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])


    #########################################################################
    #   Add a neighbor to this vertex                                       #
    #########################################################################
    def add_neighbor(self, neighbor, weight=1):
        # if not self.adjacent.has_key(neighbor):
        if neighbor not in self.adjacent:
            self.num_neighbors += 1
            self.sum_weights += weight
        self.adjacent[neighbor] = weight


    #########################################################################
    #   Check if this vertex has a given neighbor                           #
    #########################################################################
    def has_neighbor(self, neighbor):
        if neighbor in self.adjacent:
            return True
        return False
        # return self.adjacent.has_key(neighbor)


    #########################################################################
    #   Remove a particular neighbor from this list                         #
    #########################################################################
    def remove_neighbor(self, neighbor):
        # if self.adjacent.has_key(neighbor):
        if neighbor in self.adjacent:
            self.num_neighbors -= 1
            self.sum_weights -= self.adjacent[neighbor]
            del self.adjacent[neighbor]


    #########################################################################
    #   Get the list of neighbors of this vertex                            #
    #########################################################################
    def get_connections(self):
        return self.adjacent.keys()  


    #########################################################################
    #   Get a neighbor (the Vertex object of the neighbor)                  #
    #########################################################################
    def get_neighbor(self, i):
        # return self.adjacent.keys()[i]
        return list(self.adjacent.keys())[i]


    #########################################################################
    #   Getting the num of neighbors the hard way -- used for testing       #
    #########################################################################
    def get_num_neighbors_h(self):
        return len(self.adjacent)


    #########################################################################
    #   Getting the number of neighbors using our defined attribute         #
    #########################################################################
    def get_num_neighbors(self):
        # Testing script; It will be commented out
        #if self.num_neighbors != self.get_num_neighbors_h():
        #    print "Error in updating num neighbors"
        return self.num_neighbors


    #########################################################################
    #   Get the id of this vertex                                           #
    #########################################################################
    def get_id(self):
        return self.id


    #########################################################################
    #   Get the weight of edge between this vertex and a given neighbor     #
    #########################################################################
    def get_weight(self, neighbor):
        return self.adjacent[neighbor]


    #########################################################################
    #   Update the weihgt of a an edge. We do not add/remove edges here     #
    #   Be careful, this is dangrouse. We check whether the connection      #
    #   exists in the graph class                                           #
    #########################################################################
    def update_weight(self, neighbor, new_weight):
        if self.has_neighbor(neighbor):
            self.sum_weights -= self.adjacent[neighbor]
            self.adjacent[neighbor] = new_weight
        else:
            print("Shout: The edge does not exit")


    #########################################################################
    #   Increment the weight of an edge by a certain amount.                #
    #   If the edge does not exist, it means that its weight is zero, so    #
    #   we add an edge in this case                                         #
    #########################################################################
    def increment_weight(self, neighbor, inc_value):
        if self.has_neighbor(neighbor):
            self.adjacent[neighbor] += inc_value
        else:
            self.adjacent[neighbor] = inc_value
            self.num_neighbors += 1
        self.sum_weights += inc_value


    #########################################################################
    #   Getting the sum of weights the hard way -- used for testing         #
    #########################################################################
    def get_sum_weights_h(self):
        return sum(self.adjacent.values())


    #########################################################################
    #   Getting the sum of weights using our attribute                      #
    #########################################################################
    def get_sum_weights(self):
        # Testing script; It will be commented out
        #if self.sum_weights != self.get_sum_weights_h():
        #    print "Error in updating num neighbors"
        return self.sum_weights


    #########################################################################
    #   When contraction happened, we add the poor contracted vertex to     #
    #   our list of contracted vertices                                     #
    #########################################################################
    def add_contracted(self, neighbor):
        self.contracted += neighbor.get_contracted()


    #########################################################################
    #   Get the list of vertices that have been contracted to this vertex   #
    #   This is usefull when the vertex is removed from graph and we want   #
    #   to output the community                                             #
    #########################################################################
    def get_contracted(self):
        return self.contracted




#########################################################################
#   Graph class is designed to handle operations on graph               #
#########################################################################
class Graph:

    #########################################################################
    #   Inisitalize the graph with empty set of vertices                    #
    #   We also store the index of each vertex in vert_num                  #
    #########################################################################
    def __init__(self):
        self.vert_dict = {}
        self.vert_num = {}
        self.num_vertices = 0


    #########################################################################
    #   Read the list of edges of a graph from a file                       #
    #########################################################################
    def read_graph(self, graph_file):
        """ Add connections (list of tuple pairs) to graph """

        with open(graph_file) as gf:
            for line in gf:
                e = [int(v) for v in line.split()]
                self.add_edge(e[0], e[1])
        gf.close()


    #########################################################################
    #   Iterate over vertices of the graph                                  #
    #########################################################################
    def __iter__(self):
        return iter(self.vert_dict.values())


    #########################################################################
    #   Add a vertex to the graph                                           #
    #########################################################################
    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        self.vert_num[node] = self.num_vertices - 1
        return new_vertex


    #########################################################################
    #   It returns a vertex with id n, while checking its existence         #
    #########################################################################
    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None


    #########################################################################
    #   Add an edge to the network with a certain weight                    #
    #########################################################################
    def add_edge(self, frm, to, cap = 1):
        """ Add connection between frm and to """

        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cap)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cap)


    #########################################################################
    #   Remove an edge from network. Usefull in decomposition               #
    #########################################################################
    def remove_edge(self, frm, to):
        """ Remove connection between frm and to """

        if self.is_connected(frm, to):
            self.vert_dict[frm].remove_neighbor(self.vert_dict[to])
            self.vert_dict[to].remove_neighbor(self.vert_dict[frm])


    #########################################################################
    #   Print the list of edges of the network along with their weight      #
    #########################################################################
    def print_edges(self):
        for v in self:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                print('( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))


    #########################################################################
    #   Print the edge lists of the network in a form of adjacency list     #
    #########################################################################
    def print_graph(self):
        for v in self:
            print('g.vert_dict[%s]=%s' %(v.get_id(), self.vert_dict[v.get_id()]))


    #########################################################################
    #   Check if two nodes are connected                                    #
    #########################################################################
    def is_connected(self, node1, node2):
        if node1 in self.vert_dict and node2 in self.vert_dict:
            return self.vert_dict[node1].has_neighbor(self.vert_dict[node2])
        else:
            return False


    #########################################################################
    #   Get the weight of an edge between two nodes in the network          #
    #########################################################################
    def get_weight(self, node1, node2):
        if node1 in self.vert_dict and node2 in self.vert_dict:
            return self.vert_dict[node1].get_weight(self.vert_dict[node2])
        else:
            return -1


    #########################################################################
    #   Update the weight of the edge between two nodes in the network      #
    #########################################################################
    def update_weight(self, node1, node2, new_weight):
        if self.is_connected(node1, node2):
            #self.vert_dict[node1].adjacent[self.vert_dict[node2]] = new_weight
            self.vert_dict[node1].update_weight(self.vert_dict[node2], new_weight)
            return self.vert_dict[node1].get_weight(self.vert_dict[node2])
        else:
            return -1


    #########################################################################
    #   Increment the weight of this edge by a certain amount. If the edge  #
    #   is not present, consider the weight to be zero and add it           #
    #########################################################################
    def increment_weight(self, node1, node2, inc_by = 1):
        self.vert_dict[node1].increment_weight(self.vert_dict[node2], inc_by)
        #if self.is_connected(node1, node2):
        #    self.vert_dict[node1].adjacent[self.vert_dict[node2]] += inc_by
        #    return self.vert_dict[node1].adjacent.get(self.vert_dict[node2])
        #else:
        #    self.add_edge(node1, node2, inc_by)
        #    return inc_by


    #########################################################################
    #   Get a list of vertices from dictionary (keys)                       #
    #########################################################################
    def get_vertices(self):
        return self.vert_dict.keys()


    #########################################################################
    #   It detects the connecrted components of the network                 #
    #########################################################################
    def detect_connected_components(self):
        # Find the connected components of the graph
        inList = [0] * self.num_vertices
        components = [[]]
        for v in self:
            if inList[self.vert_num[v.get_id()]] == 1:
                continue
            components.append([])
            components[len(components) - 1].append(v.get_id())
            inList[self.vert_num[v.get_id()]] = 1
            qq = 0
            while qq < len(components[len(components) - 1]):
                vv = components[len(components) - 1][qq]
                for u in self.vert_dict[vv].get_connections():
                    if inList[self.vert_num[u.get_id()]] == 0:
                        components[len(components) - 1].append(u.get_id())
                        inList[self.vert_num[u.get_id()]] = 1
                qq += 1
        return components


    #########################################################################
    #   Finds connected components of the graph and returns the list of ID  #
    #   of connected component of each vertex                               #
    #########################################################################
    def detect_connected_components_inversely(self):
        # Find the connected components of the resulting graph
        inList = [0] * self.num_vertices
        connected_component_of_v = [-1] * self.num_vertices
        c = -1
        for v in self:
            if inList[self.vert_num[v.get_id()]] == 1:
                continue
            c += 1
            connected_component_of_v[self.vert_num[v.get_id()]] = c;
            component = [v.get_id()]
            inList[self.vert_num[v.get_id()]] = 1
            qq = 0
            while qq < len(component):
                vv = component[qq]
                for u in self.vert_dict[vv].get_connections():
                    if inList[self.vert_num[u.get_id()]] == 0:
                        connected_component_of_v[self.vert_num[u.get_id()]] = c;
                        component.append(u.get_id())
                        inList[self.vert_num[u.get_id()]] = 1
                qq += 1
        return connected_component_of_v


    #########################################################################
    #   Removes the edges that the weight of their endpoint is less than k  #
    #   k-core decomposition is a heuristic to make the graph smaller and   #
    #   speedup the process of this algorithm.                              #
    #########################################################################
    def decompose_kcore(self, k):
        # Let's decompose the graph into k-cores

        # Find some vertices to be removed
        to_be_removed = []
        for v in self:
            if v.get_sum_weights() < k:
                to_be_removed.append(v.get_id())

        # Iteratively removed edges with support no less than k
        while len(to_be_removed) > 0:
            u = to_be_removed.pop()
            # if not self.vert_dict.has_key(u):
            if u not in self.vert_dict:
                continue
            # mark neighbours of vertex u
            # for w in self.vert_dict[u].get_connections():
            for w in list(self.vert_dict[u].get_connections()):
                self.remove_edge(u, w.get_id())
                self.remove_edge(w.get_id(), u)
                if (w.get_sum_weights() < k):
                    to_be_removed.append(w.get_id())
            del self.vert_dict[u]
            # Do we want to output u as a community?
            #print u


    #########################################################################
    #   Decompose the k-core by removing one vertex. In most cases,         #
    #   removing one vertex will lead to removal of other vrtices           #
    #########################################################################
    def decompose_kcore_by_vertex(self, k, v):
        # Let's decompose the graph into k-cores

        # Find some vertices to be removed
        to_be_removed = [v]

        # Iteratively removed edges with support no less than k
        while len(to_be_removed) > 0:
            u = to_be_removed.pop()
            # if not self.vert_dict.has_key(u):
            if u not in self.vert_dict:
                continue
            # mark neighbours of vertex u
            # for w in self.vert_dict[u].get_connections():
            for w in list(self.vert_dict[u].get_connections()):
                self.remove_edge(u, w.get_id())
                self.remove_edge(w.get_id(), u)
                if w.get_sum_weights() < k:
                    to_be_removed.append(w.get_id())
            del self.vert_dict[u]
            # Do we want to output u as a community?
            #print u


    #########################################################################
    #   Contract an edge from network. This is a critical part of this      #
    #   algorithm. After each contraction weights are updated and edges are #
    #   moved. We do not use a DisjointSet data structure, as the           #
    #   dictionary supports O(1) edge removal and addition.                 #
    #########################################################################
    def contract_edge(self, node1, node2):
        u = node1.get_id()
        v = node2.get_id()
        # contract the edge (node1, node2) and merge it
        ## first pick the vertex with less # neighbors (w.l.g. u)
        if node1.get_num_neighbors() > node2.get_num_neighbors():
            u, v = v, u
        # print u, v 
        ## then move all neighbors of u to v
        ## consider weight updates, and edges in opposite direction
        self.remove_edge(v, u)
        self.remove_edge(u, v)
        self.vert_dict[v].add_contracted(self.vert_dict[u])
        # for w in self.vert_dict[u].get_connections():
        for w in list(self.vert_dict[u].get_connections()):
            self.increment_weight(v, w.get_id(), self.get_weight(u, w.get_id()))
            self.increment_weight(w.get_id(), v, self.get_weight(u, w.get_id()))
            self.remove_edge(u, w.get_id())
            self.remove_edge(w.get_id(), u)
        del self.vert_dict[u]
        return v


    #########################################################################
    #   Finds k-edge-connected components of the graph using random         #
    #   contraction. In each round, an edge is randomly selected and it is  #
    #   contracted. If the degree of the resulting vertex is less than k,   #
    #   it gets removed from network.                                       #
    #########################################################################
    def decompose_kecc(self, k):
        # First decompose the graph into kcores
        #self.print_graph()
        communities = []
        self.decompose_kcore(k)
        #uv = [('h','g'), ('a','c'), ('j','i'), ('b','c'), ('c','d'), ('i','g')]
        #kk = -1
        while (len(self.vert_dict) > 1): # 1 will be replaced with a condition on the number of edges
            # randomly pick an edge
            #self.print_edges()
            u = random.randrange(0, len(self.vert_dict))
            # u = self.vert_dict.keys()[u]
            u = list(self.vert_dict.keys())[u]
            v = random.randrange(0, self.vert_dict[u].get_num_neighbors())
            v = self.vert_dict[u].get_neighbor(v).get_id()
            #kk += 1
            #u = uv[kk][0]
            #v = uv[kk][1]
            # contract the randomly selected edge
            #print u, v
            if u == v:
                self.remove_edge(self.vert_dict[u], self.vert_dict[v])
                print("An exception happened here; We found a self loop")
                continue
            # v is the remaining vertex after contraction
            v = self.contract_edge(self.vert_dict[u], self.vert_dict[v])
            if self.vert_dict[v].get_sum_weights() < k:
                #print self.vert_dict[v].get_contracted()
                communities.append(self.vert_dict[v].get_contracted())
                self.decompose_kcore_by_vertex(k, v)
            #self.print_graph()
            #print "--------"
            #self.print_edges()
            # remove the updated vertex if its degree is less than k
            #break
            # if the degree of resulting vertex is less than k cut it
        #print "Resulted int he graph"
        #self.print_graph()
        #print self.vert_dict.values()[0].get_contracted()
        return communities


    #########################################################################
    #   This is finding kECC for different values of k until there is not a #
    #   connected component that contains all query vertices                #
    #########################################################################
    def query_kecc(self, query):
        k = 0
        must_increase_k = True
        community = []
        vert_dict_copy = copy.deepcopy(self.vert_dict)
        while must_increase_k:
            k += 1
            # Let's decompose the graph into kECC
            communities = self.decompose_kecc(k)
            rcomponents = {}
            for i in range(0, len(communities)):
                for vertex in communities[i]:
                    rcomponents[vertex] = i
            # if rcomponents.has_key(query[0]):
            if query[0] in rcomponents:
                t = rcomponents[query[0]]
            for q in query:
                # if not rcomponents.has_key(q):
                if q not in rcomponents:
                    must_increase_k = False
                elif rcomponents[q] != t:
                    must_increase_k = False
            if must_increase_k:
                community = communities[rcomponents[query[0]]]
            self.vert_dict = copy.deepcopy(vert_dict_copy)

        return community


class kecc_cs():

    #########################################################################
    #   Initialize by an empty dictionry                                    #
    #########################################################################
    def __init__(self, edge_path):
        self.g = Graph()
        self.g.read_graph(edge_path)

    
    def query(self, query_set):
        return self.g.query_kecc(query_set)

# edge_path = "edge.txt"
# kecc_cs = kecc_cs(edge_path)
# component = kecc_cs.query(query_set = [1, 2])
# # component = kecc_cs.query(query_set = [1, 7])
# print(component)



# g = Graph()
# Test for large networks
#g.read_graph("edges.txt")
#g.decompose_kecc(5)

# g.add_vertex('a')
# g.add_vertex('b')
# g.add_vertex('c')
# g.add_vertex('d')
# g.add_vertex('e')
# g.add_vertex('f')

# g.add_vertex('g')
# g.add_vertex('h')
# g.add_vertex('i')
# g.add_vertex('j')

# g.add_edge('a', 'b', 7)  
# g.add_edge('a', 'c', 9)
# g.add_edge('a', 'f', 14)
# g.add_edge('b', 'c', 10)
# g.add_edge('b', 'd', 15)
# g.add_edge('c', 'd', 11)
# g.add_edge('c', 'f', 2)
# g.add_edge('d', 'e', 6)
# g.add_edge('e', 'f', 9)
# g.add_edge('e', 'g', 9)
# g.add_edge('g', 'h', 9)
# g.add_edge('g', 'i', 9)
# g.add_edge('g', 'j', 9)
# g.add_edge('h', 'i', 9)
# g.add_edge('h', 'j', 9)
# g.add_edge('i', 'j', 9)

# Test for detecting kecc
#g.print_edges()
#g.decompose_kecc(20)
#
#g.print_graph()
#g.print_edges()

# Test for removing edges
#g.print_edges()
#g.print_graph()
#print ""
#g.remove_edge('a', 'b')
#g.print_graph()
#g.print_edges()


# Test for ktruss with edge removals
#g.print_edges()
#print ""
#g.decompose_ktruss(2)
#g.print_edges()

# Test for connected components
#g.print_graph()
#g.decompose_kecc(2)
#g.print_graph()
#components = g.detect_connected_components()
#print components

# Test for community search 
# g.print_graph()
# components = g.query_kecc(['a', 'b'])
# print(components)
# components = g.query_kecc(['i', 'j'])
# print(components)


