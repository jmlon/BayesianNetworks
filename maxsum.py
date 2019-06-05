#!/usr/bin/env python3


"""maxsum module: Provides an implementation of the Maxsum algorithm
using message passing.
"""

__version__ = '0.1'
__author__ = 'Jorge Londoño'


import networkx as nx
import numpy as np

np.seterr(divide='ignore')  # Supress divide by zero error when taking np.log(0)



class Maxsum:
    """Maxsum class: Implements the message passing algorithm
    for computing the maximum of a joint probability distribution.
    Optionally, when computing the maximum joint probability
    given some evidence, variables may be fixed to a given state.
    """

    def __init__(self):
        self._g = nx.Graph()    # The graphical model with variable and factor nodes
        self._toVisit = None    # List of nodes waiting for a visit


    def add_factor(self, name, **kwargs):
        """Add a factor node to the graphical model"""
        kwargs['factor'] = True
        kwargs['data'] = {} # Messages received indexed by sender
        kwargs['lnP'] = np.log(kwargs['prob'])
        self._g.add_node(name, **kwargs)
        
        
    def add_variable(self, name, **kwargs):
        """Add a variable node to the graphical model"""
        kwargs['variable'] = True
        kwargs['data'] = {} # Messages received indexed by sender

        if 'state' in kwargs:
            lnP = -np.inf*np.ones(kwargs['k'])
            lnP[kwargs['state']] = 0
        else:
            lnP = np.zeros(kwargs['k'])        
        kwargs['lnP'] = lnP
        
        self._g.add_node(name, **kwargs)
    

    def add_edge(self, src, dst):
        """Add and edge to the graphical model"""
        self._g.add_edge(src,dst)


    def _find_roots(self):
        """Find the roots (terminal nodes) of the graphical model"""
        self._roots = []    
        for n in list(self._g.nodes()):
            if self._g.degree(n)==1:
                self._g.nodes[n]['isRoot'] = True
                self._roots.append(n)
            else:
                self._g.nodes[n]['isRoot'] = False


    def _append_to_queue(self, nodeName):
        """Add a mode to the list of nodes pending to visit"""
        if not self._g.nodes[nodeName]['isRoot']:
            if nodeName not in self._toVisit:    
                if len(self._g.nodes[nodeName]['data']) >= self._g.degree(nodeName)-1:            
                    self._toVisit.append(nodeName)


    def _deliver_message(self, toNode, msg):
        """Deliver a message to a given node"""
        self._g.nodes[toNode]['data'][msg['from']] = msg['data']
        self._append_to_queue(toNode)


    def _has_not_received(self, toNode, fromNode):
        """Check if 'toNode' has already received a message from 'fromNode'"""
        return fromNode not in self._g.nodes[toNode]['data']


    def _sum_by_axis(self, matrix, vector, axis):
        """Add a vector to a given axis of the conditional probability matrix"""
        for k in np.ndindex(matrix.shape):
            matrix[k] += vector[k[axis]]
            #print(f"{k} : {k[axis]} {vector[k[axis]]} ")
        return matrix


    def _max_by_axis(self, matrix, axis, neighbors):
        """Compute the maximum of the probability matrix with respect to the 'axis' variable"""
        var = neighbors.pop(axis)
        #print(f"max of {neighbors} with respect to {var}")
        seen = -np.inf * np.ones(matrix.shape[axis])
        for k in np.ndindex(matrix.shape):
            if matrix[k]>seen[k[axis]]:
                seen[k[axis]]=matrix[k]
                coord = list(k)
                coord.pop(axis)
        return seen


    def _consolidate_variables(self):
        """Compute the final probabilities of state variables and their state"""
        max = {}
        p_max = None
        for node in self._g.nodes:
            if 'variable' in self._g.nodes[node]:
                sum = self._g.nodes[node]['lnP'].copy()
                for source,data in self._g.nodes[node]['data'].items():
                    #print(f"{source} {data}")
                    sum += data
                max[node] = np.argmax(sum)
                p_max = np.exp(np.max(sum))
                #print(f"{node} sum = {sum},  argmax = {np.argmax(sum)},  p_max = {np.exp(np.max(sum))}")
        return p_max, max
                
    

    def compute_max(self):
        """Executes the Maxsum algorithm.
        Returns the maximum joint probality and the corresponding state
        """
        self._find_roots()
        self._toVisit = self._roots.copy() 
        while len(self._toVisit)>0:
            node = self._toVisit.pop(0)
            data = self._g.nodes[node]['data']
            neighbors = set(self._g.neighbors(node))
            #print(f"neighbors = {neighbors}")
            #print(f"Visiting {node}")
                
            # To whom to send messages
            for neighbor in neighbors:
                if self._has_not_received(neighbor, node):
                    neighbors.remove(neighbor)
                    # If node has the required data to send a message to neighbor
                    if set(data.keys()).intersection(neighbors)==neighbors:

                        lnP = self._g.nodes[node]['lnP'].copy()
                        if 'factor' in self._g.nodes[node]:
                            # Factor node
                            # Compute the message for the neighbor
                            for source in neighbors:
                                axis = list(self._g.neighbors(node)).index(source)
                                #print(f"from={source} : axis={axis} - data={data[source]} ")
                                self._sum_by_axis(lnP, data[source], axis)
                            # Maximize resulting probabilities with respect to target neighbor
                            tAxis = list(self._g.neighbors(node)).index(neighbor)
                            msgData = self._max_by_axis(lnP, tAxis, list(self._g.neighbors(node)) )
                
                        else:
                            # Variable node
                            msgData = lnP
                            for source,neigborData in data.items():
                                if neighbor != source:
                                    msgData += neigborData

                        msg = {'from':node,'data':msgData}
                        self._deliver_message(neighbor, msg)
                        print(f"{node} --> {neighbor} : {msg}")
            
                    neighbors.add(neighbor)

        # Results
        return self._consolidate_variables()


