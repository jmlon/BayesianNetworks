#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import networkx as nx
import numpy as np

np.seterr(divide='ignore')  # supress divide by zero error



class Maxsum:


    def __init__(self):
        self._g = nx.Graph()    # The graphical model with variable and factor nodes
        self._toVisit = None    # List of nodes waiting for a visit


    def add_factor(self, name, **kwargs):
        kwargs['factor'] = True
        kwargs['data'] = {} # messages received indexed by sender
        kwargs['lnP'] = np.log(kwargs['prob'])
        self._g.add_node(name, **kwargs)
        
        
    def add_variable(self, name, **kwargs):
        kwargs['variable'] = True
        kwargs['data'] = {} # messages received indexed by sender

        if 'state' in kwargs:
            lnP = -np.inf*np.ones(kwargs['k'])
            lnP[kwargs['state']] = 0
        else:
            lnP = np.zeros(kwargs['k'])        
        kwargs['lnP'] = lnP
        
        self._g.add_node(name, **kwargs)
    

    def add_edge(self, src, dst):
        self._g.add_edge(src,dst)


    def _find_roots(self):
        self._roots = []    
        for n in list(self._g.nodes()):
            if self._g.degree(n)==1:
                self._g.nodes[n]['isRoot'] = True
                self._roots.append(n)
            else:
                self._g.nodes[n]['isRoot'] = False


    def _appendNodeToQueue(self, nodeName):
        if not self._g.nodes[nodeName]['isRoot']:
            if nodeName not in self._toVisit:    
                if len(self._g.nodes[nodeName]['data']) >= self._g.degree(nodeName)-1:            
                    self._toVisit.append(nodeName)


    def _deliver_message(self, toNode, msg):
        self._g.nodes[toNode]['data'][msg['from']] = msg['data']
        self._appendNodeToQueue(toNode)


    def _hasnotReceivedMessage(self, toNode, fromNode):
        return fromNode not in self._g.nodes[toNode]['data']


    def _sumByAxis(self, matrix, vector, axis):
        for k in np.ndindex(matrix.shape):
            matrix[k] += vector[k[axis]]
            #print(f"{k} : {k[axis]} {vector[k[axis]]} ")
        return matrix


    def _maxByAxis(self, matrix, axis, neighbors):
        var = neighbors.pop(axis)
        #print(f"max of {neighbors} with respect to {var}")
        seen = -np.inf * np.ones(matrix.shape[axis])
        for k in np.ndindex(matrix.shape):
            if matrix[k]>seen[k[axis]]:
                seen[k[axis]]=matrix[k]
                coord = list(k)
                coord.pop(axis)
        return seen


    def _consolidate(self):
        # Results
        maxByNode = {}
        for node in self._roots:
            #print(f"{node}:  {self._g.nodes[node]['lnP']}")
            sum = self._g.nodes[node]['lnP']
            for data in self._g.nodes[node]['data'].values():
                sum += data
            maxByNode[node] = np.exp(np.max(sum))
            maxState = np.argmax(sum)
            print(f"{node}: sum={sum}, max={maxByNode[node]}, maxState={maxState}")
        return maxByNode


    def _consolidateVariables(self):
        for node in self._g.nodes:
            if 'variable' in self._g.nodes[node]:
                sum = self._g.nodes[node]['lnP'].copy()
                for source,data in self._g.nodes[node]['data'].items():
                    #print(f"{source} {data}")
                    sum += data
                print(f"{node} sum = {sum}")
                
    

    def compute_max(self):
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
                if self._hasnotReceivedMessage(neighbor, node):
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
                                self._sumByAxis(lnP, data[source], axis)
                            # Maximize resulting probabilities with respect to target neighbor
                            tAxis = list(self._g.neighbors(node)).index(neighbor)
                            msgData = self._maxByAxis(lnP, tAxis, list(self._g.neighbors(node)) )
                
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
        self._consolidateVariables()
        return self._consolidate()


