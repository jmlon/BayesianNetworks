#!/usr/bin/env python3


from maxsum import Maxsum
import numpy as np

# This network is not a polytree, the algorithm does not
# propagate the messages to the internal nodes
# Polytree: At most one undirected path between two nodes
# The solution is to marry nodes to eliminate the cycle

def marry(f,g):
    if f.shape[0]!=g.shape[0]:
        print("Variables cannot be married")
        exit
    m = np.zeros( (f.shape[0], f.shape[1]*g.shape[1]) )
    for i in range(f.shape[0]):
        joined = 0
        for j in range(f.shape[1]):
            for k in range(g.shape[1]):
                #print(f"{i} {j} {k}")
                m[i][joined] = f[i][j]*g[i][k]
                joined += 1
    #print(m)
    return m


f1 = np.array([0.5,0.5])
f12 = np.array([[0.5,0.5], [0.9,0.1]])
f13 = np.array([[0.8,0.2], [0.2,0.8]])
f234 = np.array([ [ [0.99,0.01], [0.1,0.9] ], [ [0.1,0.9], [0.01,0.99] ] ])

married23 = marry(f12, f13)
fmarried4 = np.array([ [0.99,0.01], [0.1,0.9], [0.1,0.9], [0.01,0.99] ])

m = Maxsum()

m.add_factor('f1'  , prob=f1)

# Cannot add independent nodes
#m.add_factor('f12' , prob=f12)
#m.add_factor('f13' , prob=f13)
# Instead we add the married factor node
m.add_factor('marry23' , prob=married23)

#m.add_factor('f234', prob=f234)
# Have to change factor: Input the joined state of 2(sprinkler) and 3(rain). 4 joined states
m.add_factor('fmarried4', prob=fmarried4)

m.add_variable('cloudy'   , k=2)
m.add_variable('sprinkler-rain', k=4)   # married nodes
#m.add_variable('wet grass', k=2)
m.add_variable('wet grass', k=2, state=0)


m.add_edge('f1', 'cloudy')
m.add_edge('cloudy', 'marry23')
m.add_edge('marry23', 'sprinkler-rain')
m.add_edge('sprinkler-rain', 'fmarried4')
m.add_edge('fmarried4', 'wet grass')



p_max, x_max = m.compute_max()
print(f"max prob = {p_max},  max state = {x_max}")
