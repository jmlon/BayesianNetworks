#!/usr/bin/env python3


from maxsum import Maxsum
import numpy as np

p = 0.1 # Prob of failure
q = 1-p # Prob of not failure

f124 = np.array([ [ [ q,p ], [ 0,1 ] ], [ [ 0,1 ], [ 0,1 ] ] ])
f235 = f124

f1 = np.array([q,p])
f2 = np.array([q,p])
f3 = np.array([q,p])


m = Maxsum()


m.add_factor('f1'  , prob=f1)
m.add_factor('f2'  , prob=f2)
m.add_factor('f3'  , prob=f3)
m.add_factor('f124', prob=f124)
m.add_factor('f235', prob=f235)

m.add_variable('x1', k=2)
m.add_variable('x2', k=2)
m.add_variable('x3', k=2)
m.add_variable('x4', k=2, state=1) # Fix the given state (evidence)
m.add_variable('x5', k=2, state=0) # Fix the given state (evidence)

m.add_edge('f1', 'x1')
m.add_edge('f2', 'x2')
m.add_edge('f3', 'x3')
m.add_edge('x1', 'f124')
m.add_edge('x2', 'f124')
m.add_edge('x2', 'f235')
m.add_edge('x3', 'f235')
m.add_edge('f124', 'x4')
m.add_edge('f235', 'x5')


p_max, x_max = m.compute_max()
print(f"max prob = {p_max},  max state = {x_max}")
