#!/usr/bin/env python3


from sumproduct import Sumproduct
import numpy as np


f2  = np.array([0.3,0.6,0.1])
f21 = np.array([[0.2,0.8],[0.7,0.3],[0.4,0.6]])

s = Sumproduct()


s.add_factor('f2'   , prob=f2)
s.add_factor('f21'  , prob=f21)
s.add_variable('x1' , k=2)
s.add_variable('x2' , k=3)

s.add_edge('f2' , 'x2')
s.add_edge('x2' , 'f21')
s.add_edge('f21', 'x1')

marginals = s.compute_marginals()
print(marginals)

