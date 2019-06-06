#!/usr/bin/env python3


from sumproduct import Sumproduct
import numpy as np

f1  = np.array([0.2,0.8])
f2  = np.array([0.6,0.3,0.1])
f123 = np.array([
                    [[.1,.2,.5,.2],[.3,.1,.4,.2],[.2,.1,.3,.4]],
                    [[.3,.2,.4,.1],[.1,.3,.3,.3],[.4,.1,.2,.3]],
                ])

s = Sumproduct()


s.add_factor('f1'   , prob=f1)
s.add_factor('f2'   , prob=f2)
s.add_factor('f123' , prob=f123)
s.add_variable('x1' , k=2)
s.add_variable('x2' , k=3)
s.add_variable('x3' , k=4)

s.add_edge('f1'  , 'x1')
s.add_edge('f2'  , 'x2')
s.add_edge('x1'  , 'f123')
s.add_edge('x2'  , 'f123')
s.add_edge('f123', 'x3')

marginals = s.compute_marginals()
print(marginals)


# Direct computation
joint = np.repeat( (np.repeat(f1,3).reshape((2,3))*f2).flatten(), 4).reshape((6,4)) * f123.reshape((6,4))

# Check the results
assert np.all( np.isclose(marginals['x3'], sum(joint)) )
print(f"{marginals['x3']}\n{sum(joint)}\n")

s.set_evidence('x1', state=0)
marginals = s.compute_marginals()
print(marginals)
assert np.all( np.isclose(marginals['x3'], sum(joint[:3,:])) )

s.set_evidence('x1', state=1)
marginals = s.compute_marginals()
print(marginals)
assert np.all( np.isclose(marginals['x3'], sum(joint[3:,:])) )


