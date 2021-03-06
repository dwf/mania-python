This is a re-worked Python implementation of the GeneMANIA algorithm (MANIA
is an acronym for **M** ultiple **A** ssociation **N** etwork **I** ntegration 
**A** lgorithm) for predicting gene function using linear regression-based 
kernel fusion and modified Gaussian field label propagation. The details 
of the algorithm as well as the design considerations were first published 
in `this 2008 article <http://genomebiology.com/2008/9/S1/S4>`_ in *Genome
Biology*.

It operates on affinity networks stored as SciPy sparse matrices. 
It requires NumPy_ and SciPy_ to function.

This is based on an earlier MATLAB version by myself and 
`Sara Mostafavi <http://www.cs.toronto.edu/~smostafavi>`_, which is available
on her website (as well as more up-to-date versions accompanying her recent
publications).

This code is released under the 3-clause BSD license.
