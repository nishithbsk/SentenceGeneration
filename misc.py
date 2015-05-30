##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    epsilon = sqrt(6) / sqrt(m + n)
    A0 = random.rand(m, n)
    A0 = A0 * 2 * epsilon - epsilon
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
