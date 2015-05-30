import numpy as np
import itertools
import sys
import cPickle as pickle

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid
from nn.math import multinomial_sample
from misc import random_weight_matrix

def sigmoid_grad(x):
    f = sigmoid(x)
    return f * (1.0 - f)

class BRNNLM(NNBase):
    """
    Bi-directional RNN
    """
    """
    Implements an RNN language model of the form:
    h(t) = sigmoid(H * h(t-1) + L[x(t)])
    y(t) = softmax(U * h(t))
    where y(t) predicts the next word in the sequence

    U = |V| * dim(h) as output vectors
    L = |V| * dim(h) as input vectors

    You should initialize each U[i,j] and L[i,j]
    as Gaussian noise with mean 0 and variance 0.1

    Arguments:
        L0 : initial input word vectors
        U0 : initial output word vectors
        alpha : default learning rate
        bptt : number of backprop timesteps
    """

    def __init__(self, L0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        param_dims = dict(LH = (self.hdim, self.hdim),
                          RH = (self.hdim, self.hdim),
                          U = (self.vdim, self.hdim * 2))
        # note that only L gets sparse updates
        param_dims_sparse = dict(LL = L0.shape, RL = L0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####
        np.random.seed(rseed) # be sure to seed this for repeatability!
        self.bptt = bptt
        self.alpha = alpha

        # Initialize word vectors
        # either copy the passed L0 and U0 (and initialize in your notebook)
        # or initialize with gaussian noise here
        self.sparams.LL = np.random.randn(*L0.shape) * np.sqrt(0.1)
        self.sparams.RL = np.random.randn(*L0.shape) * np.sqrt(0.1)
        self.params.U = np.random.randn(self.vdim, self.hdim*2) * np.sqrt(0.1)

        # Initialize H matrix, as with W and U in part 1
        self.params.LH = random_weight_matrix(self.hdim, self.hdim)
        self.params.RH = random_weight_matrix(self.hdim, self.hdim)

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys):
        """
        Accumulate gradients, given a pair of training sequences:
        xs = [<indices>] # input words
        ys = [<indices>] # output words (to predict)

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.H += (your gradient dJ/dH)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # update row

        Per the handout, you should:
            - make predictions by running forward in time
                through the entire input sequence
            - for *each* output word in ys, compute the
                gradients with respect to the cross-entropy
                loss for that output word
            - run backpropagation-through-time for self.bptt
                timesteps, storing grads in self.grads (for H)
                and self.sgrads (for L,U)

        You'll want to store your predictions \hat{y}(t)
        and the hidden layer values h(t) as you run forward,
        so that you can access them during backpropagation.

        At time 0, you should initialize the hidden layer to
        be a vector of zeros.
        """

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        lhs = np.zeros((ns+1, self.hdim))
        lzs = np.zeros((ns, self.hdim))
        rhs = np.zeros((ns+1, self.hdim))
        rzs = np.zeros((ns, self.hdim))
        # predicted probas
        ps = np.zeros((ns, self.vdim))

        #### YOUR CODE HERE ####

        ##
        # Forward propagation left -> right
        for i in xrange(ns):
            x = xs[i]
            h = lhs[i - 1]
            lzs[i] = self.params.LH.dot(h) + self.sparams.LL[x]
            lhs[i] = sigmoid(lzs[i])

        ##
        # Forward propagation right -> left
        for i in reversed(xrange(ns)):
            x = xs[i]
            h = rhs[i + 1]
            rzs[i] = self.params.RH.dot(h) + self.sparams.RL[x]
            rhs[i] = sigmoid(rzs[i])

        for i in xrange(ns):
            ps[i] = softmax(self.params.U.dot(np.concatenate((lhs[i], rhs[i]))))

        ldelta_1 = []
        rdelta_1 = []
        ##
        # Backward propagation for U and delta_1
        for i in xrange(ns):
            delta_2 = ps[i]
            delta_2[ys[i]] -= 1.0
            self.grads.U += np.outer(delta_2, np.concatenate((lhs[i], rhs[i])))
            delta_1 = self.params.U.T.dot(delta_2) * sigmoid_grad(np.concatenate((lzs[i], rzs[i])))
            ldelta_1.append(delta_1[:self.hdim])
            rdelta_1.append(delta_1[self.hdim:])
        ##
        # right -> left backward propogation
        delta_1 = np.zeros(self.hdim)
        for i in reversed(xrange(ns)):
            delta_1 += ldelta_1[i]
            self.sgrads.LL[xs[i]] = delta_1
            self.grads.LH += np.outer(delta_1, lhs[i-1])
            if i > 0:
                delta_1 = self.params.LH.T.dot(delta_1) * sigmoid_grad(lzs[i-1])
        ##
        # left -> right backward propogation
        delta_1 = np.zeros(self.hdim)
        for i in xrange(ns):
            delta_1 += rdelta_1[i]
            self.sgrads.RL[xs[i]] = delta_1
            self.grads.RH += np.outer(delta_1, rhs[i+1])
            if i < ns-1:
                delta_1 = self.params.RH.T.dot(delta_1) * sigmoid_grad(rzs[i+1])

        #### END YOUR CODE ####

    def save_parameters(self):
        if hasattr(self.alpha, "__call__"):
            alphaStr = "annealing"
        else:
            alphaStr = str(self.alpha).replace('.', '')

        filename = "hdim_" + str(self.hdim) + "_vdim_" + str(self.vdim) + "_alpha_" \
                    + alphaStr
        stack = [self.params.U, self.sparams.LL, self.sparams.RL, self.params.LH, self.params.RH]
        pickle.dump(stack, open(filename, 'w'))

    def load_parameters(self, filename):
        U, LL, RL, LH, RH = pickle.load(open(filename))
        self.params.U = U
        self.sparams.LL = LL
        self.sparams.RL = RL
        self.params.LH = LH
        self.params.RH = RH

    def grad_check(self, x, y, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        NNBase.grad_check(self, x, y, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def compute_seq_loss(self, xs, ys):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        #### YOUR CODE HERE ####
        ns = len(xs)
        lhs = np.zeros((ns+1, self.hdim))
        rhs = np.zeros((ns+1, self.hdim))
        for i in xrange(ns):
            x = xs[i]
            h = lhs[i - 1]
            lhs[i] = sigmoid(self.params.LH.dot(h) + self.sparams.LL[x])
        for i in reversed(xrange(ns)):
            x = xs[i]
            h = rhs[i + 1]
            rhs[i] = sigmoid(self.params.RH.dot(h) + self.sparams.RL[x])
        for i in xrange(ns):
            y = ys[i]
            y_hat = softmax(self.params.U.dot(np.concatenate((lhs[i], rhs[i]))))
            J -= np.log(y_hat[y])

        #### END YOUR CODE ####
        return J


    def compute_loss(self, X, Y):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], np.ndarray): # single example
            return self.compute_seq_loss(X, Y)
        else: # multiple examples
            return np.sum([self.compute_seq_loss(xs,ys)
                       for xs,ys in itertools.izip(X, Y)])

    def compute_mean_loss(self, X, Y):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y)
        ntot = np.sum(map(len,Y))
        return J / float(ntot)


    def generate_sequence(self, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = list of index of start words (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = init # emitted sequence

        #### YOUR CODE HERE ####
        h = np.zeros(self.hdim)
        for x in ys:
            z = self.params.H.dot(h) + self.sparams.L[x]
            h = sigmoid(z)
        while ys[-1] != end:
            x = ys[-1]
            z = self.params.H.dot(h) + self.sparams.L[x]
            h = sigmoid(z)
            y_hat = softmax(self.params.U.dot(h))
            y = multinomial_sample(y_hat)
            J -= np.log(y_hat[y])
            ys.append(y)


        #### YOUR CODE HERE ####
        return ys, J
