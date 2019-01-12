# coding: utf-8

import numpy as np
from tqdm import tqdm
from abc import abstractmethod


class BaseHMM(object):
    def __init__(self, n_obs, n_state, max_iters=100):
        """
        Base class of HMM, derived by continuous and discrete HMMs.

        :param n_obs: number of available choices of observations.
        :param n_state: number of available choice of hidden states.
        :param max_iters: max number of iteration when training HMM to get optimal parameters.

        Attribution:

        n_obs: number of available choices of observations.
        n_state: number of available choice of hidden states.
        max_iters: max number of iteration when training HMM to get optimal parameters.
        trained: whether this model has been trained
        start_prob: start probability of each hidden state, 1-d array with shape (n_state,)
        transfer_prob: transfer probability between two hidden state from one step to next step, array with
        shape (n_state, n_state)
        """
        self.n_obs = n_obs
        self.n_state = n_state
        self.max_iters = max_iters

        self.trained = False  # whether this model has been trained, train method must be processed before testing
        self.start_prob = None  # \pi
        self.transfer_prob = None  # A_{jk}, j is the hidden state of `n-1` step whiling k is that of `n` step

    def initialize(self, X=None):
        """
        Initialize parameters of HMM before training

        :param X: total observations inputs. May be useless for some kind of HMMs
        """
        self.start_prob = np.ones(self.n_state, dtype=np.float32) / self.n_state
        self.transfer_prob = np.ones(shape=(self.n_state, self.n_state), dtype=np.float32) / self.n_state

    @abstractmethod
    def emit_prob(self, x):
        """
        Calculate the probability of observation being exactly `x` with current model parameters. Continuous and
        discrete HMMs' differences are mainly included in this method
        :param x: observation, 1-d array
        :return: conditional probability of `p(x|z)`, 1-d array with shape (n_state,)
        """
        pass

    def forward(self, X, Z):
        """
        Forward probability recursive method

        :param X: observation with shape (n_seq, 1)
        :param Z: hidden states with shape (n_seq, n_state). In training process, it is a matrix only contains 1.
        :return: alpha matrix with shape (n_seq, n_state), forward probability of `n_state` kinds of states at each
        sequence step.
        """
        assert X.ndim == 2
        assert Z.ndim == 2

        n_seq = len(X)

        alpha = np.zeros(shape=(n_seq, self.n_state), dtype=np.float32)  # \alpha(z_{nj})
        alpha[0] = self.start_prob * self.emit_prob(X[0]) * Z[0]
        # normalization, values of alpha matrix prone to be zero with iteration
        c = np.zeros(n_seq, dtype=np.float32)
        c[0] = alpha[0].sum()
        alpha[0] = alpha[0] / c[0]

        for i in range(1, n_seq):
            alpha[i] = np.dot(alpha[i - 1], self.transfer_prob) * self.emit_prob(X[i]) * Z[i]
            c[i] = alpha[i].sum()
            try:
                alpha[i] = alpha[i] / c[i]
            except ZeroDivisionError:
                pass
        return alpha, c

    def backward(self, X, Z, c):
        """
        Backward probability recursive method

        :param X: observation with shape (n_seq, 1)
        :param Z: hidden states with shape (n_seq, n_state). In training process, it is a matrix only contains 1.
        :return: beta matrix with shape (n_seq, n_state), backward probability of `n_state` kinds of states at each
        sequence step.
        """
        assert X.ndim == 2
        assert Z.ndim == 2

        n_seq = len(X)

        beta = np.zeros(shape=(n_seq, self.n_state), dtype=np.float32)  # \beta(z_{nj})
        beta[n_seq - 1] = 1

        for i in range(n_seq - 2, -1, -1):
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transfer_prob.T) * Z[i]
            try:
                beta[i] = beta[i] / c[i + 1]
            except ZeroDivisionError:
                pass
        return beta

    def train(self, X, Z=None):
        """
        Train HMM with input `X`.

        :param X: total inputs.
        """
        n_seq = len(X)

        # Initialize model parameters
        self.initialize(X)

        if Z is None:
            Z = np.ones(shape=(n_seq, self.n_state), dtype=np.int32)  # hidden state of each element in the sequence
        else:
            tmp = np.zeros(shape=(n_seq, self.n_state), dtype=np.int32)
            tn = np.arange(0, n_seq, dtype=np.int32)
            for tx, ty in zip(tn, Z):
                tmp[tx, ty] = 1
            Z = tmp

        # Iteration begins
        for _ in tqdm(range(self.max_iters)):
            # E step
            alpha, c = self.forward(X, Z)
            beta = self.backward(X, Z, c)

            gamma = alpha * beta  # \gamma value at each step in sequence
            # calculate sum of \xi matrix from step 2 to fianl step.
            # this is because updating parameters only need the sum value of \xi
            xi = np.zeros(shape=(self.n_state, self.n_state), dtype=np.float32)
            for i in range(1, n_seq):
                xi += np.outer(alpha[i - 1], beta[i] * self.emit_prob(X[i])) * self.transfer_prob / c[i]

            # M step
            self.start_prob = gamma[0] / np.sum(gamma[0])
            self.transfer_prob = xi / np.sum(xi, axis=1, keepdims=True)
            # for k in range(self.n_state):
            #     self.transfer_prob[k] = xi[k] / np.sum(xi[k])

            self.update_emit_prob(X, gamma)
        self.trained = True

    def decode(self, X):
        """
        Infer hidden state sequence with given observation sequence X and trained model. Using Vertibi algorithm.
        :param X: observation sequence X
        :return: inference of hidden state sequence
        """
        if self.trained is False:
            raise ValueError("Model must be trained before inference process")

        n_seq = len(X)
        hidden_state = np.zeros(shape=n_seq, dtype=np.int32)

        # last hidden state node of the most possible path of current node
        pre_state = np.zeros(shape=(n_seq, self.n_state), dtype=np.int32)
        # probability of the most possible path of current node
        max_prob = np.zeros(shape=(n_seq, self.n_state), dtype=np.float32)

        _, c = self.forward(X, np.ones(shape=(n_seq, self.n_state)))

        # start probability
        max_prob[0] = self.emit_prob(X[0]) * self.start_prob / c[0]
        # forward process
        for i in range(1, n_seq):
            for k in range(self.n_state):
                state_prob = self.emit_prob(X[i])[k] * self.transfer_prob[:, k] * max_prob[i - 1]
                max_prob[i, k] = np.max(state_prob) / c[i]
                pre_state[i, k] = np.argmax(state_prob)

        # backward process
        hidden_state[-1] = int(np.argmax(max_prob[-1, :]))
        for i in range(n_seq - 2, -1, -1):
            hidden_state[i] = pre_state[i, hidden_state[i + 1]]
        return hidden_state

    @abstractmethod
    def update_emit_prob(self, X, gamma):
        pass

    def clear_model(self):
        """
        Delete parameters of this model. Must be called before re-train with new data.
        """
        self.start_prob = None
        self.transfer_prob = None


class DiscreteHMM(BaseHMM):
    """
    The random variable of observation is discrete

    Attribution:

    emit_prob_mat: condition probability of observation with each hidden state, array with shape (n_state, n_obs)
    """
    def __init__(self, n_obs, n_state, max_iters=100):
        super(DiscreteHMM, self).__init__(n_obs, n_state, max_iters)
        self.emit_prob_mat = None

    def initialize(self, X=None):
        super(DiscreteHMM, self).initialize(X)
        self.emit_prob_mat = np.random.random((self.n_state, self.n_obs))
        self.emit_prob_mat = self.emit_prob_mat / np.sum(self.emit_prob_mat, axis=1, keepdims=True)

    def emit_prob(self, x):
        """
        :param x: a scalar or 1-d array with shape (1,)
        :return: emit probability of each hidden state according to x, 1-d array with shape (n_state,)
        """
        return self.emit_prob_mat[:, int(x)]

    def update_emit_prob(self, X, gamma):
        self.emit_prob_mat = np.zeros((self.n_state, self.n_obs), dtype=np.float32)

        n_seq = len(X)
        for n in range(n_seq):
            self.emit_prob_mat[:, int(X[n])] += gamma[n]
        self.emit_prob_mat = self.emit_prob_mat / np.sum(self.emit_prob_mat, axis=1, keepdims=True)

    def clear_model(self):
        super(DiscreteHMM, self).clear_model()
        self.emit_prob_mat = None
