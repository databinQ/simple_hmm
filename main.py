# coding: utf-8

import numpy as np
from hmmlearn.hmm import MultinomialHMM

from hmm import DiscreteHMM


if __name__ == "__main__":
    start_probability = np.array([0.2, 0.4, 0.4])
    transition_probability = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])
    emission_probability = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])

    disc_hmm = MultinomialHMM(n_components=3)
    disc_hmm.startprob_= start_probability
    disc_hmm.transmat_= transition_probability
    disc_hmm.emissionprob_= emission_probability

    X, Z = disc_hmm.sample(100)

    my_model = DiscreteHMM(n_obs=2, n_state=3)
    my_model.train(X, Z)
    print(X)
