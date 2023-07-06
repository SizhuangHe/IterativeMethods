import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def MAD(feature, target_mask=None):
    '''
    See this AAAI paper:
        Measuring and Relieving the Over-Smoothing Problem for Graph Neural Networks from the Topological View
    I tried to keep variable names consistent with the paper.
    Here, I only want to look at two 
    '''
    D = 1 -  cosine_similarity(feature.detach())
    M_tgt = np.ones(D.shape)
    if target_mask is None:
        target_mask = np.ones(D.shape[0])
    for i in range(len(target_mask)):
        if target_mask[i] == 0:
            M_tgt[i] = 0 # row
            M_tgt[:,i] = np.zeros(len(target_mask))
    D_tgt = np.multiply(D, M_tgt)
    num_nonzero = np.count_nonzero(D_tgt)
    MAD = np.sum(D_tgt)/num_nonzero
    return MAD

