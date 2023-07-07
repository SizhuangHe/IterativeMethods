import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def MAD(feature):
    '''
    See this AAAI paper:
        Measuring and Relieving the Over-Smoothing Problem for Graph Neural Networks from the Topological View
    I tried to keep variable names consistent with the paper.
    Only the global MAD is implemented here. That is, we average over all nodes in the graph.
    '''
    D = cosine_distances(feature.detach())
    mad = np.mean(D)
    return mad

