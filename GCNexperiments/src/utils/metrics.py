import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import average_precision_score

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

def eval_ap(y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return sum(ap_list)/len(ap_list)
