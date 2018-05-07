import numpy as np

def estimate_gaussian(X):
    
    """
    Estimate the mean and standard deviation of a numpy matrix X on a column by column basis
    """
    mu = np.zeros((X.shape[1],))
    var = np.zeros((X.shape[1],))
    ####################################################################
    #               YOUR CODE HERE                                     #
    ####################################################################
    mu = np.average(X, axis = 0)
    var = np.var(X, axis = 0, ddof = 0)
    ####################################################################
    #               END YOUR CODE                                      #
    ####################################################################
    return mu, var


def select_threshold(yval,pval):
    """
    select_threshold(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    """

    best_epsilon = 0
    bestF1 = 0
    stepsize = (max(pval)-min(pval))/1000
    for epsilon in np.arange(min(pval)+stepsize, max(pval), stepsize):
        
        ####################################################################
        #                 YOUR CODE HERE                                   #
        ####################################################################

        tp = np.dot((pval < epsilon) * 1, (yval == 1) * 1)[0]
        fp = np.dot((pval < epsilon) * 1, (yval == 0) * 1)[0]
        fn = np.dot((pval >= epsilon) * 1, (yval == 1) * 1)[0]
        prec = 1.0 * tp / (tp + fp)
        rec = 1.0 * tp / (fp + fn)
        f1 = 2.0 * prec * rec / (prec + rec)
        if(f1 > bestF1):
            bestF1 = f1
            best_epsilon = epsilon

        ####################################################################
        #                 END YOUR CODE                                    #
        ####################################################################
    return best_epsilon, bestF1
