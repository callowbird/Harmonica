# After running main.py for multiple stages (and fix some important variables), we will run base algorithm for fine tuning.
# Here, we only implemented random search and hyperband as the base algorithm.
# You may implement your favorite hyperparameter algorithm instead. Just use the mask files in a proper way.

import samplings
import numpy as np
from utils import printSeparator

# Random search algorithm. Just sample a few random points
def base_random_search(mask_list,n_samples,N):   # mask_list defines a subspace for searching. N is the dimension.
    printSeparator()
    printSeparator()
    print("Running random search as the base algorithm.")
    x,y=samplings.batch_sampling(mask_list, n_samples, N)       # Just call the batch sampling method
    return np.min(y)                                      # return the y vector

# Hyperband algorithm. You need to implement your own intermediate sampling algorithm.
# Here we only implement a sketch of hyperband algorithm
def base_hyperband(mask_list,budget, N):       #use hyperband to search the space
    printSeparator()
    printSeparator()
    print("Running hyperband as the base algorithm.")
    max_iter=100                                # Total iterations
    eta=3.0                                     # hyperparameter eta
    def logeta(x,eta):
        return np.log(x)/np.log(eta)
    s_max=int(np.floor(logeta(max_iter,eta)))   # range for hyperparameter s
    B=budget//(s_max+1)                         # budget for each s
    print("B=",B)
    print("Smax=",s_max)
    s_min=0                                     # minimum s=0, this case is just random search
    best_ans=100000                             # best answer

    print('WARNING! Please implement the intermediate sampling algorithm.')
    print('The current intermediate sampling algorithm is trivial. It cannot be applied to your application.')

    for s in range(s_min,s_max+1):              # For every s, do the following
        printSeparator()
        print("s=",s)
        n=int(np.floor(B/max_iter/(s+1)*np.power(eta,s)))   # number of initial random configurations
        print("n=",n)
        x=[]
        for i in range(n):
            x.append(samplings.mask_random_sample(mask_list,N))         # Get some random initial configurations
        remaining=n                                         # The number of remaining configurations
        endEpoch=int(max_iter*np.power(eta,-s))             # The first time we start to remove a few configurations

        lastEpoch=0                                         # The last time we remove a few configurations
        for i in range(s+1):                                # for s steps
            print("Remaining..",remaining)
            print("r=",endEpoch)

            ###########################
            # Please implement the intermediate sampling algorithm for y, based on x
            # The current implementation is a trivial one. CANNOT BE APPLIED TO YOUR APPLICATION!
            y=samplings.batch_intermediate_sampling(x[:remaining],lastEpoch,endEpoch)
            ###########################

            best_ans=min(np.min(y), best_ans)               # Update best answer
            sorted_ind=np.argsort(y)                        # Sort y
            remaining=int(np.ceil(remaining/eta))           # remove a few configurations.
            lastEpoch=endEpoch                              # Update the last epoch
            endEpoch=np.ceil(endEpoch*eta)                  # Update the end epoch
            if endEpoch>max_iter:
                endEpoch=max_iter
            tmpx=x[:]
            for j in range(0,remaining):                    # Only keep the best configurations
                tmpx[j]=x[sorted_ind[j]]
            x=tmpx[:]

    return best_ans

