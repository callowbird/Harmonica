import argparse
import numpy as np
from sklearn import linear_model
from samplings import batch_sampling
from sklearn.preprocessing import PolynomialFeatures
from utils import addNames,readOptions,printSeparator
from base_alg import base_hyperband,base_random_search


parser = argparse.ArgumentParser()
parser.add_argument('-alpha', type=float, default=3, help='weight of the l_1 regularization in Lasso')
parser.add_argument('-nSample', type=int, default=300, help='number of samples per stage?')
parser.add_argument('-nStage', type=int, default=3, help='number of stages?')
parser.add_argument('-degree', type=int, default=3, help='degree of the features? If -1, will tune it automatically.')
parser.add_argument('-nMono', type=int, default=5, help='number of monomials selected in Lasso?')
parser.add_argument('-N', type=int, default=60, help='number of hyperparameters?')
parser.add_argument('-t', type=int, default=1, help='number of masks picked in each stage?')
opt = parser.parse_args()

learnedFeature=[]                   # This is a list of important features that we extracted
bestAnswer=100000                   # Best answer so far
maskList=[]                         # List of masks of the value for the fixed important variables

optionNames=readOptions()           # Read the names of options
print("Total options : ",len(optionNames))
assert(len(optionNames)==opt.N)     # Please be consistent on the number of options names and the number of options.
extendedNames=[]                    # This list stores the names of the features, including vanilla features and low degree features
featureIDList=[]                    # This list stores the variable IDs of each feature

LassoSolver = linear_model.Lasso(fit_intercept=True, alpha=opt.alpha)     # Lasso solver

def getBasisValue(input,basis,featureIDList): # Return the value of a basis
    ans=0
    for (weight,pos) in basis:                  # for every term in the basis
        term=weight
        for entry in featureIDList[pos]:        # for every variable in the term
            term=term*input[entry]
        ans+=term
    return ans

selected_degree=opt.degree

for currentStage in range(opt.nStage):                      # Multi-stage Lasso
    printSeparator()
    print("Stage",currentStage)
    print("Sampling..")
    x,y=batch_sampling(maskList, opt.nSample, opt.N)        # Get a few samples at this stage
    bestAnswer=min(np.min(y), bestAnswer)                   # Update the best answer

    for i in range(0,len(y)):
        for basis in learnedFeature:
            y[i]-=getBasisValue(x[i],basis,featureIDList)     # Remove the linear component previously learned

    def get_features(x,degree):
        print("Extending feature vectors with degree "+str(degree)+" ..")
        featureExtender = PolynomialFeatures(degree, interaction_only=True)   # This function generates low degree monomials. It's a little bit slow though. You may write your own function using recursion.
        tmp=[]
        for current_x in x:
            tmp.append(featureExtender.fit_transform(np.array(current_x).reshape(1, -1))[0].tolist())   # Extend feature vectors with monomials
        return tmp

    if selected_degree<0:  #tune automatically
        last_ind=-1
        print("Searching for the best degree parameter..")
        selected_degree=2
        while True:
            tmp=get_features(x,selected_degree)
            tmp=np.array(tmp)                         # Make it array
            LassoSolver.fit(tmp, y)                       # Run Lasso to detect important features
            coef=LassoSolver.coef_
            index=np.argsort(-np.abs(coef))             # Sort the coefficient, find the top ones
            index=index[:opt.nMono]                      # select the top indices
            print("get the following indicse",index)
            find_useful=False
            for i in index:
                if i>last_ind:
                    find_useful=True
                    break
            if find_useful:
                last_ind=len(coef)
                print("new len",last_ind)
                selected_degree+=1
            else:
                print("done with degree ",selected_degree)
                selected_degree-=1
                break
    if (currentStage==0):
        for depth in range(0, selected_degree+1):
            addNames('', 0, depth, 0, [], optionNames, extendedNames, featureIDList,opt.N)    # This method computes extendedNames and featureIDList
        print("Number of features : ",len(extendedNames))

    x=np.array(get_features(x,selected_degree))                     # Make it array

    print("Running linear regression..")
    LassoSolver.fit(x, y)                       # Run Lasso to detect important features
    coef=LassoSolver.coef_
    index=np.argsort(-np.abs(coef))             # Sort the coefficient, find the top ones
    cur_basis=[]
    print("Found the following sparse low degree polynomial:\n f = ",end="")
    for i in range(0,opt.nMono):
        if i>0 and coef[index[i]]>=0:
            print(" +",end="")
        print("%.2f %s"%(coef[index[i]],extendedNames[index[i]]),end="")       # Print the top coefficient value, the corresponding feature IDs
        cur_basis.append((coef[index[i]],index[i]))          # Add the basis, and its position, only add top nMono
    print("")
    learnedFeature.append(cur_basis[:])                      # Save these features in learned features

    mapped_count = np.zeros(opt.nSample)                     # Initialize the count matrix

    for cur_monomial in cur_basis:                          # For every important feature (a monomial) that we learned
        for cur_index in featureIDList[cur_monomial[1]]:                   # For every variable in the monomial
            mapped_count[cur_index]+=1                      # We update its count

    config_enumerate = np.zeros(opt.nSample)                # Use this array to enumerate all possible configuration (to find the minimum for the current sparse polynomial)
    l=[]                                                    # All relevant variables.
    for i in range(0, opt.nSample):                         # We only need to enumerate the value for those relevant variables
        if mapped_count[i] > 0:                             # This part can be made slightly faster. If count=1, we can always set the best value for this variable.
            l.append(i)

    lists=[]
    for i in range(0, 2**len(l)):  # for every possible configuration
        for j in range(0, len(l)):
            config_enumerate[l[j]]=1 if ((i % (2 ** (j + 1))) // (2 ** j) == 0) else -1
        score=0
        for cur_monomial in cur_basis:
            base=cur_monomial[0]
            for cur_index in featureIDList[cur_monomial[1]]:
                base= base * config_enumerate[cur_index]
            score=score+base
        lists.append((config_enumerate.copy(), score))
    lists.sort(key=lambda x : x[1])
    maskList.append(lists[:opt.t])
y=base_hyperband(maskList, 40000,opt.N)               # Run a base algorithm for fine tuning
#y=base_random_search(maskList,100,N)                  # Run random search as the base algorithm
bestAnswer=min(y, bestAnswer)
printSeparator()
print('best answer : ', bestAnswer)

