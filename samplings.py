# You need to implement your own query method
# See the batch_sampling method at the bottom. Currently it's running in sequential. Please modify it accordingly if you need parallel implementation (which is straightforward).

import numpy as np

# Example 1: a simple sparse linear function, please uncomment the code below

##########################################################################
def query(x):
    assert(len(x)==60)
    return 10*(x[2]*x[8]+x[15]+x[46]+x[38]+x[3]*x[6])  #return a simple sparse linear function
##########################################################################





# Example 2: a multi-stage sparse function, please uncomment the code below

##########################################################################
# class sparse_fun():
#     def __init__(self,N,nFeature,nStage,randomness,weight,degree):  #only consider sparse functions..
#         manualSeed=np.random.randint(10000)
#         np.random.seed(0)
#         self.N=N
#         self.stageList=[]
#         self.nFeature=nFeature
#         self.cases=2**self.nFeature
#         self.nStage=nStage
#         self.randomness=randomness
#         self.weight=weight
#         self.degree=degree
#         self.init_ids()
#         self.base_len=len(self.ids)
#         for i in range(nStage):  #now generate ...
#             curList=[]
#             for j in range(self.cases**i): #total number of cases.
#                 basis=[]
#                 for k in range(nFeature):
#                     basis.append(
#                         ((np.random.random()*self.weight+100)*((-1)**(np.random.randint(2)))/(4**i),
#                          np.random.randint(self.base_len))
#                     )
#                 curList.append(basis)
#             self.stageList.append(curList)
#         np.random.seed(manualSeed)
#     def init_ids(self):  #generate ids.
#         self.ids=[]
#         def Namedepth(curd,targetd,lasti,curID):
#                 if curd<targetd:
#                     for i in range(lasti,self.N):
#                         Namedepth(curd+1, targetd,i+1, curID+[i])
#                 elif curd==targetd:
#                     self.ids.append(curID)
#         for dep in range(1,self.degree+1):
#             Namedepth(0, dep, 0,[])
#     def getTermVal(self,input,weight,pos):
#         ans=1
#         term=self.ids[pos]
#         for entry in term:
#             ans=ans*input[entry]  #multiply +1 or -1
#         return ans*weight
#     def getBasisValue(self,input,basis): #return the value
#         ans=0
#         for (weight,pos) in basis:
#             ans+=self.getTermVal(input,weight,pos)
#         return ans
#     def getCubeID(self, input, basis):
#         ans=0
#         for (weight,pos) in basis:
#             if self.getTermVal(input,weight,pos)>0:
#                 ans=ans*2+1
#             else:
#                 ans=ans*2
#         return ans
#     def query(self,input):  #ask for the input
#         curID=0
#         ans=0
#         for i in range(self.nStage):
#             print("C%i\t"%(curID),end="")          # Print cube ID
#             cur_basis=self.stageList[i][curID]      # Get basis
#             cur_value=self.getBasisValue(input,cur_basis)
#             ans+=cur_value
#             curID=curID*self.cases+self.getCubeID(input, cur_basis)
#         return ans+self.randomness*np.random.random()
#
# N = 60
# nFeature = 5
# nStage = 3
# randomness = 4
# weight = 100
# degree = 3
# func = sparse_fun(N, nFeature, nStage, randomness, weight, degree)  # define a multi-stage sparse function
#
# def query(x):
#     assert(len(x)==N)
#     return func.query(x)
##########################################################################

def mask_random_sample(mask_list,N):                  # mask_list is a list of lists of masks.
    current_x=np.zeros(N)                              # Initialized with zeros
    for masks in mask_list:                            # need to enumerate all mask levels
        mask_picked=np.random.randint(len(masks))      # need to pick one mask from a number of masks
        mask=masks[mask_picked][0]                        # picked mask
        for j in range(N):
            if (mask[j]!=0) and (current_x[j]==0):         # mask has value, not written before
                current_x[j]=mask[j]                   # fill in mask value!
    for j in range(N):                                 # If not filled by masks, fill it with random value
        if current_x[j]==0:
            current_x[j]=(-1)**np.random.randint(2)
    return current_x

# This method is a trivial method. It adds some artificial noise to the result.
# If you need to implement this method for your own application, you should implement your own training algorithm
def batch_intermediate_sampling(x,start_epoch,end_epoch):
    y=[]
    for i in x:
        y.append(query(i)+np.random.random()*(300-end_epoch)/300)
    return y

# Sequential implementation of batch_sampling.
# Please modify it if you want parallel implementation, using your favorite tool.
def batch_sampling(mask_list, n_samples, N):
    x=[]
    for i in range(n_samples):
        x.append(mask_random_sample(mask_list,N))
    y=[]
    for i in range(n_samples):
        y.append(query(x[i]))
    return x,y

