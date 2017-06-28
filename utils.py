# This method computes the name (and IDs) of the features by recursion
def addNames(curName, curd, targetd, lasti, curID,options,names,ids,N):
    if curd<targetd:
        for i in range(lasti,N):
            addNames(curName + (' * ' if len(curName)>0 else '') + options[i], curd + 1, targetd, i + 1, curID + [i], options, names, ids,N)
    elif curd==targetd:
        names.append(curName)
        ids.append(curID)

# This method reads option names from options.txt, return as a list
# options.txt contains names for the options.
# See option_writer.py for how to generate a options.txt file
# Of course, you can write your own options.txt file. See options_example.txt for example.
def readOptions():
    with open("options.txt") as f:
        content=f.readlines()
    ans=[]
    for line in content:
        s=line.split(",")
        ans.append(s[0])
    return ans

def printSeparator():
    print("----------------------------------------------------------")
