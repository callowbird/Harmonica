# A mini program for writing options.txt
# This program just outputs x1,x2,x3.... one for each line, N lines in total
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=100, help='number of hyperparameters?')
opt = parser.parse_args()

with open("options.txt","w") as f:
    for i in range(opt.N):
        if i>0:
            f.write("\n")
        f.write("x"+str(i)+",")
