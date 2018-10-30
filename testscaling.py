import time
import subprocess
import numpy as np

command0 = ["python","./pytorch/trg.py"]
command1 = ["julia","./julia/trg.jl"]
command2 = ["./itensor/trg"]

iterPoints = [10,20,30,40,50,60]
cutPoints = [10,20,30,40,50]
trys = 2

def testrun(command,trys):
    during = []
    for _ in range(trys):
        start = time.time()
        subprocess.check_call(command)
        end = time.time()
        during.append((end - start))
    during = np.array(during)
    return during.mean(),during.std()


def testdraw(command,trys,points,subfix = [],prefix = []):
    times = []
    stds = []
    for ptr in points:
        time,std = testrun(command+prefix+[str(ptr)]+subfix,trys)
        times.append(time)
        stds.append(std)
    return np.array(times),np.array(stds)

print("Testing pytorch scaling with iterations")

times00,std00 = testdraw(command0,trys,iterPoints,prefix = ["-iters"])

print("Testing pytorch scaling with maximum cut")

times01,std01 = testdraw(command0,trys,cutPoints,prefix = ["-cut"])

print("Testing itensor scaling with iterations")

times20,std20 = testdraw(command2,trys,iterPoints,prefix = ["1.0"],subfix=["20"])

print("Testing itensor scaling with maximum cut")

times21,std21 = testdraw(command2,trys,cutPoints,prefix = ["1.0","20"])

print("Testing julia scaling with iterations")

times10,std10 = testdraw(command1,trys,iterPoints,prefix = ["1.0"],subfix=["20"])

print("Testing julia scaling with maximum cut")

times11,std11 = testdraw(command1,trys,cutPoints,prefix = ["1.0","20"])

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.errorbar(iterPoints,times00,yerr=std00,label="pytorch")
ax.errorbar(iterPoints,times20,yerr=std20,label="itensor")
ax.errorbar(iterPoints,times10,yerr=std10,label="julia")

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Times(s)")
plt.title("Time Scaling with Iterations")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.errorbar(cutPoints,times01,yerr=std01,label="pytorch")
ax.errorbar(cutPoints,times21,yerr=std21,label="itensor")
ax.errorbar(cutPoints,times11,yerr=std11,label="julia")
ax.set_yscale("log")

plt.legend()
plt.xlabel("Maximum Cut")
plt.ylabel("Times(s)")
plt.title("Time Scaling with Maximum Cut")

plt.show()

import pdb
pdb.set_trace()

