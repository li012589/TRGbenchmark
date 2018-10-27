import time
import subprocess

command0 = ["python","./pytorch/trg.py"]
command1 = ["julia","./julia/trg.jl"]
command2 = ["./itensor/trg"]

iterPoints = [10,20,30,40,50,60]
cutPoints = [10,20,30,40,50]
trys = 3

def testrun(command,trys):
    during = 0
    for _ in range(trys):
        start = time.time()
        subprocess.check_call(command)
        end = time.time()
        during += (end - start)

    return during/trys

def testdraw(command,trys,points,subfix = [],prefix = []):
    times = []
    for ptr in points:
        times.append(testrun(command+prefix+[str(ptr)]+subfix,trys))
    return times

print("Testing pytorch scaling with iterations")

times00 = testdraw(command0,trys,iterPoints,prefix = ["-iters"])

print("Testing pytorch scaling with maximum cut")

times01 = testdraw(command0,trys,cutPoints,prefix = ["-cut"])

print("Testing itensor scaling with iterations")

times20 = testdraw(command2,trys,iterPoints,prefix = ["1.0"],subfix=["20"])

print("Testing itensor scaling with maximum cut")

times21 = testdraw(command2,trys,cutPoints,prefix = ["1.0","20"])

print("Testing julia scaling with iterations")

times10 = testdraw(command1,trys,iterPoints,prefix = ["1.0"],subfix=["20"])

print("Testing julia scaling with maximum cut")

times11 = testdraw(command1,trys,cutPoints,prefix = ["1.0","20"])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(iterPoints,times00,label="pytorch")
plt.plot(iterPoints,times20,label="itensor")
plt.plot(iterPoints,times10,label="julia")

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Times(s)")
plt.title("Time Scaling with Iterations")

plt.figure()
plt.plot(cutPoints,times01,label="pytorch")
plt.plot(cutPoints,times21,label="itensor")
plt.plot(cutPoints,times11,label="julia")

plt.legend()
plt.xlabel("Maximum Cut")
plt.ylabel("Times(s)")
plt.title("Time Scaling with Maximum Cut")

plt.show()

import pdb
pdb.set_trace()

