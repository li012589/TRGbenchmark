import time
import subprocess
import numpy as np

trys = 10
command0 = ["python","./pytorch/trg.py"]
command1 = ["julia","./julia/trg.jl"]
command2 = ["./itensor/trg"]

def testrun(command,trys):
    during = []
    for _ in range(trys):
        start = time.time()
        subprocess.check_call(command)
        end = time.time()
        during.append((end - start))
    during = np.array(during)
    return during.mean(),during.std()

print("Testing pytorch")

average0,std0 = testrun(command0,trys)

print("pytorch average:",average0,"+/-",std0,"of",trys,"runs")

print("Testing itensor")

average2,std2 = testrun(command2,trys)

print("itensor average:",average2,"+/-",std2,"of",trys,"runs")

print("Testing julia")

average1,std1 = testrun(command1,trys)

print("julia average:",average1,"+/-",std1,"of",trys,"runs")

print("Final result: pytorch:",average0,"+/-",std0,"itensor:",average2,"+/-",std2,"julia:",average1,"+/-",std1)



