import time
import subprocess

trys = 2
command0 = ["python","./trg.py"]
command1 = ["julia","./trg.jl"]
command2 = ["./itrg/trg"]

def testrun(command,trys):
    during = 0
    for _ in range(trys):
        start = time.time()
        subprocess.check_call(command)
        end = time.time()
        during += (end - start)

    return during/trys

print("Testing pytorch")

average0 = testrun(command0,trys)

print("pytorch average:",average0,"of",trys,"runs")

print("Testing itensor")

average2 = testrun(command2,trys)

print("itensor average:",average2,"of",trys,"runs")

print("Testing julia")

average1 = testrun(command1,trys)

print("julia average:",average1,"of",trys,"runs")

print("Final result: pytorch:",average0,"itensor:",average2,"julia:",average1)



