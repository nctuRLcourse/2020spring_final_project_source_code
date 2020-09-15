import matplotlib.pyplot as plt
import pickle
import matplotlib
import argparse
import os
import re
import numpy as np
parser = argparse.ArgumentParser(description='input filename')
parser.add_argument('-mm','--mm', action='store_true', 
                    help='mm?')
args = parser.parse_args()
def plot(times, filename, param_text):
    # plt.close()


    params = [1 + i*0.5 for i in np.arange(0, len(times))]
    assert(len(params) == len(times))
    average_time = np.mean(times)
    plt.plot(params, times, color='b', linestyle='-', linewidth=2)
    plt.ylabel("Running Time")
    plt.ylim(0, 600)
    plt.xticks([p for i, p in enumerate(params) if i%2==0 ])
    plt.text(0.7, plt.ylim()[0]+5, "Average: {}".format(average_time), ha='left', wrap=True)
    # plt.ylim(-150, 300)
    plt.xlabel(param_text)
    plt.savefig("./graph/{}.png".format(filename))

if args.mm:
    RE = "mm_.*result$"
    FILENAME = "mm_time"
    PARAM_TEXT = "omega"
else:
    RE = "softmax_.*result$"
    FILENAME = "softmax_time"
    PARAM_TEXT = "beta"

files = [f for  f in os.listdir("./") if re.search(RE, f)]
# print(files)
files.sort(key = lambda x: int(x.split('_')[1].split('.')[0]))

times = []
for file in files:
    result_file = file
    # print(file)
    with open(result_file, "r") as f:
        for line in reversed(f.readlines()):
            if line.startswith("real"):
                times.append(int(line.strip().split("\t")[1].split('m')[0]))
                
# print(times)


plot(times, FILENAME, PARAM_TEXT)



