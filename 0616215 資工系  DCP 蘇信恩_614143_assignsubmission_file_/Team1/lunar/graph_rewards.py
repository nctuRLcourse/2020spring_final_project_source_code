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
def plot(rewards, filename, param_text):
    plt.close()


    params = [1 + i*0.5 for i in np.arange(0, len(rewards))]
    assert(len(params) == len(rewards))
    average_reward = np.mean(rewards)
    plt.plot(params, rewards, color='b', linestyle='-', linewidth=2)
    plt.ylabel("Running reward")
    plt.xticks([p for i, p in enumerate(params) if i%2==0 ])
    plt.text(0.7, 270, "Average: {}".format(average_reward), ha='left', wrap=True)

    plt.ylim(-150, 300)
    plt.xlabel(param_text)
    plt.savefig("./graph/{}.png".format(filename))
if args.mm:
    RE = ".*_mm_.*pk$"
    FILENAME = "mm_distribution"
    PARAM_TEXT = "omega"
else:
    RE = "softmax_.*pk$"
    FILENAME = "softmax_distribution"
    PARAM_TEXT = "beta"

files = [f for  f in os.listdir("./") if re.search(RE, f)]
files.sort(key = lambda x: int(x.split('_')[3].split('.')[0]))

rewards = []
for file in files:
    pk_file = file
    # print(file)
    with open(pk_file, "rb") as f:
        r = pickle.load(f)
        rewards.append(r[-1])

plot(rewards, FILENAME, PARAM_TEXT)



