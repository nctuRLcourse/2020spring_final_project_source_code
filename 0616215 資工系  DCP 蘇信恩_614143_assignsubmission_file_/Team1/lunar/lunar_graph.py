import matplotlib.pyplot as plt
import pickle
import matplotlib
import argparse
import os
parser = argparse.ArgumentParser(description='input filename')
parser.add_argument('-f','--filename', type=str, 
                    help='filename',default="default")
args = parser.parse_args()
def plot(filename):
    plt.close()
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # print(data)
    episode = []
    rewards = []
    for i, r in enumerate(data):
        episode.append(i)
        rewards.append(r)
        
    plt.plot(episode, rewards, color='b', linestyle='-', linewidth=2)
    plt.ylabel("Mean Reward")
    plt.ylim(-200, 200)
    plt.xlabel("episode")
    plt.savefig("{}.png".format(filename))

if args.filename == "default":
    for file in os.listdir("./"):
        if file.endswith(".pk"):
            print(os.path.join("./", file))
            plot(file)
else:
    plot(args.filename)