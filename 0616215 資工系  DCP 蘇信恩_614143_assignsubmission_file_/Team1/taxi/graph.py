import matplotlib.pyplot as plt
import pickle
import matplotlib

filename = "mm"

with open(filename, "rb") as f:
    data = pickle.load(f)
# print(data)
params = []
rewards = []
for e in data:
    p, r = e
    params.append(p)
    rewards.append(r)
plt.title("Mellowmax Softmax")
plt.ylim(0, 20)
plt.plot(params, rewards, color='b', linestyle='-', linewidth=2)
plt.ylabel("Mean Reward")
plt.xlabel("omega")
plt.savefig("{}.png".format(filename))
