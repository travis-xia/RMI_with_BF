import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


def cdf(x):
    if type(x) == np.ndarray:
        loc = x.mean()
        scale = x.std()
        n = x.size
        pos = norm.cdf(x, loc, scale) * n
        return pos
    else:
        print("Wrong Type!~")
        exit(-1)

plt.style.use('bmh')
print(plt.style.available)


data = np.fromfile('../data/books_200M_uint32.txt', dtype=np.int32)
data.sort()
x = np.arange(len(data))
plt.plot(data,cdf(data))

plt.xlabel("key")
plt.ylabel("position")

plt.show()




