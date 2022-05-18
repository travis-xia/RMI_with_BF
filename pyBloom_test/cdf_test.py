from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import norm

x = np.arange(100)

def cdf(x):
    loc = x.mean()
    scale = x.std()
    n = x.size
    pos = norm.cdf(x, loc, scale) * n
    return pos

y = cdf(x)

print(y)




