import numpy as np


x = np.random.randn(10000)
print((x**2).var())

print(x.mean())
print(x.std())
print(x.var())
