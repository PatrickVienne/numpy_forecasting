import numpy as np

price_dtype = [("timestamp", "f8"), ("price", "f8")]

random_values = np.random.randn(1000)

if __name__ == '__main__':
    # random walk
    a = np.zeros((1000,))
    for i in range(1,1000):
        a[i] = a[i-1] + random_values[i]
