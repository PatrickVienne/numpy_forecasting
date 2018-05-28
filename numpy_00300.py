import numpy as np
import matplotlib.pyplot as plt

price_dtype = [("timestamp", "f8"), ("price", "f8")]

random_values = np.random.randn(10000)

if __name__ == '__main__':
    # autocorrelation
    factors = np.array([0.1, -0.5, 0.5, 0.2, 0.1, 0.1, 0.4])
    b = np.zeros((1000,))
    for i in range(len(factors),1000):
        b[i] = np.dot(b[i-len(factors):i], factors) + random_values[i]

    plt.plot(b)
    plt.show()
