import numpy as np
import matplotlib.pyplot as plt

price_dtype = [("timestamp", "f8"), ("price", "f8")]

def generate_random_arima_timeseries(size, factors):

    # autocorrelation
    random_values = np.random.randn(size)
    b = np.zeros((size,2))
    b[:, 0] = np.arange(size)
    for i in range(len(factors),size):
        b[i,1] = np.dot(b[i-len(factors):i,1], factors) + random_values[i]
    return b

def get_bootstrapped_timeseries(timeseries, fraction, n=1):
    return [timeseries[np.where(np.random.sample(100) < fraction)] for _ in range(n)]


if __name__ == '__main__':
    factors = np.array([0.1, -0.5, 0.5, 0.2, 0.1, 0.1, 0.4])
    TOTAL_SIZE = 1000
    timeseries = generate_random_arima_timeseries(TOTAL_SIZE, factors)
    b_ts = get_bootstrapped_timeseries(timeseries, 0.8)[0]

    f, axarr = plt.subplots(2, 1)
    axarr[0].scatter(timeseries[:,0], timeseries[:,1], s=1, c="black")
    axarr[0].set_title('time series')
    axarr[1].scatter(b_ts[:,0], b_ts[:,1], s=1, c="blue")
    axarr[1].set_title('bootstrapped time series')
    plt.show()

