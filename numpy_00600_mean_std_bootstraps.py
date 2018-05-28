import numpy as np
import matplotlib.pyplot as plt

price_dtype = [("timestamp", "f8"), ("price", "f8")]

def generate_random_arima_timeseries(size, factors):

    # autocorrelation
    random_values = np.random.randn(size)
    random_values = np.cumsum(random_values)
    b = np.zeros((size,2))
    b[:, 0] = np.arange(size, dtype=int)
    for i in range(len(factors),size):
        b[i,1] = np.dot(b[i-len(factors):i,1], factors) + random_values[i]# + np.sum(np.dot(random_values[i-len(factors):i], factors))
    return b

def get_bootstrapped_timeseries(timeseries, fraction, n=1):
    return [timeseries[np.where(np.random.sample(len(timeseries)) < fraction)] for _ in range(n)]


if __name__ == '__main__':
    factors = np.array([0.01, -0.01, 0.01, 0.02, 0.01, 0.01, 0.04, 0.02, -0.01, -0.02, -0.03, 0.01])
    TOTAL_SIZE = 1000
    timeseries = generate_random_arima_timeseries(TOTAL_SIZE, factors)
    n_bootstraps = 40
    b_ts = get_bootstrapped_timeseries(timeseries, 0.6, n_bootstraps)
    means = np.empty((TOTAL_SIZE, n_bootstraps + 1))
    means[:, 0] = np.arange(TOTAL_SIZE, dtype=int)
    means[:, 1:] *= np.nan
    for i, ts in enumerate(b_ts):
        #means[ts[:, 0].astype(int), i + 1] = ts[:, 1]
        means[:, i + 1] = np.interp(np.arange(TOTAL_SIZE, dtype=int), ts[:, 0].astype(int), ts[:, 1])

    bt_mean = np.empty((TOTAL_SIZE, 4))
    bt_mean[:, 0] = np.arange(TOTAL_SIZE, dtype=int)
    bt_mean[:, 1] = np.nanmean(means[:,1:], axis=1)
    # 1 sigma is the middle 68.27%
    # 2 sigma is the middle 95.45%
    # 3 sigma is the middle 99.73%
    bt_mean[:, 2] = np.percentile(means[:,1:], 2.3, axis=1)
    bt_mean[:, 3] = np.percentile(means[:, 1:], 100-2.3, axis=1)

    # f, axarr = plt.subplots(3, 1)
    # axarr[0].scatter(timeseries[:, 0], timeseries[:, 1], s=1, c="black")
    # axarr[0].set_title('time series')
    # axarr[1].scatter(bt_mean[:, 0], bt_mean[:, 1], s=1, c="blue")
    # axarr[1].set_title('bootstrapped time series')
    # axarr[2].scatter(bt_mean[:, 0], bt_mean[:, 1], s=1, c="blue")
    # axarr[2].scatter(bt_mean[:, 0], bt_mean[:, 2], s=1, c="red")
    # axarr[2].scatter(bt_mean[:, 0], bt_mean[:, 3], s=1, c="green")
    # axarr[2].set_title('bootstrapped time series')
    # plt.show()

    plt.plot(timeseries[:, 0], timeseries[:, 1], label="original")
    plt.plot(bt_mean[:, 0], bt_mean[:, 1], label="mean")
    plt.plot(bt_mean[:, 0], bt_mean[:, 2], label="lower quantile")
    plt.plot(bt_mean[:, 0], bt_mean[:, 3], label="upper quantile")
    plt.fill_between(bt_mean[:, 0], bt_mean[:, 2], bt_mean[:, 3], facecolor='yellow', alpha=0.2, label='2 sigma range')
    plt.title('bootstrapped time series')
    plt.legend()
    plt.show()
