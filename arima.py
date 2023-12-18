import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


class arima_forecast:

    def __init__(self, ticker, start_date, end_date, train_pct):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_pct = train_pct

    def fetch_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        ts = data['Adj Close']
        return ts

    def arima(self):
        ts = self.fetch_data()
        train, test = ts[0:int(len(ts) * self.train_pct)], ts[int(len(ts) * self.train_pct):len(ts)]
        history = [x for x in train]
        predictions = list()

        for i in range(len(test)):
            model_fit = ARIMA(history, order=(1, 1, 1)).fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[i]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))

        predictions = np.array(predictions).reshape(len(predictions), -1)
        test = np.array(test).reshape(len(test), -1)

        return predictions, test

    def plot_arima(self):
        predictions, test = self.arima()
        plt.figure()
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.show()

    def rmse(self):
        predictions, test = self.arima()
        root_mse = np.sqrt(np.mean(np.power((test - predictions), 2)))
        return root_mse


if __name__ == '__main__':
    pred = arima_forecast('AAPL', '2020-01-01', '2022-01-01', 0.8)
    pred.plot_arima()
    # pred.rmse()
