import pandas as pd
import numpy as np
import json
import os
import random
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
import xgboost as xgb
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings


class LSTM_ARIMA_XGBoost:
    '''
    This is a class that uses LSTM neural network, ARIMA model and XGBoost method
    to fit stock prices
    '''

    def __init__(self, ticker, start_date, end_date, train_pct):
        '''
        Initialize the parameters, namely the stock symbol, the analyse period
        and the percentage of data used to train the model
        '''
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.train_pct = train_pct

    def fetch_data(self):
        '''
        Retrieve the data from yahoo finance
        '''
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        data.drop(columns='Close', axis=1, inplace=True)
        data.to_csv('data.csv')
        return data

    def train_test_split_lstm(self):
        '''
        Split the data into train and test set for LSTM neural network
        '''
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            df = df.set_index(['Date'], drop=True)
        else:
            df = self.fetch_data()

        # rolling window, i.e., history time period used for prediction
        time_stamp = 60

        # split into train and test set
        train = df[:int(len(df) * self.train_pct) + time_stamp]
        test = df[int(len(df) * self.train_pct) - time_stamp:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_sc = scaler.fit_transform(train)
        test_sc = scaler.fit_transform(test)

        # train set
        x_train, y_train = [], []

        for i in range(time_stamp, len(train)):
            x_train.append(train_sc[i - time_stamp:i])
            y_train.append(train_sc[i, 3])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # test set
        x_test, y_test = [], []
        for i in range(time_stamp, len(test)):
            x_test.append(test_sc[i - time_stamp:i])
            y_test.append(test_sc[i, 3])

        x_test, y_test = np.array(x_test), np.array(y_test)

        return x_train, y_train, x_test, y_test

    def lstm(self):
        '''
        LSTM training and write result to files
        '''
        random.seed(100)
        x_train, y_train, x_test, y_test = self.train_test_split_lstm()

        # LSTM parameters: return_sequences=True, LSTM returns a sequence; default=False, returns a value
        # input_dim：dimension of sample characteristic
        # input_length：input time length
        model = Sequential()
        model.add(LSTM(units=15, return_sequences=True))
        model.add(Dropout(0.02))
        model.add(LSTM(units=15, return_sequences=False))
        model.add(Dropout(0.02))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=1, callbacks=[early_stop])

        y_pred = model.predict(x_test)
        fitted_values = model.predict(x_train)

        # save
        with open('y_pred.jsonl', 'w', encoding='utf8') as file:
            y_pred_list = y_pred.tolist()

            for batch in y_pred_list:
                file.write(json.dumps(batch) + '\n')

        with open('y_test.jsonl', 'w', encoding='utf8') as file:
            y_test_list = y_test.tolist()

            for batch in y_test_list:
                file.write(json.dumps(batch) + '\n')

        with open('fitted_values.jsonl', 'w', encoding='utf8') as file:
            fitted_values_list = fitted_values.tolist()

            for batch in fitted_values_list:
                file.write(json.dumps(batch) + '\n')

        with open('y_train.jsonl', 'w', encoding='utf8') as file:
            y_train_list = y_train.tolist()

            for batch in y_train_list:
                file.write(json.dumps(batch) + '\n')

        return y_pred, y_test, fitted_values, y_train

    def lstm_res(self):
        '''
        Inverse transform to get the original values
        '''
        # judge if the files exist
        # if exist, read directly; if not, retrain LSTM
        if os.path.exists('y_pred.jsonl') and os.path.exists('y_test.jsonl') and os.path.exists(
                'fitted_values.jsonl') and os.path.exists('y_train.jsonl'):

            # read
            with open('y_pred.jsonl', 'r', encoding='utf8') as file:
                y_pred_list = []

                for line in file:
                    y_pred_list.append(json.loads(line.strip()))

                y_pred = np.array(y_pred_list)

            with open('y_test.jsonl', 'r', encoding='utf8') as file:
                y_test_list = []

                for line in file:
                    y_test_list.append(json.loads(line.strip()))

                y_test = np.array(y_test_list)

            with open('fitted_values.jsonl', 'r', encoding='utf8') as file:
                fitted_values_list = []

                for line in file:
                    fitted_values_list.append(json.loads(line.strip()))

                fitted_values = np.array(fitted_values_list)

            with open('y_train.jsonl', 'r', encoding='utf8') as file:
                y_train_list = []

                for line in file:
                    y_train_list.append(json.loads(line.strip()))

                y_train = np.array(y_train_list)

        else:
            y_pred, y_test, fitted_values, y_train = self.lstm()

        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            df = df.set_index(['Date'], drop=True)
        else:
            df = self.fetch_data()

        time_stamp = 60
        test = df[int(len(df) * self.train_pct) - time_stamp:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(pd.DataFrame(test['Adj Close'].values))

        # inverse transform
        y_pred_origin = scaler.inverse_transform(y_pred)
        y_test_origin = scaler.inverse_transform(y_test.reshape(len(y_test), 1))

        fitted_origin = scaler.inverse_transform(fitted_values)
        y_train_origin = scaler.inverse_transform(y_train.reshape(len(y_train), 1))

        return y_pred_origin, y_test_origin, fitted_origin, y_train_origin

    def plot_lstm(self):
        '''
        Plot the LSTM prediction vs real prices
        '''
        y_pred_origin, y_test_origin, fitted_origin, y_train_origin = self.lstm_res()
        plt.figure()
        plt.plot(y_pred_origin, label='lstm', color='red')
        plt.plot(y_test_origin, label='real')
        plt.legend()
        plt.show()

    def lstm_residual(self):
        '''
        Retrieves the residual of LSTM estimation
        '''
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            df = df.set_index(['Date'], drop=True)
        else:
            df = self.fetch_data()

        y_pred_origin, y_test_origin, fitted_origin, y_train_origin = self.lstm_res()

        test_resid = y_pred_origin - y_test_origin
        train_resid = fitted_origin - y_train_origin
        resid = np.vstack((train_resid, test_resid))
        ts = pd.DataFrame(resid)
        ts.index = df.index
        ts.columns = ['residual']
        ts = ts['residual']
        return ts

    def plot_residual(self):
        '''
        ACF and PACF plot of the LSTM residual
        '''
        ts = self.lstm_residual()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(ts, lags=20, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(ts, lags=20, ax=ax2)
        plt.show()

    def residual_adf(self):
        '''
        ADF stationary test for the LSTM residual
        '''
        ts = self.lstm_residual()
        temp = np.array(ts)
        t = adfuller(temp, regression='ct', autolag='BIC')
        output = pd.DataFrame(
            index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
                   "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
        output['value']['Test Statistic Value'] = t[0]
        output['value']['p-value'] = t[1]
        output['value']['Lags Used'] = t[2]
        output['value']['Number of Observations Used'] = t[3]
        output['value']['Critical Value(1%)'] = t[4]['1%']
        output['value']['Critical Value(5%)'] = t[4]['5%']
        output['value']['Critical Value(10%)'] = t[4]['10%']
        return output

    def order_selection(self):
        '''
        Specify ARIMA model orders
        '''
        ts = self.lstm_residual()
        pmax = 5  # int(len(ts) / 10)    #normally no more than length /10
        qmax = 5  # int(len(ts) / 10)
        bic_matrix = []
        for p in range(pmax + 1):
            temp = []
            for q in range(qmax + 1):
                try:
                    temp.append(ARIMA(ts, order=(p, 0, q)).fit().bic)
                except:
                    temp.append(None)
            bic_matrix.append(temp)

        bic_matrix = pd.DataFrame(bic_matrix)
        # Use stack to flatten, then use idxmin to find the position of minimum value
        p, q = bic_matrix.stack().idxmin()
        # print(u'The p and q that minimize BIC：%s,%s' %(p, q))
        order = [p, q]

        with open('order.jsonl', 'w', encoding='utf8') as file:
            for batch in order:
                file.write(json.dumps(batch) + '\n')

        return p, q

    def arima(self):
        '''
        Forecast LSTM residual based on ARIMA model.
        '''
        ts = self.lstm_residual()
        if os.path.exists('order.jsonl'):
            with open('order.jsonl', 'r', encoding='utf8') as file:
                order = []
                for line in file:
                    order.append(json.loads(line.strip()))
                p, q = order[0], order[1]
        else:
            p, q = self.order_selection()

        train, test = ts[0:int(len(ts) * self.train_pct)], ts[int(len(ts) * self.train_pct):len(ts)]
        history = [x for x in train]
        predictions = list()
        for i in range(len(test)):
            model_fit = ARIMA(history, order=(p, 0, q)).fit()
            output = model_fit.forecast()
            yhat = np.array(output)
            predictions.append(yhat[0])
            obs = test[i]
            history.append(obs)
            # print('predicted=%f, expected=%f' % (yhat, obs))

        # save file
        with open('predictions.jsonl', 'w', encoding='utf8') as file:
            predictions_list = predictions

            for batch in predictions_list:
                file.write(json.dumps(batch) + '\n')

        with open('test.jsonl', 'w', encoding='utf8') as file:
            test_list = test.tolist()

            for batch in test_list:
                file.write(json.dumps(batch) + '\n')

        return predictions, test

    def plot_arima_forecast(self):
        '''
        Plot ARIMA forecast values against original LSTM residual
        '''
        if os.path.exists('predictions.jsonl') and os.path.exists('test.jsonl'):

            # read file
            with open('predictions.jsonl', 'r', encoding='utf8') as file:
                predictions_list = []

                for line in file:
                    predictions_list.append(json.loads(line.strip()))

                predictions = np.array(predictions_list)

            with open('test.jsonl', 'r', encoding='utf8') as file:
                test_list = []

                for line in file:
                    test_list.append(json.loads(line.strip()))

                test = np.array(test_list)

        else:
            predictions, test = self.arima()
            predictions = np.array(predictions)
            test = np.array(test)

        plt.figure()
        plt.plot(test, label='residual')
        plt.plot(predictions, label='arima', color='red')
        plt.legend()
        plt.show()

    def lstm_arima(self):
        '''
        Generate adjusted prediction after adjusting LSTM prediction to ARIMA-based residual
        '''
        y_pred_origin, y_test_origin, fitted_origin, y_train_origin = self.lstm_res()

        if os.path.exists('predictions.jsonl') and os.path.exists('test.jsonl'):

            # read file
            with open('predictions.jsonl', 'r', encoding='utf8') as file:
                predictions_list = []

                for line in file:
                    predictions_list.append(json.loads(line.strip()))

                predictions = np.array(predictions_list)

            with open('test.jsonl', 'r', encoding='utf8') as file:
                test_list = []

                for line in file:
                    test_list.append(json.loads(line.strip()))

                test = np.array(test_list)

        else:
            predictions, test = self.arima()
            predictions = np.array(predictions)
            test = np.array(test)

        predictions = predictions.reshape(len(predictions), -1)
        justified = y_pred_origin - predictions
        return justified, y_test_origin

    def plot_lstm_arima(self):
        '''
        Plot LSTM-ARIMA model vs real prices
        '''
        justified, y_test_origin = self.lstm_arima()
        plt.figure()
        plt.plot(justified, label='justifued', color='pink')
        plt.plot(y_test_origin, label='real', color='skyblue')
        plt.legend()
        plt.show()

    def train_test_split_xgb(self):
        '''
        Split the data into train and test set for XGBoost
        '''

        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            df = df.set_index(['Date'], drop=True)
            df = pd.DataFrame(df["Adj Close"])
        else:
            df = self.fetch_data()
            df = pd.DataFrame(df["Adj Close"])

        # length of time period
        time_stamp = 60

        # Split the data into train and test set
        train = df[:int(len(df) * self.train_pct) + time_stamp]
        test = df[int(len(df) * self.train_pct) - time_stamp:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_sc = scaler.fit_transform(train)
        test_sc = scaler.fit_transform(test)

        # train set
        x_train_xgb, y_train_xgb = [], []

        for i in range(time_stamp, len(train)):
            x_train_xgb.append(train_sc[i - time_stamp:i])
            y_train_xgb.append(train_sc[i])

        x_train_xgb, y_train_xgb = np.array(x_train_xgb), np.array(y_train_xgb)
        x_train_xgb = x_train_xgb.reshape(len(x_train_xgb), time_stamp)

        # test set
        x_test_xgb, y_test_xgb = [], []
        for i in range(time_stamp, len(test)):
            x_test_xgb.append(test_sc[i - time_stamp:i])
            y_test_xgb.append(test_sc[i])

        x_test_xgb, y_test_xgb = np.array(x_test_xgb), np.array(y_test_xgb)
        x_test_xgb = x_test_xgb.reshape(len(x_test_xgb), time_stamp)

        return x_train_xgb, y_train_xgb, x_test_xgb, y_test_xgb

    def xgboost(self):
        '''
        Train then XGBoost model and save the results
        '''
        x_train_xgb, y_train_xgb, x_test_xgb, y_test_xgb = self.train_test_split_xgb()

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.fit(x_train_xgb, y_train_xgb)

        xgb_pred = model.predict(x_test_xgb)

        # save
        with open('xgb_pred.jsonl', 'w', encoding='utf8') as file:
            xgb_pred_list = xgb_pred.tolist()

            for batch in xgb_pred_list:
                file.write(json.dumps(batch) + '\n')

        with open('y_test_xgb.jsonl', 'w', encoding='utf8') as file:
            y_test_xgb_list = y_test_xgb.tolist()

            for batch in y_test_xgb_list:
                file.write(json.dumps(batch) + '\n')

        return xgb_pred, y_test_xgb

    def xgb_res(self):
        '''
        Inverse transform to get original scaled values
        '''
        if os.path.exists('xgb_pred.jsonl') and os.path.exists('y_test_xgb.jsonl'):

            # read
            with open('xgb_pred.jsonl', 'r', encoding='utf8') as file:
                xgb_pred_list = []

                for line in file:
                    xgb_pred_list.append(json.loads(line.strip()))

                xgb_pred = np.array(xgb_pred_list)

            with open('y_test_xgb.jsonl', 'r', encoding='utf8') as file:
                y_test_xgb_list = []

                for line in file:
                    y_test_xgb_list.append(json.loads(line.strip()))

                y_test_xgb = np.array(y_test_xgb_list)

        else:
            xgb_pred, y_test_xgb = self.xgboost()

        xgb_pred = xgb_pred.reshape(len(xgb_pred), -1)

        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            df = df.set_index(['Date'], drop=True)
            df = pd.DataFrame(df["Adj Close"])
        else:
            df = self.fetch_data()
            df = pd.DataFrame(df["Adj Close"])

        time_stamp = 60
        test = df[int(len(df) * self.train_pct) - time_stamp:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(test)

        xgb_pred_origin = scaler.inverse_transform(xgb_pred)
        y_test_origin = scaler.inverse_transform(y_test_xgb)

        return xgb_pred_origin, y_test_origin

    def plot_xgb(self):
        '''
        Plot XGBoost estimation against real prices
        '''
        xgb_pred_origin, y_test_origin = self.xgb_res()
        plt.figure()
        plt.plot(xgb_pred_origin, label='xgboost', color='red')
        plt.plot(y_test_origin, label='real')
        plt.legend()
        plt.show()

    def lstm_arima_xgboost(self):
        '''
        Combine LSTM-ARIMA model prediction with XGBoost prediction
        '''
        justified, y_test_origin = self.lstm_arima()
        xgb_pred_origin, y_test_origin = self.xgb_res()

        a = (justified * y_test_origin).sum() - (xgb_pred_origin * y_test_origin).sum() - (
                    justified * xgb_pred_origin).sum() + (xgb_pred_origin * xgb_pred_origin).sum()
        b = ((justified - xgb_pred_origin) ** 2).sum()
        lam = a / b
        final = lam * justified + (1 - lam) * xgb_pred_origin
        return final, y_test_origin

    def plot_lstm_arima_xgboost(self):
        '''
        Plot LSTM-ARIMA-XGBoost prediction against real prices
        '''
        final, y_test_origin = self.lstm_arima_xgboost()
        plt.figure()
        plt.plot(final, label='lstm-arima-xgboost', color='red')
        plt.plot(y_test_origin, label='real')
        plt.legend()
        plt.show()

    def judge_indicators(self):
        '''
        Calculate three evaluation criteria (metrics): RMSE, MAE, MAPE
        '''
        y_pred_origin, y_test_origin, fitted_origin, y_train_origin = self.lstm_res()
        justified, y_test_origin = self.lstm_arima()
        final, y_test_origin = self.lstm_arima_xgboost()

        # RMSE
        a = np.sqrt(np.mean(np.power((y_test_origin - y_pred_origin), 2)))
        b = np.sqrt(np.mean(np.power((y_test_origin - justified), 2)))
        c = np.sqrt(np.mean(np.power((y_test_origin - final), 2)))
        # mean_squared_error(test, predictions)

        # MAE
        d = np.mean(abs(y_test_origin - y_pred_origin))
        e = np.mean(abs(y_test_origin - justified))
        f = np.mean(abs(y_test_origin - final))

        # MAPE
        g = np.mean(abs((y_test_origin - y_pred_origin) / y_test_origin))
        h = np.mean(abs((y_test_origin - justified) / y_test_origin))
        i = np.mean(abs((y_test_origin - final) / y_test_origin))

        dict = {'RMSE': [a, b, c], 'MAE': [d, e, f], 'MAPE': [g, h, i]}  # () also works
        df = pd.DataFrame(dict, columns=['RMSE', 'MAE', 'MAPE'], index=['LSTM', 'LSTM-ARIMA', 'LSTM-ARIMA-XGBoost'])

        return df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    predict = LSTM_ARIMA_XGBoost("0700.HK", "2017-01-01", "2022-01-01", 0.8)
    predict.plot_lstm()
    predict.plot_lstm_arima()
    predict.plot_xgb()
    predict.plot_lstm_arima_xgboost()
    predict.judge_indicators()
