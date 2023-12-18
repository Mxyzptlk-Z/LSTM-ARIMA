import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from statsmodels.tsa.arima.model import ARIMA
import warnings


class LSTM_ARIMA:
    '''
    This is a class that uses LSTM neural network and ARIMA model to predict stock prices
    '''

    def __init__(self, df):
        self.df = df

    def lstm_arima(self):
        # lstm step
        df = self.df
        time_stamp = 60

        train = df[:len(df) - time_stamp]
        x_pred = df[len(df) - time_stamp:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_sc = scaler.fit_transform(train)
        x_pred_sc = scaler.fit_transform(x_pred)

        x_train, y_train = [], []

        for i in range(time_stamp, len(df)):
            x_train.append(np.vstack((x_train_sc, x_pred_sc))[i - time_stamp:i])
            y_train.append(np.vstack((x_train_sc, x_pred_sc))[i, 3])

        x_train, y_train = np.array(x_train), np.array(y_train)

        model = Sequential()
        model.add(LSTM(units=15, return_sequences=True))
        model.add(Dropout(0.02))
        model.add(LSTM(units=15, return_sequences=False))
        model.add(Dropout(0.02))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=1, callbacks=[early_stop])

        y_pred = model.predict(np.array([x_pred_sc]))
        fitted_values = model.predict(x_train)

        scaler.fit_transform(pd.DataFrame(df['Adj Close'].values))

        y_pred_origin = scaler.inverse_transform(y_pred)
        fitted_origin = scaler.inverse_transform(fitted_values)
        y_train_origin = scaler.inverse_transform(y_train.reshape(len(y_train), 1))

        resid = fitted_origin - y_train_origin
        ts = pd.DataFrame(resid)
        ts.index = df.index[:len(ts)]
        ts.columns = ['residual']
        ts = ts['residual']

        # arima step
        pmax = 5
        qmax = 5
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
        p, q = bic_matrix.stack().idxmin()

        model_fit = ARIMA(ts, order=(p, 0, q)).fit()
        output = model_fit.forecast()
        yhat = np.array(output)

        adj_pred = y_pred_origin[0] - yhat

        return adj_pred


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    df = yf.download("0700.HK", "2017-01-01", "2022-01-03")
    pred = LSTM_ARIMA(df)
    price = pred.lstm_arima()
    print("Next day price is estimated to be: ", price)

