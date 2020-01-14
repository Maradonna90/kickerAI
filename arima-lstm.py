import csv
import json
import numpy as np
import pandas as pd
from reader import Reader
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from bidict import bidict
from sklearn.metrics import mean_squared_error, make_scorer
#import lightgbm as lgb
import pickle
from sklearn.linear_model import Lasso, HuberRegressor
from statsmodels.tsa.arima_model import ARIMA
from functools import reduce
from math import sqrt
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras import backend as K
def main():
    #TODO: create a predictor from ws data to kickerpoints to fill out gaps
    #preprocessing
    #TODO: rolling forecast through all years beginning from 0-2 train and predict 3
    r = Reader()
    seasons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    season_X = []
    season_Y = []
    clubs = ws_load_database("all_clubs")
    names = ws_load_database("all_names")
    for season in seasons:
        season_x, _ =  r.read("lstm_data/"+str(season).zfill(2)+".csv")
        season_x = pd.DataFrame(season_x)
        ws_refresh_database("all_names", season_x['name'])
        ws_refresh_database("all_clubs", season_x['club'])
        clubs = ws_load_database("all_clubs")
        names = ws_load_database("all_names")
        season_x['name'] = season_x.apply(lambda x: names[x['name']], axis=1)
        season_x['club'] = season_x.apply(lambda x: clubs[x['club']], axis=1)
        season_X.append(season_x)
    #print(season_X[0].head())
    #print(season_Y[0].head())
    #TODO: filter the training seasons
    remove_cols = ['position_x', 'position_y', 'age_x', 'age_y', 'club_x', 'club_y']
    df_names = reduce(lambda left,right: pd.merge(left, right, on='name'), season_X[-4:])
    df_names = df_names.drop(remove_cols, axis=1)
    #print(df_names.head())
    predictions = list()
    print("ARIMA running...")
    for (i, row) in df_names.iterrows():
        series = row.iloc[1:-34]
        # training sets should just contain player that are in prediction set and all season of prediction set
        series = np.asarray(series, dtype=int)
        model = ARIMA(series, order=(7,1,0))
        model_fit = model.fit(disp=0, trend='c')
        output = model_fit.forecast(steps=34)
        yhat = output[0]
        predictions.append(yhat)
        obs = row.iloc[-34:]
        #print('predicted={}, expected={}'.format(yhat, obs))
    test = df_names.iloc[:,-34:]
    #print(test)
    error = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % error)
    #pyplot.plot(test.iloc[1,:])
    #pyplot.plot(predictions[1], color='red')
    #pyplot.show()
    #+++++++++++++++++
    #+ LSTM TRAINING +
    #+++++++++++++++++
    print("Running LSTM...")
    output_dim = 34
    train = df_names.iloc[:, 1:-34]
    test = df_names.iloc[:, -34:]
    n_steps = 3*34
    n_features = 1
    train = train.to_numpy()
    train = train.reshape((train.shape[0], train.shape[1], n_features))

    test = test.to_numpy()
    #test = test.reshape((test.shape[0], test.shape[1], n_features))

    # define model
    # tf.keras.metrics.RootMeanSquaredError()
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(34))
    model.compile(optimizer='adam', loss=root_mean_squared_error)
    # fit model
    model.fit(train, test, epochs=200, verbose=0)
    # demonstrate prediction
    yhat = model.predict(train, verbose=0)
    predictions = np.array(predictions)
    for p in np.linspace(0,1,11):
        #print(type(p), type(predictions), type(yhat), predictions.dtype, yhat.dtype)
        y_hybrid = (predictions * p) + (yhat * (1-p))
        err = root_mean_squared_error(np.array(test, dtype='float64'), y_hybrid)
        print(p, "RMSE:", err.numpy())

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def write(filename, res):
    with open(filename+'.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for data in res:
            writer.writerow(data)

def ws_load_database(db_name):
    db = bidict({})
    db.load("db/"+db_name)
    return db

def ws_refresh_database(db_name, data):
    db = bidict({})
    db.load("db/"+db_name)
    for dat in data:
        if dat not in db.keys():
            db[dat] = len(db.values())
    db.save("db/"+db_name)

if __name__ == "__main__":
    main()
