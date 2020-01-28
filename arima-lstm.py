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
from keras.models import model_from_json
def main():
    #TODO: create a predictor from ws data to kickerpoints to fill out gaps
    train()
    #predict_lstm(19, "model")
def train():
    #TODO: normalize data
    r = Reader()
    #seasons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    seasons =  [15, 16, 17, 18]

    season_X = []
    season_Y = []
    clubs = ws_load_database("all_clubs")
    names = ws_load_database("all_names")
    for season in seasons:
        season_x, _ =  r.read("lstm_data/"+str(season).zfill(2)+".csv")
        season_x = pd.DataFrame(season_x)
        #ws_refresh_database("all_names", season_x['name'])
        #ws_refresh_database("all_clubs", season_x['club'])
        #clubs = ws_load_database("all_clubs")
        #names = ws_load_database("all_names")
        season_x['name'] = season_x.apply(lambda x: names[x['name']], axis=1)
        season_x['club'] = season_x.apply(lambda x: clubs[x['club']], axis=1)
        season_X.append(season_x)
    #print(season_X[0].head())
    #print(season_Y[0].head())
    #TODO: filter the training seasons
    df_names = pd.DataFrame()
    remove_cols = ['position_x', 'position_y', 'age_x', 'age_y', 'club_x', 'club_y']
    for k, i in enumerate(range(-4, -len(seasons)-1, -1)):
        print(i, len(seasons)-k)
        df_names = df_names.append(reduce(lambda left,right: pd.merge(left, right, on='name'), season_X[i:len(seasons)-k]))
    df_names = df_names.drop(remove_cols, axis=1)
    print(df_names.shape)
    predictions = list()
    print("ARIMA running...")
    #for (i, row) in df_names.iterrows():
    #    series = row.iloc[1:-34]
    #    # training sets should just contain player that are in prediction set and all season of prediction set
    #    series = np.asarray(series, dtype=int)
    #    model = ARIMA(series, order=(7,1,0))
    #    model_fit = model.fit(disp=0, trend='c')
    #    output = model_fit.forecast(steps=34)
    #    yhat = output[0]
    #    predictions.append(yhat)
    #    obs = row.iloc[-34:]
        #print('predicted={}, expected={}'.format(yhat, obs))
    #test = df_names.iloc[:,-34:]
    #print(test)
    #error = sqrt(mean_squared_error(test, predictions))
    #print('Test RMSE: %.3f' % error)
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
    n_steps = 34*3
    n_features = 1
    train = train.to_numpy()
    train = train.reshape((train.shape[0], train.shape[1], n_features))

    test = test.to_numpy()
    #test = test.reshape((test.shape[0], test.shape[1], n_features))

    # define model
    model = Sequential()
    model.add(LSTM(5, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(34))
    model.compile(optimizer='adam', loss=root_mean_squared_error_sum, metrics=[root_mean_squared_error])
    # fit model
    model.fit(train, test, epochs=10, verbose=1, batch_size=1)

    df_names = pd.DataFrame()
    #demonstrate prediction
    df_names = df_names.append(reduce(lambda left,right: pd.merge(left, right, on='name'), season_X[-4:]))
    df_names = df_names.drop(remove_cols, axis=1)

    train = df_names.iloc[:, 1:-34]
    test = df_names.iloc[:, -34:]
    train = train.to_numpy()
    train = train.reshape((train.shape[0], train.shape[1], n_features))
    test = test.to_numpy()

    yhat = model.predict(train, verbose=0)
    err = root_mean_squared_error(np.array(test, dtype='float64'), yhat)
    print("RMSE:", K.eval(err))
    err = root_mean_squared_error_sum(np.array(test, dtype='float64'), yhat)
    print("RMSE_SUM:", K.eval(err))

    predictions = np.array(predictions)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    for p in np.linspace(0,1,11):
        #print(type(p), type(predictions), type(yhat), predictions.dtype, yhat.dtype)
        y_hybrid = (predictions * p) + (yhat * (1-p))
        err = root_mean_squared_error(np.array(test, dtype='float64'), y_hybrid)
        print(p, "RMSE:", K.eval(err))


def predict_lstm(season, model_string):
    #++++++++++++++++++
    #+++ PREDICTION +++
    #++++++++++++++++++
    # load json and create model
    r = Reader()
    json_file = open(model_string+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_string+".h5")
    print("Loaded model from disk")
    pred_seasons = [season-3, season-2, season-1, season]
    pred_X = []
    clubs = ws_load_database("all_clubs")
    names = ws_load_database("all_names")
    for season in pred_seasons:
        pred_x, _ =  r.read("lstm_data/"+str(season).zfill(2)+".csv")
        pred_x = pd.DataFrame(pred_x)
        ws_refresh_database("all_names", pred_x['name'])
        ws_refresh_database("all_clubs", pred_x['club'])
        clubs = ws_load_database("all_clubs")
        names = ws_load_database("all_names")
        pred_x['name'] = pred_x.apply(lambda x: names[x['name']], axis=1)
        pred_x['club'] = pred_x.apply(lambda x: clubs[x['club']], axis=1)
        pred_X.append(pred_x)
    df_names = pd.DataFrame()
    remove_cols = ['position_x', 'position_y', 'age_x', 'age_y', 'club_x', 'club_y']
    df_names = df_names.append(reduce(lambda left,right: pd.merge(left, right, on='name'), pred_X[0:4]))
    df_names = df_names.drop(remove_cols, axis=1)
    #TODO: do prediction on 19 Season data
    train = df_names.iloc[:, 1:]
    n_features = 1
    train = train.to_numpy()
    train = train.reshape((train.shape[0], train.shape[1], n_features))
    prediction = model.predict(train)
    res_names = df_names.iloc[:,:1]
    res_names = res_names.to_numpy()
    res = np.concatenate((res_names, prediction), axis=1)
    res = pd.DataFrame(res)
    res.iloc[:,:1] = res.apply(lambda x: names.inverse[int(x[0])][0], axis=1)

    cost_data, _ = r.read("int-data.csv", price=True)
    cost_data = pd.DataFrame(cost_data)
    cost_data = cost_data.drop(["club", "position"], axis=1)
    price_pred = fuzzy_merge(res, cost_data, 0, "name", threshold=70)
    price_pred = price_pred[price_pred['matches'].astype(bool)]
    price_pred = pd.merge(price_pred, cost_data, left_on='matches', right_on="name")
    price_pred = price_pred.drop(['matches', 'name'], axis=1)
    #print(price_pred)

    #TODO: add prediction to IDs
    write("lstm-pred", price_pred.to_numpy())
    #TODO: combine predictions and price data

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_error_sum(y_true, y_pred):
    sum_true = K.cast_to_floatx(K.sum(y_true))
    sum_pred = K.cast_to_floatx(K.sum(y_pred))
    sum_res = K.sqrt(K.mean(K.square(sum_pred - sum_true)))
    return sum_res

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

def fuzzy_merge(df_1, df_2, key1, key2, threshold=70, limit=1):
    '''
    df_1 is the left table to join
    df_2 is the right table to join
    key1 is the key column of the left table
    key2 is the key column of the right table
    threshold is how close the matches should be to return a match
    limit is the amount of matches will get returned, these are sorted high to low
    '''
    s = df_2[key2].tolist()

    # get the current row a match from the list of df_2 entries in key2 column
    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))
    # put the found matches from df_2 into column 'matches' in df_1
    df_1['matches'] = m

    # filter all matches above a given threshold and refresh column
    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2

    return df_1


if __name__ == "__main__":
    main()
