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

from matplotlib import pyplot
def main():
    #TODO: create a predictor from ws data to kickerpoints to fill out gaps
    #preprocessing
    r = Reader()
    seasons = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
    remove_cols = ['position_x', 'position_y', 'age_x', 'age_y', 'club_x', 'club_y']
    df_names = reduce(lambda left,right: pd.merge(left, right, on='name'), season_X[-4:])
    df_names = df_names.drop(remove_cols, axis=1)
    print(df_names.head())
    predictions = list()
    for (i, row) in df_names.iterrows():
        series = row.iloc[1:-34]
        # training sets should just contain player that are in prediction set and all season of prediction set
        series = np.asarray(series, dtype=int)
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(steps=34)
        yhat = output[0]
        predictions.append(yhat)
        obs = row.iloc[-34:]
        #print('predicted={}, expected={}'.format(yhat, obs))
    test = df_names.iloc[:,-34:]
    print(test)
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    pyplot.plot(test.iloc[1,:])
    pyplot.plot(predictions[1], color='red')
    pyplot.show()
def run_model(name, model, data, seasons, posis, predict=False):
    print("train and predict "+name)
    season_index = len(seasons)
    if predict:
        season_index = -1
    mean_error = []
    pred = None
    dat = None
    real = None
    for season in seasons[1:]:
        train = data[data['season'] < season]
        val = data[data['season'] == season]
        error = mean_squared_error(yts, pred)
        print('Season %d - Error %.5f' % (season, error))
        mean_error.append(error)
    print('Mean Error = %.5f' % np.mean(mean_error))
    #return np.mean(mean_error)
    res = []
    clubs = ws_load_database("clubs")
    names = ws_load_database("names")
    dat = pd.DataFrame(dat)
    for player, test, prediction in zip(dat.iterrows(), real, pred):
        player = player[1]
        res.append([names.inverse[int(player['name'])][0], int(player['age']), clubs.inverse[int(player['club'])][0], posis.inverse[int(player['position'])][0], int(test), prediction])
    write(name+"-lstm-pred", res)

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
