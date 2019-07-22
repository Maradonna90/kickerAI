from reader import Reader
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking
from keras.wrappers.scikit_learn import KerasRegressor
from collections import Counter
from sklearn.linear_model import BayesianRidge
import csv
from time import time
from scipy.stats import randint as sp_randint
import numpy as np
import multiprocessing
from parser import Parser
import pandas
import lightgbm as lgb
def main():
    r = Reader()
    seasons = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    x_train = {}
    y_train = {}
    x_test = {}
    y_test = {}
    vec = DictVectorizer(sparse=False)
    for season in seasons[:-1]:
        x, y = r.read("data/"+str(season)+".csv")
        x_train[season], y_train[season] = x, y
    x, y = r.read("data/"+str(seasons[-1])+".csv")
    x_test[seasons[-1]], y_test[seasons[-1]] = x, y
    
    #  read pred_data
    x, y = r.read("data/19.csv", interactive=True)
    x_pred = {}
    y_pred = {}
    x_pred[19], y_pred[19] = x, y

   
    #x_test.update(x_train)
    #y_test.update(y_train)
    #x_train, y_train = transform_to_lstm(x_train, y_train)
    #x_test, y_test = transform_to_lstm(x_test, y_test)
    x_train, y_train = dict_list_transform(x_train, y_train)
    x_test, y_test = dict_list_transform(x_test, y_test)
    pred_data, y_pred = dict_list_transform(x_pred, y_pred)

    #print(len(x_test[0]))
    #print(x_data['Marco Reus'], y_data['Marco Reus'])
    x_all = pandas.DataFrame(x_test+x_train+pred_data, columns=['name', 'position', 'age', 'club'])
    x_train = pandas.DataFrame(x_train, columns=['name', 'position', 'age', 'club'])
    x_test = pandas.DataFrame(x_test, columns=['name', 'position', 'age', 'club'])
    pred_data = pandas.DataFrame(pred_data, columns=['name', 'position', 'age', 'club'])
    vec.fit(x_all.to_dict('records'))
    #print(x_test.to_dict('records'))
    #train = pandas.DataFrame(x_train.assign(pts=y_train), columns=['name', 'position', 'age', 'club', 'pts'])
    X_train, X_test = vec.transform(x_train.to_dict('records')), vec.transform(x_test.to_dict('records'))
    pred_data = vec.transform(pred_data.to_dict('records'))
    
    X_train = pandas.DataFrame(X_train).values
    X_test = pandas.DataFrame(X_test).values
    y_train = pandas.DataFrame(y_train).values
    y_test = pandas.DataFrame(y_test).values
    
    # init other
    kf = KFold(shuffle=True)
    tscv = TimeSeriesSplit(n_splits=3)
    scaler = StandardScaler()
    lgbm = lgb.LGBMRegressor(boosting_type='dart', num_leaves=40, learning_rate=0.1)
    brdg = BayesianRidge(compute_score=True)
    
    #get interactive data
    p = Parser()
    p_int = p.parse_interactive()
    run_model("BayesianRidge", brdg, [scaler], X_train, X_test, y_train, y_test, kf, vec, cv=False, out=False, pred_data=pred_data, price_data=None, hyper=True)
    #run_model("LGBM", lgbm, [scaler], X_train, X_test, y_train, y_test, kf, vec, cv=True, out=False, pred_data=None, price_data=p_int, hyper=True)


def run_model(name, model, steps, x_train, x_test, y_train, y_test, kfold, vec, non_cv=False, cv=False, out=False, para=False, pred_data=None, price_data=None, hyper=False):
    print("starting", name)
    clf = make_pipeline(*steps, model)
    if cv:
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scorer)
        print("train_avg_score_"+name+":", np.mean(scores))
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        scores = mean_squared_error(y_test, predict)
        print("test_avg_score_"+name+":", np.mean(scores))
    if non_cv:
        clf.fit(x_train, y_train)
        predict = clf.predict(x_test)
        scores = mean_squared_error(y_test, predict)
        print("test_avg_score_"+name+":", np.mean(scores))
        #print(predict)

    if out:
        res = []
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
        #print(vec.inverse_transform(x_test))
        for player, test, prediction in zip(vec.inverse_transform(x_test), y_test, predict):
            player = [p.split("=")[1] for p in player.keys()]
            if player[1] != 'None':
                res.append([player[2], player[0], player[1], player[3], *test, prediction])
        write(name+"-pred", res)

    if pred_data is not None and price_data is not None:
        print("Start real pred")
        x_all = np.concatenate([x_train,x_test])
        y_all = np.concatenate([y_train, y_test])
        clf.fit(x_all, y_all)
        predict = clf.predict(pred_data)
        pred_data = pred_data.reshape(pred_data.shape[0], pred_data.shape[1])
        res = []
        for p_name, p_price, p_club in price_data:
            p_out = []
            for player, prediction in zip(vec.inverse_transform(pred_data), predict):
                player = [p.split("=")[1] for p in player.keys()]
                if p_club.lower() in player[1].lower() and p_name in player[2]:
                    p_out = [player[2], player[0], player[1], player[3], prediction]
                    p_out.append(p_price)
            res.append(p_out)
        write(name+"-pred-real", res)

    if hyper:
        param_dist = {"alpha_1": [0.00001, 0.000001, 0.0000001],
              "alpha_2": [0.00001, 0.000001, 0.0000001],
              "lambda_1": [0.00001, 0.000001, 0.0000001],
              "lambda_2": [0.00001, 0.000001, 0.0000001],
              "n_iter": [300]
            }
        clf = model
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        x_all = np.concatenate([x_train,x_test])
        y_all = np.concatenate([y_train, y_test])
        #param_dist = dict(optimizer=optimizers, epochs=epochs)
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=10, verbose=2, scoring=scorer, n_jobs=multiprocessing.cpu_count(), cv=kfold)
        start = time()

        random_search.fit(x_all, y_all)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter settings." % ((time() - start), 10))
        report(random_search.cv_results_)


def dict_list_transform(x_data, y_data):
    x_res = []
    y_res = []
    for season, season_pts in zip(x_data.values(), y_data.values()):
        for player, pts in zip(season, season_pts):
            x_res.append([player['name'], player['position'], player['age'], player['club']])
            y_res.append(pts)
    return x_res, y_res

def write(filename, res):
    with open(filename+'.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for data in res:
            writer.writerow(data)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
if __name__ == "__main__":
    main()
