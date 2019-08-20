import csv
import json
import numpy as np
import pandas as pd
from reader import Reader
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from bidict import bidict
from sklearn.metrics import mean_squared_error, make_scorer
import lightgbm as lgb
import pickle
from sklearn.linear_model import Lasso, HuberRegressor
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from skopt.utils import use_named_args
def main():
    seasons = [9,10,11,12,13,14,15,16,17,18,19]
    df = pd.DataFrame({})
    for season in seasons:
         d = pd.read_pickle("db/"+str(season).zfill(2)+".pkl")
         d = d.drop(columns=["firstName", "lastName"])
         d.loc[:,"season"] = season
         df = df.append(d)
    df = df.sort_values(["season", "name"])
    pos = bidict({"Tor": 0, "Abwehr" : 1, "Mittelfeld" : 2, "Sturm" : 3})
    df.loc[:,"position"] = df.loc[:,"position"].map(pos)
    print(df.shape)
    df = df.astype({"age" : "int64"})
    name_ids = df['name'].drop_duplicates().values.tolist()
    feat_eng_df = pd.DataFrame({})
    for n in name_ids:
        rob = df[df.name.isin([n])].copy()
        rob = rob.sort_values(["season"], ascending=False)
        rob['ewma_pts'] = rob['pts']
        ewma_cols = ["apps", "subOn", "manOfTheMatch", "goal", "assistTotal", "shotsPerGame", "aerialWonPerGame", "rating", "minsPlayed", "yellowCard", "redCard", "passSuccess", "ranking", "tacklePerGame", "interceptionPerGame", "foulsPerGame", "offsideWonPerGame", "clearancePerGame", "wasDribbledPerGame", "outfielderBlockPerGame", "goalOwn", "keyPassPerGame", "dribbleWonPerGame", "foulGivenPerGame", "offsideGivenPerGame", "dispossessedPerGame", "turnoverPerGame", "totalPassesPerGame", "accurateCrossesPerGame", "accurateLongPassPerGame", "accurateThroughBallPerGame", "ewma_pts"]
        rob.loc[:,ewma_cols] = rob.loc[:,ewma_cols].ewm(com=.8).mean()
        rob.loc[:,ewma_cols] = rob.loc[:,ewma_cols].shift(periods=-1)
        feat_eng_df = feat_eng_df.append(rob)
    print(feat_eng_df.shape)
    lgbm = lgb.LGBMRegressor()
    #[0.014746003657271805, 29, 66, 179, 0.42233695027257256, 0.7756175270966108, 'dart']
    #1412.0915923717405
    run_model2("lgbm", lgbm, feat_eng_df, seasons, pos, predict=True)
    #hyper_tune("lgbm", lgbm, feat_eng_df, seasons)
    #TODO: make 19 data available (use init_data function and have a case for pred data
def run_model2(name, model, data, seasons, posis, predict=False):
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
        
        xtr, xts = train.drop(['pts'], axis=1), val.drop(['pts'], axis=1)
        ytr, yts = train['pts'].values, val['pts'].values
        
        #mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
        model.fit(xtr, ytr)
        
        #p = np.expm1(mdl.predict(xts))
        pred = model.predict(xts)
        dat = xts
        real = yts
        if predict and seasons[-1] == season:
            break
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
    write(name+"-ws-pred", res)

def hyper_tune(name, model, data, seasons):
    space = [Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
            Integer(1, 30, name='max_depth'),
            Integer(2, 100, name='num_leaves'),
            Integer(10, 1000, name='min_data_in_leaf'),
            Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
            Real(0.1, 1.0, name='subsample', prior='uniform'),
            Categorical(['gbdt','dart','goss'], name='boosting_type'),
         ]

    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        return run_model2(name, model, data, seasons, None)
    res_gp = gp_minimize(objective, space, n_calls=200, random_state=0)
    print("Best Score:", res_gp.fun)
    print("Parameter Vals:", res_gp.x)
def write(filename, res):
    with open(filename+'.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        for data in res:
            writer.writerow(data)



def init_data(seasons, interactive=False):
    data = {}
    r = Reader()
    clubs = []
    names = []
    for season in seasons:
        k_data_x = {}
        k_data_y = {}
        k_data = {}
        del_cols = ["height", "weight", "age", "isManOfTheMatch", "isActive", "playedPositions", "playedPositionsShort", "teamRegionName", "regionCode", "tournamentShortName", "playerId", "positionText", "teamId", "teamName", "seasonId", "seasonName", "isOpta", "tournamentId", "tournamentRegionId", "tournamentRegionCode", "tournamentRegionName", "tournamentName"]
        data[season] = pd.DataFrame.from_dict(ws_read_json(season)).drop(columns=del_cols)
        if interactive:
            k_data_x[season], k_data_y[season] = r.read("new_data/"+str(season).zfill(2)+".csv", interactive=interactive)
        else:
            k_data_x[season], k_data_y[season] = r.read("new_data/"+str(season).zfill(2)+".csv")
        k_data_x[season] = pd.DataFrame.from_dict(k_data_x[season])
        k_data_x[season]["pts"] = k_data_y[season]
        k_data[season] = k_data_x[season]
        data[season] = ws_merge_kicker(k_data[season], data[season])
        clubs.extend(data[season]['club'].drop_duplicates().values.tolist())
        names.extend(data[season]['name'].drop_duplicates().values.tolist())
    ws_refresh_database("clubs", clubs)
    ws_refresh_database("names", names)
    clubs = ws_load_database("clubs")
    names = ws_load_database("names")

    for k, s_dat in data.items():
        s_dat.loc[:,"club"] = s_dat.loc[:,"club"].map(clubs)
        s_dat.loc[:,"name"] = s_dat.loc[:,"name"].map(names)
        s_dat.to_pickle("db/"+str(k)+".pkl")

def ws_refresh_database(db_name, data):
    db = bidict({})
    db.load("db/"+db_name)
    for dat in data:
        if dat not in db.keys():
            db[dat] = len(db.values())
    db.save("db/"+db_name)

def ws_load_database(db_name):
    db = bidict({})
    db.load("db/"+db_name)
    return db

def ws_merge_kicker(k, ws):
    k = k.drop_duplicates(subset=['name'], keep='first')
    ws = ws.drop_duplicates(subset=['name'], keep='first')
    print(ws.shape)
    merged = fuzzy_merge(ws, k, 'name', 'name')
    merged = merged.merge(k, how='inner', left_on=['matches'], right_on=['name'])
    merged = merged.drop(["name_y", "matches"], axis=1)
    merged.rename(columns={'name_x': 'name'}, inplace=True)
    print(merged.shape)
    return merged

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

def ws_read_json(season):
    data = []
    with open('whoscored/ws-'+str(season).zfill(2)+'-summary.json') as sum_data, open('whoscored/ws-'+str(season).zfill(2)+'-defensive.json') as def_data, open("whoscored/ws-"+str(season).zfill(2)+"-offensive.json") as off_data, open("whoscored/ws-"+str(season).zfill(2)+"-passing.json") as pass_data:
        sum_ = json.load(sum_data)
        def_ = json.load(def_data)
        off_ = json.load(off_data)
        pass_ = json.load(pass_data)
        for s, d, o, p in zip(sum_["playerTableStats"], def_["playerTableStats"], off_["playerTableStats"], pass_["playerTableStats"]):
            data.append({**s, **d, **o, **p})

        return data
if __name__ == "__main__":
    main()
