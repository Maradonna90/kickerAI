import json
import pandas as pd
from reader import Reader
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
def main():
    seasons = [9,10,11,12,13,14,15,16,17,18]
    data = {}
    k_data_x = {}
    k_data_y = {}
    r = Reader()
    for season in seasons:
        del_cols = ["height", "weight", "age", "isManOfTheMatch", "isActive", "playedPositions", "playedPositionsShort", "teamRegionName", "regionCode", "tournamentShortName", "playerId", "positionText", "teamId", "teamName", "seasonId", "seasonName", "isOpta", "tournamentId", "tournamentRegionId", "tournamentRegionCode", "tournamentRegionName", "tournamentName"]
        data[season] = pd.DataFrame.from_dict(ws_read_json(season)).drop(columns=del_cols)
        k_data_x[season], k_data_y[season] = r.read("new_data/"+str(season).zfill(2)+".csv")
        k_data_x[season] = pd.DataFrame.from_dict(k_data_x[season])
        k_data_y[season] = pd.DataFrame.from_records({"pts": k_data_y[season]}, columns=["pts"])
        check_fuzzy_proposals(k_data_x[season], data[season])
        new_data = ws_merge_kicker(k_data_x[season], data[season])
        print(new_data.shape)
        print(new_data)
        break
    #print(data)
    #print(k_data_x)
    #print(k_data_y)


def ws_refresh_database(db, data):
    #TODO: refresh the database, add new entries if needed.
    pass

def ws_merge_kicker(k, ws):
    pass
    #TODO: apply merge. WS always has priority!
    #TODO: check at first for relevant entries (non-duplicates)
    #TODO: than get the replacements
    #TODO: than replace with transporting data

def fuzzy_merge(df_1, df_2, key1, key2, threshold=80, limit=1):
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

def check_fuzzy_proposals(k, ws):
    print(k.shape, ws.shape)
    k_name = k[["name"]]
    ws_name = ws[["name"]]
    k_name = k_name.append(ws_name)
    res = k_name.drop_duplicates(keep=False)
    res2 = fuzzy_merge(ws,k, "name", "name")
    print(res2[["name", "matches"]].merge(res, how="inner", on=['name', 'name']))

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
