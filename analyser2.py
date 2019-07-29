from reader import Reader
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
def main():
    r = Reader()
    seasons = [0,1,2,3,4,5,6,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18]
    x_train = {}
    y_train = {}
    for season in seasons:
        x, y = r.read("new_data/"+str(season).zfill(2)+".csv")
        x_train[season], y_train[season] = x, y
    pts_per_age(x_train, y_train)

def pts_per_club(x_train, y_train):
    #TODO: implement
    pass

def pts_per_age(x_train, y_train):
    vals = []
    for data, pts in zip(x_train.values(), y_train.values()):
        for p_data, p_pts in zip(data, pts):
            vals.append((int(p_data["age"]), p_pts))
    #print(players)
    ages = defaultdict(list)
    for k,v in vals:
        ages[k].append(v)
    ages = OrderedDict(sorted(ages.items()))
    x = ages.keys()
    y = ages.values()
    plt.xlabel("age")
    plt.ylabel("points")
    plt.boxplot(y)
    print(x)
    plt.xticks(range(1, len(x)+1), list(x))
    plt.show()
    #plt.savefig('img/'+key+'_pts_per_age.png', bbox_inches="tight")
    #plt.close()
if __name__ == "__main__":
    main()

