from reader import Reader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
def main():
    r = Reader()
    data, pred = r.read("pred/BayesianRidge-pred.csv", prediction=True)
    #print(data, pred)
    acc_per_pos(data, pred)
    acc_per_pts(data, pred)
    acc_per_team(data, pred)

def acc_per_pos(data, pred):
    posis = {"Tor": [], "Abwehr": [], "Mittelfeld": [], "Sturm": []}
    y_vals = []
    for p, pred in zip(data, pred):
        posis[p["position"]].append(abs(pred-p["real_pts"]))
    for k, v in posis.items():
        y_vals.append(np.mean(v))
    
    #plotting
    labels = posis.keys()
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x, y_vals, width)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    autolabel(ax, rects1)

    plt.show()

def acc_per_pts(data, pred):
    y_vals = []
    for p, p_pts in zip(data, pred):
        y_vals.append(abs(p["real_pts"]-p_pts))
    bins = np.arange(-75,225, 25)
    x = [p['real_pts'] for p in data]
    y = pred
    plt.hist([x, y], bins, label=['real', 'pred'])
    plt.legend(loc='upper right')
    plt.show()
    
    # pred acc per real points
    real_bin_data = data_in_bins(x, bins, y_vals)
    pred_bin_data = data_in_bins(y, bins, y_vals)
    fig, ax = plt.subplots()
    width = 2.0  # the width of the bars
    rects1 = ax.bar(bins-width, [np.mean(y) for y in real_bin_data], width)
    rects2 = ax.bar(bins+width, [np.mean(y) for y in pred_bin_data], width)
    ax.set_xticks(bins)
    ax.set_xticklabels(bins)
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    ax.legend(["real_pts", "pred_pts"])
    plt.show()

def acc_per_team(data, pred):
    teams = {}
    y_vals = []
    for p, pred in zip(data, pred):
        if p["club"] in teams:
            teams[p["club"]].append(abs(pred-p["real_pts"]))
        else:
            teams[p["club"]] = [abs(pred-p["real_pts"])]
    for k, v in teams.items():
        y_vals.append(np.mean(v))
    
    #plotting
    labels = teams.keys()
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x, y_vals, width)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation="vertical")

    autolabel(ax, rects1)

    plt.show()

def data_in_bins(x, bins, y):
    binplace = np.digitize(x, bins)
    bin_data = []
    for i in range(1,len(bins)+1):
        new_bin = []
        for ind in np.where(binplace == i)[0]:
            new_bin.append(y[ind])
        bin_data.append(new_bin)
    return bin_data

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

if __name__ == "__main__":
    main()
