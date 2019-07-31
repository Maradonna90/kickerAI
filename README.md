# Results (2019)
* LGBM
	* train_avg_score = -1760.51
	* test_avg_score = 2112.20
* BRDG (stock)
	* train_avg_score = -1381.04
	* test_avg_score = 1722.98
* XGB
	* train_avg_score = -1655.42
	* test_avg_score = 2031.60
* Lasso
	* train_avg_score = -1897.46
	* test_avg_score = 1936.68
* elasticNet
	* train_avg_score = -2002.45
	* test_avg_score = 1849.47
* SVR (linear)
	* train_avg_score = -1846.32
	* test_avg_score = 1781.06
* EnsembleRegressors
	* train_avg_score = 
	* test_avg_ score =

* Lasso(alpha=0.1, selection='random', max_iter=500)
	* train_avg_score = -1730.54
	* test_avg_score = 1695.78
* elasticNet(alpha=0.1, l1_ratio=1.0, selection='random', max_iter=1833)
	* train_avg_score = -1727.11
	* test_avg_score = 1695.77
* SVR()
	* training_avg_score =
	* test_avg_score =
* BRDG()
	* training_avg_score = 
	* test_avg_score = 
=> SVR and BRDG fails

# How to improve performance
* more features via feature engineering?
	* what features though?
* more data?
* one specific regressor for each position

# analyser
* check accuracy for bin classificatioun

# Conclusions
* trust lasso and elastic on players with pred_pts < 25 and > 150
* inbetween is very random worst range 75-125
* Goalie prediction bad, probably bc unsure whos main and goalies do not injure as much and in general are not rotated due to fatigue
* Regressor per position
* more individual stats
	* whoscored.com
	* https://www.whoscored.com/StatisticsFeed/1/GetPlayerStatistics?category=summary&subcategory=all&statsAccumulationType=0&isCurrent=true&stageId=15243&tournamentOptions=3&sortBy=Rating&field=Overall&isMinApp=false&page=1&numberOfPlayersToPick=1000
		* subcategory: passing, offensive, defensive, summary
		* alle season so holen, dann zippen und parsen und mergen mit den punkten
		* DBs aufbuauen für Name -> NameId, club -> clubId (bidirectionale dicts?)
		* maybe some sort of feature engineering so player stats of previous seasons is used for upcoming season (e.g x[18], contains data of 09-17 not 18)
			* use exponentiall weighted moving average for all previous seasons as input, for all stats, including points
	* understat.com
	* 

# Strategies
* pick goalies that are cheap and are starter by pred pts.
* pick cheapest defenders if in random range, otherwise pick top
* pick top MF
* pick top ST

# Teams
* Lasso
	* Timo Horn
	* Jerome Roussillon
	* Wout Weghorst
	* Thomas Müller
	* Claudio Pizarro
	* Kevin Stöger
	* Heintz
	* Modeste
	* Sahin
	* hummels
	* Witsel 

* Handpicked
	* Timo Horn
	* Subotic
	* Ndicka
	* Plattenhardt
	* Delaney
	* Neuhaus
	* Kampl
	* Gnabry
	* Plea
	* Paco
	* Poulsen
