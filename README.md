# Results (2019)
* Lasso(alpha=0.1, selection='random', max_iter=500)
	* train_avg_score = -1730.54
	* test_avg_score = 1695.78
* elasticNet(alpha=0.1, l1_ratio=1.0, selection='random', max_iter=1833)
	* train_avg_score = -1727.11
	* test_avg_score = 1695.77

# Conclusions
* Regressor per position
* more individual stats
	* whoscored.com
	* https://www.whoscored.com/StatisticsFeed/1/GetPlayerStatistics?category=summary&subcategory=all&statsAccumulationType=0&isCurrent=true&stageId=15243&tournamentOptions=3&sortBy=Rating&field=Overall&isMinApp=false&page=1&numberOfPlayersToPick=1000
		* maybe some sort of feature engineering so player stats of previous seasons is used for upcoming season (e.g x[18], contains data of 09-17 not 18)
			* use exponentiall weighted moving average for all previous seasons as input, for all stats, including points
	* understat.com
	* 

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
