# Results (2019)
* Lasso(alpha=0.1, selection='random', max_iter=500)
	* train_avg_score = -1730.54
	* test_avg_score = 1695.78
* elasticNet(alpha=0.1, l1_ratio=1.0, selection='random', max_iter=1833)
	* train_avg_score = -1727.11
	* test_avg_score = 1695.77
* LGBM(0.014746003657271805, 29, 66, 179, 0.42233695027257256, 0.7756175270966108, 'dart')
	* 1412.0915923717405
* LGBM(vanilla)
	* 1474.90919
# Conclusions
* more individual stats
	* whoscored.com
	* https://www.whoscored.com/StatisticsFeed/1/GetPlayerStatistics?category=summary&subcategory=all&statsAccumulationType=0&isCurrent=true&stageId=15243&tournamentOptions=3&sortBy=Rating&field=Overall&isMinApp=false&page=1&numberOfPlayersToPick=1000
		* maybe some sort of feature engineering so player stats of previous seasons is used for upcoming season (e.g x[18], contains data of 09-17 not 18)
	* understat.com
	* 

# Teams
* Lasso: 442 (28.01.20)
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

* Handpicked: 448 (28.01.20)
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

# KickerAI 2020
## Datasources
	* Understat.com
		* xG and derived advanced stats on week-by-week basis
	* Football Manager (GER, ENG, SPA, ITA, FRA)
		* CA: Aiding prediction for missing player kicker points (maybe even as a feature for weekly predictions?)
		* PA: Aiding as a talent value, could help to identify the young playres that might have a break through season in comparison to squad fillers
		* Injury Proneness: how inconsistent playing time might be
	* historical kicker points
		* as a time series prediction
	* Custom Features
		* squad ranking (CA rank within team)

## Data Merging Pipeline
1. Interactive Price Data as basis
2. Fuzzy merge with historic kicker data of k-seasons (k=3)
	2.1 Dataset is split into two parts: 1) players who have historic kicker points for the last three season (h_players) and players who don't (nh_player)
3. Merge nh_players with k seasons of historic understat data (k=3) => Filter players from obscure leagues and not enough historic data
4. Merge nh_players with historc FM data
	4.1 If players are not found manually export them from missing seasons
5. Train a predictor to predict the missing kicker points for nh_players to fill gaps.
6. Concatenate nh_players and h_players
7. Calculate custom features
8. Train kicker points predictor
