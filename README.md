# NHLGamePrediction
By connecting to the NHL's API, this code seeks to predict if the home team will win after one period has been played.
This code can be run at anytime and it will sleep until the next game's first intermission. After the first intermisison the stats are recorded and a prediction is made.

Relevant features include:
Home team score, Away team score, Score differential (Home-Away), Home team Penalty Infraction Minutes (PIM) (in 1st period only), Away team PIM (in 1st period only), Home team advantage, Home team Faceoff win percentage, Home team number of blocked shots, Away team number of blocked shots & Season record.
Target feature is if the home team won or not.

Using a Random Forest Classifier the model is created and updated after each day. 
