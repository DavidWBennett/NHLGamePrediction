# NHLGamePrediction
By connecting to the NHL's API, this code seeks to predict if the home team will win after one period has been played.
The Boston Bruins (my favorite team) are team id 6: https://statsapi.web.nhl.com/api/v1/teams/6
Here is the link to their next game: https://statsapi.web.nhl.com/api/v1/teams/6?expand=team.schedule.next 

Relevant features include:
Home team score, Away team score, Score differential (Home-Away), Home team Penalty Infraction Minutes (PIM) (in 1st period only), Away team PIM (in 1st period only), Home team posession time, Away team possession time, Home team advantage, Home team Faceoff win percentage, Home team number of blocked shots, Away team number of blocked shots & Season record.
Target feature is if the home team won or not.

What should the dataset look like?
Each row should be a different game. Each column should be a feature listed above.
