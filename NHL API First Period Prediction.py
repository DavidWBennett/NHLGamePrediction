#!/usr/bin/env python
# coding: utf-8

# # Load necessary packages

# In[2]:


import urllib.request, json 
import pandas as pd
from datetime import datetime, timezone
import time
from dateutil.parser import parse


# # Find all the game ids for games being played today

# In[3]:


todays_date = str(datetime.today().year) + "-" + str(datetime.today().month) + "-" + str(datetime.today().day)
games_links = f"https://statsapi.web.nhl.com/api/v1/schedule?startDate={todays_date}&endDate={todays_date}"
#games_links = f"https://statsapi.web.nhl.com/api/v1/schedule?startDate=2020-01-02&endDate=2020-01-02"
with urllib.request.urlopen(games_links) as url:
    dates = json.loads(url.read().decode())
num_of_games = dates["totalGames"]
games_id_list = [dates["dates"][0]["games"][i]["gamePk"] for i in range(num_of_games)]
games_id_list.sort() #Need to order the game id list so the start times are in order.
print(f"Number of Games on {todays_date}:", len(games_id_list))
print("Game Ids: ", games_id_list)


# # Find the difference in seconds between the different start times

# In[5]:


#Use this to only look at select games
#games_id_list = [2019020615]#, 2019020614]#, 2019020608]
#games_id_list = games_id_list[7:]
#print(games_id_list)


# In[21]:


if len(games_id_list) > 1:
    start_times = []
    for game_id in games_id_list:
        with urllib.request.urlopen(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live") as url:
            data = json.loads(url.read().decode())
        start_time = data["gameData"]["datetime"]["dateTime"]
        #current_time = datetime.now(timezone.utc)
        start_time = parse(start_time)
        start_times.append(start_time)
        start_times = sorted(start_times)
    delta_seconds_start_times = [(start_times[i+1]- start_times[i]).total_seconds() for i in range(len(start_times)-1)]
else:
    delta_seconds_start_times = [0]
#print(len(delta_seconds_start_times))
#print(delta_seconds_start_times)


# # Find all the first period stats for the games being played today

# In[7]:


#Extract the win, loss, OT records for each team playing.
def _team_records(game_id):  
    """This function is used to extract the team records for the teams playing today."""
    game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
    with urllib.request.urlopen(game_link) as url:
        data = json.loads(url.read().decode())
    away_team_id = data["gameData"]["teams"]["away"]["id"]
    home_team_id = data["gameData"]["teams"]["home"]["id"]
    for team_id in ([away_team_id] + [home_team_id]):
        team_link = f"https://statsapi.web.nhl.com/api/v1/schedule?teamId={team_id}"
        with urllib.request.urlopen(team_link) as url:
            team_id_data = json.loads(url.read().decode())
        home_record = list(team_id_data["dates"][0]["games"][0]["teams"]["home"]["leagueRecord"].values())[:3]
        away_record = list(team_id_data["dates"][0]["games"][0]["teams"]["away"]["leagueRecord"].values())[:3]
    return(home_record + away_record)


# In[8]:


#df.loc[str(game_id)] = home_team + away_team + team_records + home_team_values + away_team_values
def differences(df):
    """Used to calculate the feature differences between the home and away teams, and to
    convert the percentage features to numeric."""
    df["Win_Diff"] = df["Home_wins"] - df["Away_wins"]
    df["Loss_Diff"] = df["Home_losses"] - df["Away_losses"]
    df["OT_Diff"] = df["Home_OT"] - df["Away_OT"]
    df["Goals_Diff"] = df["Home_goals"] - df["Away_goals"]
    df["Shots_Diff"] = df["Home_shots"] - df["Away_shots"]
    df["Blocked_Diff"] = df["Home_blocked"] - df["Away_blocked"]
    df["PIM_Diff"] = df["Home_pim"] - df["Away_pim"]
    df["PowerPlayGoals_Diff"] = df["Home_powerPlayGoals"] - df["Away_powerPlayGoals"]
    df["Takeaways_Diff"] = df["Home_takeaways"] - df["Away_takeaways"]
    df["Giveaways_Diff"] = df["Home_giveaways"] - df["Away_giveaways"]
    df["Hits_Diff"] = df["Home_hits"] - df["Away_hits"]
    df["Home_powerPlayPercentage"] = pd.to_numeric(df["Home_powerPlayPercentage"])/100
    df["Away_powerPlayPercentage"] = pd.to_numeric(df["Away_powerPlayPercentage"])/100
    df["Home_faceOffWinPercentage"] = pd.to_numeric(df["Home_faceOffWinPercentage"])/100
    df["Away_faceOffWinPercentage"] = pd.to_numeric(df["Away_faceOffWinPercentage"])/100
    return(df)


# In[9]:


game_counter = 0
df = pd.DataFrame(columns = [i for i in range(30)])
for game_id in games_id_list:
    #print(game_id)
    Period = 0 
    Start_time_counter = 0
    while Period < 1:
        game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
        with urllib.request.urlopen(game_link) as url:
            data = json.loads(url.read().decode())
        if data["liveData"]["linescore"]["currentPeriod"] == 1 and data["liveData"]["linescore"]["currentPeriodTimeRemaining"] == "END": 
            team_record = team_records(game_id)
            some_columns = ["Home_team", "Away_team", "Home_wins", "Home_losses", "Home_OT", "Away_wins", "Away_losses", "Away_OT"]
            home_team_categories = list(data['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats'].keys())
            away_team_categories = list(data['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats'].keys())
            home_team_categories = [f"Home_{i}" for i in home_team_categories]
            away_team_categories = [f"Away_{i}" for i in away_team_categories]
            home_team = [data["gameData"]["teams"]["home"]["triCode"]]
            away_team = [data["gameData"]["teams"]["away"]["triCode"]]
            df.columns = some_columns + home_team_categories + away_team_categories
            away_team_stats = data['liveData']['boxscore']['teams']['away']['teamStats']['teamSkaterStats']
            home_team_stats = data['liveData']['boxscore']['teams']['home']['teamStats']['teamSkaterStats']
            home_team_values = list(home_team_stats.values())
            away_team_values = list(away_team_stats.values())
            df.loc[str(game_id)] = home_team + away_team + team_record + home_team_values + away_team_values
            Period = 1
            game_counter += 1
            print("Game ", str(game_counter) ,"/", str(len(games_id_list)), f". ID: {game_id} completed at: ", str(datetime.today().hour), ":", str(datetime.today().minute))
            time.sleep(delta_seconds_start_times[Start_time_counter])
            Start_time_counter += 1
        else:
            Period = 0
            print("Check Point: ", datetime.today().hour, ":", datetime.today().minute)
            time.sleep(300) 
df_all_features = differences(df)
print(df_all_features)


# # Report the Winner

# In[11]:


for game_id in games_id_list:
    game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
    with urllib.request.urlopen(game_link) as url:
        data = json.loads(url.read().decode())
    home_team = data["gameData"]["teams"]["home"]["triCode"]
    away_team = data["gameData"]["teams"]["away"]["triCode"]
    num_periods = len(data["liveData"]["linescore"]["periods"])
    if data["liveData"]["linescore"]["hasShootout"] == False:
        if sum([int(data["liveData"]["linescore"]["periods"][i]["home"]["goals"]) for i in range(num_periods)]) >             sum([int(data["liveData"]["linescore"]["periods"][i]["away"]["goals"]) for i in range(num_periods)]):#2 for the third period, using 0 indexing
            print(f"{home_team} Wins")
            df_all_features.loc[str(game_id), "Winner"] = home_team
            df_all_features.loc[str(game_id), "Winner_binary"] = 0
        else: 
            print(f"{away_team} Wins")
            df_all_features.loc[str(game_id), "Winner"] = away_team
            df_all_features.loc[str(game_id), "Winner_binary"] = 1
    else:
        if int(data["liveData"]["linescore"]["shootoutInfo"]["home"]["scores"]) >             int(data["liveData"]["linescore"]["shootoutInfo"]["away"]["scores"]):#2 for the third period, using 0 indexing
            print(f"{home_team} Wins")
            df_all_features.loc[str(game_id), "Winner"] = home_team
            df_all_features.loc[str(game_id), "Winner_binary"] = 0
        else: 
            print(f"{away_team} Wins")
            df_all_features.loc[str(game_id), "Winner"] = away_team
            df_all_features.loc[str(game_id), "Winner_binary"] = 1


# # Prepare the final dataset for analysis by converting everything to floats

# In[12]:


df_all_features_copy = df_all_features.copy()
df_all_features = df_all_features.drop(columns = ["Home_team", "Away_team", "Winner"])
df_all_features.astype(float)
df_all_features.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First Period Prediction\\{todays_date}_df_all_features.csv", index = True)

