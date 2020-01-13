#!/usr/bin/env python
# coding: utf-8

# # Load necessary packages

# In[1]:


import urllib.request, json 
import pandas as pd
from datetime import datetime, timezone
import time
from dateutil.parser import parse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


# In[2]:


def API_reader(link, param =""):
    """This function calls the NHL API and returns a file in a JSON/Dictionary format."""
    with urllib.request.urlopen(link + param) as url:
        data = json.loads(url.read().decode())
    return(data)


# # Find all the game ids for games being played today

# In[3]:


def NHL_games_today(todays_date, print_binary = 0):
    """This function looks at all the games being played today (or input any date in 'YYY-MM-DD' format) then finds their 
    starting times, and sorts them by starting time. Then it calculates how long to wait between starting times."""
    games_links = f"https://statsapi.web.nhl.com/api/v1/schedule?startDate={todays_date}&endDate={todays_date}"
    #games_links = f"https://statsapi.web.nhl.com/api/v1/schedule?startDate=2020-01-02&endDate=2020-01-02"
    dates = API_reader(games_links)
    num_of_games = dates["totalGames"]
    games_id_list = [dates["dates"][0]["games"][i]["gamePk"] for i in range(num_of_games)]
    #Find the difference in seconds between the start times.
    if len(games_id_list) > 1:
        start_times = []
        game_start_dict = {}
        for game_id in games_id_list:
            data = API_reader(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live")
            start_time = data["gameData"]["datetime"]["dateTime"]
            game_start_dict[str(game_id)] = start_time
            start_time = parse(start_time)
            start_times.append(start_time)
            start_times = sorted(start_times)
        delta_seconds_start_times = [(start_times[i+1]- start_times[i]).total_seconds() for i in range(len(start_times)-1)]+ [0]
    else:
        delta_seconds_start_times = [0]
    # Solution to sorting a dict found here: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    game_start_dict = {k : v for k, v in sorted(game_start_dict.items(), key=lambda item: item[1])}
    games_id_list =[int(j) for j in [k for k in game_start_dict.keys()]]
    if print_binary == 1:
        print(f"Number of Games on {todays_date}:", len(games_id_list))
        print("Game Ids: ", games_id_list)
        print(delta_seconds_start_times)
    return((games_id_list, delta_seconds_start_times))


# # Find the difference in seconds between the different start times

# In[4]:


#Use this to only look at select games
#games_id_list = [2019020650,2019020651,2019020652]
#games_id_list = games_id_list[:8]
#print(games_id_list)


# # Find all the first period stats for the games being played today

# In[4]:


#Extract the win, loss, OT records for each team playing.
def _team_records(game_id):  
    """This function is used to extract the team records for the teams playing today."""
    game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
    data = API_reader(game_link)
    away_team_id = data["gameData"]["teams"]["away"]["id"]
    home_team_id = data["gameData"]["teams"]["home"]["id"]
    for team_id in ([away_team_id] + [home_team_id]):
        team_link = f"https://statsapi.web.nhl.com/api/v1/schedule?teamId={team_id}"
        team_id_data = API_reader(team_link)
        home_record = list(team_id_data["dates"][0]["games"][0]["teams"]["home"]["leagueRecord"].values())[:3]
        away_record = list(team_id_data["dates"][0]["games"][0]["teams"]["away"]["leagueRecord"].values())[:3]
    return(home_record + away_record)


# In[5]:


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


RF_from_joblib = joblib.load("RF_Classifier_Model.pkl")
#team_name = input("Enter your team: ")
team_name = 'VAN'
data = API_reader("https://statsapi.web.nhl.com/api/v1/teams")
team_dict = {}
for i in range(len(data["teams"])):
    team_dict[data['teams'][i]["abbreviation"]] = data['teams'][i]["id"]
team_id = team_dict[team_name]
data = API_reader(f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.schedule.next")
specific_game_id = data["teams"][0]["nextGameSchedule"]["dates"][0]["games"][0]["gamePk"]
print(specific_game_id)


# In[7]:


def prepare_vars_for_prediction(specific_game_id):
    """This function prepares a pre-selected teams's game for prediction"""
    df_vars_for_prediction = pd.DataFrame(columns = some_columns + home_team_categories + away_team_categories)
    df_vars_for_prediction.loc[str(specific_game_id)] = home_team + away_team + team_record + home_team_values + away_team_values
    df_vars_for_prediction = differences(df_vars_for_prediction)
    #print(df_vars_for_prediction)
    df_vars_for_prediction["Points_Diff"] = (df_vars_for_prediction["Home_wins"]*2 - df_vars_for_prediction["Away_wins"]*2) + (df_vars_for_prediction["Home_OT"] - df_vars_for_prediction["Away_OT"])
    df_vars_for_prediction = df_vars_for_prediction.drop(columns = ["Home_team", "Away_team", "Home_wins",         "Home_losses", "Home_OT", "Away_wins", "Away_losses", "Away_OT", "Win_Diff", "Loss_Diff", "OT_Diff"])
    df_vars_for_prediction.astype(float)
    #print(df_vars_for_prediction)
    variables =  df_vars_for_prediction.loc[str(specific_game_id)]
    variables = variables.to_numpy().reshape(1,-1)
    #print(variables)
    #print(RF_from_joblib.predict(variables))
    return(RF_from_joblib.predict(variables))


# In[8]:


todays_date = str(datetime.today().year) + "-" + str(datetime.today().month) + "-" + str(datetime.today().day)
games_id_list = NHL_games_today(todays_date, 1)[0]
delta_seconds_start_times = NHL_games_today(todays_date)[1]


# Problem: If a game has a promotion beforehand, it will start late. This could lead to the while loop getting caught in an infinte loop waiting for a previous game to finish the first period when the said game is already in the 2nd period.
# Solution: Create a function that goes through all the games that start at the same time. Then, run this function for as many unique start times as there are, waiting for delta_seconds_start_times after each time the function is run. Create a separate list for each start time.

# In[12]:


groups = [i for i,v in enumerate(delta_seconds_start_times) if v != 0]
group1 = (games_id_list[:0], games_id_list[:0])
group2 = (games_id_list[0:5], games_id_list[0:5])
group3 = (games_id_list[5:6], games_id_list[5:6])
group4 = (games_id_list[6:7], games_id_list[6:7])
group5 = (games_id_list[7:9], games_id_list[7:9])
group6 = (games_id_list[9:], games_id_list[9:])


# In[10]:


#time.sleep(36000)
game_counter = 0
Start_time_counter = 0
df = pd.DataFrame(columns = [i for i in range(30)])
#games_id_list = [2019020692]
for game_id in games_id_list:
    Period = 0 
    while Period < 1:
        game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
        data = API_reader(game_link)
        if data["liveData"]["linescore"]["currentPeriod"] == 1 and data["liveData"]["linescore"]["currentPeriodTimeRemaining"] == "END": 
            team_record = _team_records(game_id)
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
            df.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First period Prediction\\{todays_date}_raw.csv", index = True)
            print("Game ", str(game_counter) ,"/", str(len(games_id_list)), f"ID: {game_id} ({away_team}@{home_team}) completed at: ", str(datetime.today().hour), ":", str(datetime.today().minute))
            if delta_seconds_start_times[Start_time_counter] != 0:
                print("Now sleeping for:", str(delta_seconds_start_times[Start_time_counter]/60/60), "hours.")
            #if game_id == specific_game_id:
            prediction = prepare_vars_for_prediction(game_id)   
            if int(prediction) == 0:
                print(f"The team that is predicted to win is: {home_team}")
            else:
                print(f"The team that is predicted to win is: {away_team}")
            time.sleep(delta_seconds_start_times[Start_time_counter])
            Start_time_counter += 1
        else:
            Period = 0
            print("Check Point: ", datetime.today().hour, ":", datetime.today().minute)
            time.sleep(180) 


# In[11]:


df_all_features = differences(df)
print(df_all_features)
df_all_features.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First period Prediction\\{todays_date}_df_all_features.csv", index = True)


# # Report the Winner

# In[13]:


#time.sleep(60*45)
for game_id in games_id_list:
    game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
    data = API_reader(game_link)
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

# In[14]:


df_all_features_copy = df_all_features.copy()
df_all_features = df_all_features.drop(columns = ["Home_team", "Away_team", "Winner"])
df_all_features.astype(float)
df_all_features.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First period Prediction\\{todays_date}_df_all_features_winner.csv", index = True)


# In[ ]:




