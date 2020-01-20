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
import collections


# In[2]:


def API_reader(link, param =""):
    """This function calls the NHL API and returns a file in a JSON/Dictionary format."""
    with urllib.request.urlopen(link + param) as url:
        data = json.loads(url.read().decode())
    return(data)


# # Find all the game ids for games being played today

# In[3]:


#Try this one on 1/15/20
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
        print(len(set(start_times))) #6 different start times
        #print(game_start_dict)
    return((games_id_list, delta_seconds_start_times, game_start_dict, start_times))


# In[4]:


#On 1/15/20 try this:
RF_from_joblib = joblib.load("RF_Classifier_Model.pkl")
todays_date = str(datetime.today().year) + "-" + str(datetime.today().month) + "-" + str(datetime.today().day)
games_id_list = NHL_games_today(todays_date, 1)[0]
delta_seconds_start_times = NHL_games_today(todays_date)[1]
game_start_dict = NHL_games_today(todays_date)[2]
start_times = NHL_games_today(todays_date)[3]
delta_seconds_start_times_unique = [x for x in delta_seconds_start_times if x != 0.0 and type(x) == float]
delta_seconds_start_times_unique = delta_seconds_start_times_unique + [0]


# In[5]:


#on 1/15/20 try this
def regroup_games_by_start_times(games_id_list):
#In order to group the games by their start times I need to reformat the dictionary.
    game_start_dict_reformatted = []
    for i in range(len(games_id_list)): 
        game_start_dict_reformatted.append({"game" : games_id_list[i], "start_time" : start_times[i]})
    #Grouping dictionary items: https://www.saltycrane.com/blog/2014/10/example-using-groupby-and-defaultdict-do-same-task/
    grouped = collections.defaultdict(list)
    for item in game_start_dict_reformatted:
        grouped[item['start_time']].append(item)    
    game_start_list = []
    for i in grouped.items():
        game_start_list.append(i[1])
    # Now create a list of the games according to their start times.
    games_grouped_by_start = []
    for j in range(len(game_start_list)):
        groups_of_start_times = []
        for i in range(len(game_start_list[j])):
            groups_of_start_times.append(game_start_list[j][i]["game"])
        games_grouped_by_start.append(groups_of_start_times)
    #print(len(games_grouped_by_start))    
    return(games_grouped_by_start)

list_of_groups = regroup_games_by_start_times(games_id_list)
print(list_of_groups)


# In[6]:


def order_same_start_times(games_id_sublist):
    time_remaining = []
    for game_id in games_id_sublist:
        game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
        data = API_reader(game_link)
        time_remaining.append(data["liveData"]["linescore"]["currentPeriodTimeRemaining"])
    time_remaining_dict = dict(zip(games_id_sublist, time_remaining))
    time_remaining_dict = {k : v for k,v in sorted(time_remaining_dict.items(), key = lambda item: item[1])}
    return(list(time_remaining_dict.keys()))


# # Find all the first period stats for the games being played today

# In[7]:


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


# In[33]:


def prepare_vars_for_prediction(prep_df, specific_game_id):
    """This function prepares a teams's first period stats for prediction"""
    #df_vars_for_prediction = pd.DataFrame(columns = some_columns + home_team_categories + away_team_categories)
    df_vars_for_prediction = pd.DataFrame(columns = list(prep_df.columns))
    #df_vars_for_prediction.loc[str(specific_game_id)] = home_team + away_team + team_record + home_team_values + away_team_values
    df_vars_for_prediction.loc[str(specific_game_id)] = prep_df.loc[str(specific_game_id)]
    df_vars_for_prediction = differences(df_vars_for_prediction)
    df_vars_for_prediction["Points_Diff"] = (df_vars_for_prediction["Home_wins"]*2 - df_vars_for_prediction["Away_wins"]*2) + (df_vars_for_prediction["Home_OT"] - df_vars_for_prediction["Away_OT"])
    df_vars_for_prediction = df_vars_for_prediction.drop(columns = ["Home_team", "Away_team", "Home_wins",         "Home_losses", "Home_OT", "Away_wins", "Away_losses", "Away_OT", "Win_Diff", "Loss_Diff", "OT_Diff"])
    df_vars_for_prediction.astype(float)
    #print(df_vars_for_prediction)
    variables =  df_vars_for_prediction.loc[str(specific_game_id)]
    variables = variables.to_numpy().reshape(1,-1)
    #print(variables)
    #print(type(variables))
    return(RF_from_joblib.predict(variables))

#This is what 'varibles' looks like
#[[1 2 10 0.0 0.0 0.0 0.562 6 4 1 7 0 0 7 0.0 0.0 1.0 0.43799999999999994
#  7 5 1 9 1 3 -1 2 0.0 -1 0 -2 -3]]
#<class 'numpy.ndarray'>

#This is what df_vars_for_prediction looks like:
# Home_goals Home_pim Home_shots  Home_powerPlayPercentage  \
#2019020756          1        2         10                       0.0   

#            Home_powerPlayGoals  Home_powerPlayOpportunities  \
#2019020756                  0.0                          0.0   

#            Home_faceOffWinPercentage Home_blocked Home_takeaways  \
#2019020756                      0.562            6              4   

#           Home_giveaways  ... Away_hits Goals_Diff Shots_Diff Blocked_Diff  \
#2019020756              1  ...         9          1          3           -1   

#            PIM_Diff  PowerPlayGoals_Diff  Takeaways_Diff  Giveaways_Diff  \
#2019020756         2                  0.0              -1               0   

#           Hits_Diff Points_Diff  
#2019020756        -2          -3  

#[1 rows x 31 columns]


# In[25]:


#on 1/15/20 try this
def calculate_stats(df, games_id_sublist, game_counter, per = 1, state = "END"):
    Start_time_counter = 0
    #game_counter = 0
    #games_id_list = [2019020692]
    for game_id in games_id_sublist:
        Period = 0 
        while Period < 1:
            game_link = f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live"
            data = API_reader(game_link)
            if data["liveData"]["linescore"]["currentPeriod"] == per and data["liveData"]["linescore"]["currentPeriodTimeRemaining"] == state: 
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
                #df.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First period Prediction\\{todays_date}_raw.csv", index = True)
                print("Game ", str(game_counter) ,"/", str(len(games_id_list)), f"ID: {game_id} ({away_team[0]}@{home_team[0]}) completed at: ", str(datetime.today().hour), ":", str(datetime.today().minute))
                #print(df)
                prediction = prepare_vars_for_prediction(df,game_id)
                if int(prediction) == 0:
                    print(f"The team that is predicted to win is: {home_team[0]}")
                else:
                    print(f"The team that is predicted to win is: {away_team[0]}")
                Start_time_counter += 1
            else:
                Period = 0
                print("Check Point: ", datetime.today().hour, ":", datetime.today().minute)
                time.sleep(120)
    return(df)


# In[11]:


time.sleep(60*40*6)
df = pd.DataFrame(columns = [i for i in range(30)])
new_order = order_same_start_times(list_of_groups[0])
print(new_order)
calculate_stats(df, new_order, 0)


# In[34]:


new_order = order_same_start_times(list_of_groups[1])
print(new_order)
#new_order = [2019020744, 2019020745]
calculate_stats(df, new_order, 1) #whatever game_counter is, the result will be that plus 1


# In[36]:


new_order = order_same_start_times(list_of_groups[2])
print(new_order)
calculate_stats(df, new_order, 2)


# In[54]:


for i in range(2,len(list_of_groups)):
    new_order = order_same_start_times(list_of_groups[i])
    print(new_order)
    calculate_stats(df, new_order, i)
    #print("Now sleeping for 50 minutes")
    #time.sleep(60*50) #Need a way of avoiding this after the last game.


# In[48]:


calculate_stats(df, new_order, 3)


# In[37]:


df_all_features = differences(df)
print(df_all_features)
df_all_features.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First period Prediction\\{todays_date}_df_all_features.csv", index = True)


# # Report the Winner

# In[38]:


#time.sleep(60*90)
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

# In[39]:


df_all_features_copy = df_all_features.copy()
df_all_features = df_all_features.drop(columns = ["Home_team", "Away_team", "Winner"])
df_all_features.astype(float)
df_all_features.to_csv(f"C:\\Users\\David\\OneDrive\\Documents\\OneDrive\\NHL API First period Prediction\\{todays_date}_df_all_features_winner.csv", index = True)


# In[88]:


#https://stackoverflow.com/questions/29373842/sorting-python-list-to-make-letters-come-before-numbers
testing = {2019: '13:39', 2020: '12:00', 2021: 'END', 2022: '01:10'}
testing1 = {k : v for k,v in sorted(testing.items(), key = lambda item: item[1])}
#testing2 = {k : v for k,v in sorted(testing.items(), key = lambda item: ([str,int].index(type(item)), item[1]))}
print(testing1)
#print(testing2)
#testing1.values() 
'END' in testing1.values()
list(testing1.values())


# In[ ]:


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


# In[ ]:


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
            print("Game ", str(game_counter) ,"/", str(len(games_id_list)), f"ID: {game_id} ({away_team[0]}@{home_team[0]}) completed at: ", str(datetime.today().hour), ":", str(datetime.today().minute))
            prediction = prepare_vars_for_prediction(game_id)   
            if int(prediction) == 0:
                print(f"The team that is predicted to win is: {home_team[0]}")
            else:
                print(f"The team that is predicted to win is: {away_team[0]}")
            if delta_seconds_start_times[Start_time_counter] != 0:
                print("Now sleeping for:", str(delta_seconds_start_times[Start_time_counter]/60/60), "hours.")
            time.sleep(delta_seconds_start_times[Start_time_counter])
            Start_time_counter += 1
        else:
            Period = 0
            print("Check Point: ", datetime.today().hour, ":", datetime.today().minute)
            time.sleep(180) 


# In[ ]:


#team_name = input("Enter your team: ")
team_name = 'BOS'
data = API_reader("https://statsapi.web.nhl.com/api/v1/teams")
team_dict = {}
for i in range(len(data["teams"])):
    team_dict[data['teams'][i]["abbreviation"]] = data['teams'][i]["id"]
team_id = team_dict[team_name]
data = API_reader(f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.schedule.next")
specific_game_id = data["teams"][0]["nextGameSchedule"]["dates"][0]["games"][0]["gamePk"]
print(specific_game_id)

