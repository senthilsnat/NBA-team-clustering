import numpy as np
import pandas as pd

with open("clustertags.csv", 'r') as myFile:
    dataLines = myFile.read().splitlines()

data_temp = []
for z in range(1, len(dataLines)):
    data_temp.append(dataLines[z].split(','))

tags = {}
for i in range(len(data_temp)):
    tags[str(data_temp[i][0])] = str(data_temp[i][1])


with open("adv_201617gamelogs.csv", 'r') as myFile:
    dataLines = myFile.read().splitlines()

data_temp = []
for z in range(1, len(dataLines)):
    data_temp.append(dataLines[z].split(','))

logs = []
for i in range(len(data_temp)):
    # team, opponent, pace, points against
    temp = [str(data_temp[i][0]), str(data_temp[i][4]), float(data_temp[i][10]), float(data_temp[i][7])]
    logs.append(temp)

print tags
print logs


team_dict = {}
for game in logs:
    if game[0] in team_dict:
        d_tag = tags[game[1]]
        team_dict[game[0]][d_tag][0] += game[2]
        team_dict[game[0]][d_tag][1] += game[3]
    else:
        # cat: [possessions, points]
        team_dict[game[0]] = {'1': [0, 0], '2': [0, 0], '3': [0, 0]}
        d_tag = tags[game[1]]
        team_dict[game[0]][d_tag][0] += game[2]
        team_dict[game[0]][d_tag][1] += game[3]

for team in team_dict:
    print team_dict[team]
    for style in team_dict[team]:
        team_dict[team][style].append(np.round((team_dict[team][style][1]/team_dict[team][style][0])*100, 2))

print team_dict


output_framer = []
for team in team_dict:
    total_drtg = ((team_dict[team]['1'][1] + team_dict[team]['2'][1] + team_dict[team]['3'][1]) /
                  (team_dict[team]['1'][0] + team_dict[team]['2'][0] + team_dict[team]['3'][0])) * 100
    total_pace = (team_dict[team]['1'][0] + team_dict[team]['2'][0] + team_dict[team]['3'][0])/82
    output_framer.append([team, team_dict[team]['1'][2], team_dict[team]['2'][2], team_dict[team]['3'][2],
                          total_drtg, total_pace])

print output_framer
output_df = pd.DataFrame(output_framer, columns=['Team', 'vs Cat 1', 'vs Cat 2', 'vs Cat 3', 'OVR DRTG', 'OVR PACE'])
output_df.to_csv("def_against_raw.csv")
