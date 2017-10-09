import numpy as np
import operator
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from radar import radar_graph

# import and look at centers data
with open("team_offenses_percentilever.csv", 'r') as myFile:
    dataLines = myFile.readlines()

data_temp = []
for z in range(1, len(dataLines)):
    data_temp.append(dataLines[z].split(','))
    # print data_temp[x-1]

data = []
for i in range(len(data_temp)):
    temp = []
    for j in range(1, len(data_temp[0])):
        if data_temp[i][j] == '':
            temp.append(0)
        else:
            temp.append(float(data_temp[i][j]))
    temp.append(str(data_temp[i][0]))

    data.append(temp)

# prepare data for feeding into radar chart
label = dataLines[0].split(',')
label.remove('Team')

print data
print label

for n in data:
    ind = data.index(n)
    graph_name = n[-1]
    legend_names = [n[-1]]
    case1 = n[:-1]
    case2 = n[:-1]
    radar_graph(graph_name, label, legend_names, case1, case2)

graph_name = "Comparing the 3 Offensive Styles"
legend_names = ['Inefficients', 'What\'s the point', 'Movers']
case1 = data[-1][:-1]
case2 = data[-2][:-1]
case3 = data[-3][:-1]
radar_graph(graph_name, label, legend_names, case1, case2, case3)

