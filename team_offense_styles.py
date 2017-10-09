import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import statistics


# import and look at centers data
datafile = "team_offenses.csv"

with open("team_offenses.csv", 'r') as myFile:
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

# scale data
train = data
temp = np.array(data)[:, 0:-1]
scaler = preprocessing.StandardScaler().fit(temp[:, 0:-1])
centers = scaler.transform(temp[:, 0:-1]).tolist()
centers = np.array(centers)

print type(centers)
print type(temp)

pca = PCA(n_components=5)
pca.fit(temp)
print(pca.explained_variance_ratio_)

# excluding positional avg
a_temp = temp[:-1]
a_centers = centers[:-1]


# find optimal amount of kmeans clusters
def optimizer():
    scores = []
    for n in range(2, 11):
        ktest = KMeans(init='k-means++', n_clusters=n, n_init=10).fit(a_centers)
        labels = ktest.labels_
        score = metrics.silhouette_score(a_centers, labels, metric='euclidean')
        scores.append(score)

    opt = scores.index(min(scores)) + 2
    return opt

optimizer_mode_set = []
for n in range(100):
    optimizer_mode_set.append(optimizer())

opt_mode = statistics.mode(optimizer_mode_set)
print opt_mode

# use optimal amount of clusters to group data points
reduced_data = PCA(n_components=3).fit_transform(a_centers)

kmeans = KMeans(init='k-means++', n_clusters=opt_mode, n_init=10)
kmeans.fit(reduced_data)
clusters = kmeans.predict(reduced_data)
print clusters

classifications = []
for n in range(len(a_centers)):
    classifications.append([train[n][-1], clusters[n]])
print classifications

classes = {}
for n in range(len(a_centers)):
    if clusters[n] not in classes:
        classes[clusters[n]] = []
    classes[clusters[n]].append(train[n][-1])

print "k-means"
print classes

# Plot 2D visualization of clustering - code adapted from Scikit-Learn Examples

# Step size of the mesh
h = .02

# Plot the decision boundary
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

names = []
for n in train[:-1]:
    names.append(n[-1])
names = np.array(names)

# Plot the nodes using the coordinates of our embedding
plt.scatter(reduced_data.T[0], reduced_data.T[1], s=100, c=clusters,
            cmap=plt.cm.get_cmap('Accent'))

for index, (name, label, (x, y, z)) in enumerate(
        zip(names, clusters, reduced_data)):

    dx = x - reduced_data.T[0]
    dx[index] = 1
    dy = y - reduced_data.T[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x += .01
    else:
        horizontalalignment = 'right'
        x -= .01
    if this_dy > 0:
        verticalalignment = 'bottom'
        y += .01
    else:
        verticalalignment = 'top'
        y -= .01
    plt.text(x, y, name, size=8,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.get_cmap('Accent')(label / float(opt_mode)),
                       alpha=.7))

plt.title('K-means clustering of Offensive Styles (PCA-reduced data)')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
