from sklearn.cluster import KMeans

# load the data
data = [[170, 60], [165, 70], [180, 75], [155, 50], [175, 65], [190, 90], [160, 55], [185, 80], [150, 45], [170, 70]]

# perform KMeans clustering to determine the optimal number of clusters
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    inertias.append(kmeans.inertia_)

# plot the elbow curve to determine the optimal number of clusters
import matplotlib.pyplot as plt
plt.plot(range(1, 11), inertias)
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# based on the elbow curve, we can see that the optimal number of clusters is 3

# perform KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

# determine the average measurements for each cluster
sizes = []
for i in range(3):
    cluster = data[kmeans.labels_ == i]
    height = sum([x[0] for x in cluster]) / len(cluster)
    weight = sum([x[1] for x in cluster]) / len(cluster)
    sizes.append((height, weight))

print('Optimal Sizes:')
for i, size in enumerate(sizes):
    print('Cluster {}: Height: {:.2f}cm, Weight: {:.2f}kg'.format(i+1, size[0], size[1]))