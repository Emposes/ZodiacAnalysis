import pandas as pd

import numpy as np
from pathlib import Path

filename = Path(".src/horoscopes.xlsx")

df = pd.read_excel(filename)
df = df.dropna()

words = ["äôt","don","äì","äôs","Äôre", "äôt", "äôs", "Äì","Äôt", "Äôs" ]

for word in words:
     df['Horoscope'] = df['Horoscope'].str.replace(word, ' ')
     
     
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

cv = CountVectorizer(analyzer = 'word', max_features = 5000, lowercase=True, preprocessor=None, tokenizer=None, stop_words = 'english')  
vectors = cv.fit_transform(df.Horoscope)

Sum_of_squared_distances = []
K = range(2,12)
for k in K:
   km = KMeans(n_clusters=k,init='k-means++', max_iter=100, n_init=10, random_state = 42)
   km = km.fit(vectors)
   Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
    
kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
kmean_indices = kmeans.fit_predict(vectors)
totalDistance = np.min(kmean_indices, axis=0).sum()

df = df.dropna()
df_copy = df.assign(cluster=pd.Series(kmean_indices).values)

pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

fig = plt.figure(figsize=(4.5,4.5))
ax = fig.add_subplot()
x = np.array(x_axis)
y = np.array(y_axis)

scatter = ax.scatter(x,y, marker="o", c=kmean_indices, s=2, cmap="RdBu")
legend1 = ax.legend(*scatter.legend_elements(),
                     title="Clusters")
ax.add_artist(legend1)
plt.show()

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

for i in range (0,7):
    print("cluster",i)
    common_words = get_top_n_words(df_copy["Horoscope"].loc[df_copy["cluster"] == i], 10)
    for word, freq in common_words:
        print(word, freq)
    
from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(kmeans.cluster_centers_)

import numpy as np
tri_dists = dists[np.triu_indices(7, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(vectors,kmean_indices)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_)
df_feature = pd.DataFrame (feat_importances,columns=['feat_importance'])
df_feature['features'] = cv.get_feature_names()
df_feature.set_index('features', inplace= True)
df_feature['feat_importance'].nlargest(10).plot(kind='barh')
plt.show()

