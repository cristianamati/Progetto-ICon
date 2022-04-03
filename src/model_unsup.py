from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

#elbow method per modello kmeans
def elbow(data):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)

    print("\nDo you want to the elbow curve? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        plt.figure(figsize=(16, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
    else:
        print("OK!")


#clusterizzazione secondo diverse proprietà
def clusters(data):
    kmean_data = data.copy()
    i = 1
    kmean_clusters = [["culmen_length_mm", "culmen_depth_mm"], ["culmen_length_mm", "flipper_length_mm"],
                      ["culmen_length_mm", "body_mass_g"], ["culmen_depth_mm", "flipper_length_mm"],
                      ["culmen_depth_mm", "body_mass_g"], ["flipper_length_mm", "body_mass_g"]]

    plt.figure()
    for cluster in kmean_clusters:
        X_kmean = kmean_data[cluster]
        kmeans = KMeans(n_clusters=3, random_state=14)
        kmeans.fit(X_kmean)
        y_kmeans = kmeans.predict(X_kmean)
        kmean_data[f"Cluster{i}"] = y_kmeans

        plt.subplot(2, 3, i)
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.title(f"{i}.) Clustering by: \n{cluster}")
        sns.scatterplot(X_kmean.loc[:, cluster[0]], X_kmean.loc[:, cluster[1]], c=y_kmeans, s=50, cmap="flare")
        centers = kmeans.cluster_centers_
        sns.scatterplot(centers[:, 0], centers[:, 1], s=200, color="k", alpha=0.5)
        i += 1
    plt.savefig("kmeans.png")

    print("\nVuoi mostrare la mappa di clusters per Kmean? Y/N")
    answer = input()
    if answer == 'Y' or answer == 'y' or answer == 'yes' or answer == 'Yes':
        plt.show()
    else:
        print("OK!")

def cluster1(data):
    print('\nCLUSTERING SU LUNGHEZZA BECCO E PROFONDITA BECCO')
    X = data[['culmen_length_mm', 'culmen_depth_mm']]
    kmeans1 = KMeans(n_clusters=3)
    kmeans1.fit(X)

    labels = kmeans1.predict(X)
    matrix = pd.DataFrame({'labels': labels, 'specie': y})
    ct = pd.crosstab(matrix['labels'], matrix['specie'])
    print(ct)

def cluster2(data):
    print('\nCLUSTERING SU LUNGHEZZA PINNA E PROFONDITA BECCO')
    X = data[['flipper_length_mm', 'culmen_depth_mm']]
    kmeans2 = KMeans(n_clusters=3)
    kmeans2.fit(X)

    labels = kmeans2.predict(X)
    matrix = pd.DataFrame({'labels': labels, 'specie': y})
    ct = pd.crosstab(matrix['labels'], matrix['specie'])
    print(ct)


def cluster3(data):
    print('\nCLUSTERING SU LUNGHEZZA PINNA E MASSA')
    X = data[['flipper_length_mm', 'body_mass_g']]
    kmeans3 = KMeans(n_clusters=3)
    kmeans3.fit(X)

    labels = kmeans3.predict(X)
    matrix = pd.DataFrame({'labels': labels, 'specie': y})
    ct = pd.crosstab(matrix['labels'], matrix['specie'])
    print(ct)

def cluster4(data):
    print('\nCLUSTERING SU LUNGHEZZA PINNA E LUNGHEZZA BECCO')
    X = data[['flipper_length_mm', 'culmen_length_mm']]
    kmeans4 = KMeans(n_clusters=3)
    kmeans4.fit(X)

    labels = kmeans4.predict(X)
    matrix = pd.DataFrame({'labels': labels, 'specie': y})
    ct = pd.crosstab(matrix['labels'], matrix['specie'])
    print(ct)


if __name__ == '__main__':
    penguin = pd.read_csv('out.csv')
    y = penguin['species']
    penguin.drop('species', axis=1, inplace=True)
    #elbow(penguin)
    clusters(penguin)
    cluster1(penguin)
    cluster2(penguin)
    cluster3(penguin)
    cluster4(penguin)


