import matplotlib.pyplot as plt
import time

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import Isomap, TSNE
from keras.datasets import mnist,fashion_mnist
import seaborn as sns
import h5py
import numpy as np

def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

"""(_, _), (x_test, y_test) = mnist.load_data()
d = x_test.reshape((x_test.shape[0], -1))
y=np.array(y_test)
y = y.reshape((y.shape[0]))
d = np.divide(d, 255.)"""

"""(_, _), (x_test, y_test) = fashion_mnist.load_data()
d = x_test.reshape((x_test.shape[0], -1))
y=np.array(y_test)
y = y.reshape((y.shape[0]))
d = np.divide(d, 255.)"""

with h5py.File("datasets/usps.h5", 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]
    d = np.concatenate((X_tr, X_te), axis=0)
    y = np.concatenate((y_tr, y_te), axis=0)

#d=np.array(d)
print(np.shape(d))
#y=np.array(y)
print(np.shape(y))

###Only GMM####

time_start = time.time()
GMM_labels = GaussianMixture(n_components=10).fit_predict(d)
time_end = time.time()
acc = np.round(cluster_acc(y, GMM_labels),5)
print("GMM Accuracy: ",acc)
print("GMM NMI: ",normalized_mutual_info_score(y, GMM_labels))
print('GMM runtime: {} seconds'.format(time_end-time_start))

###PCA + GMM#####
pca = PCA(n_components=3)
embedding = pca.fit_transform(d)
time_start = time.time()
GMM_labels = GaussianMixture(n_components=10).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, GMM_labels),5)
print("PCA + GMM Accuracy: ",acc)
print("PCA + GMM NMI: ",normalized_mutual_info_score(y, GMM_labels))
print('PCA + GMM runtime: {} seconds'.format(time_end-time_start))

"""feat_cols = [ 'pixel'+str(i) for i in range(d.shape[1])]
df = pd.DataFrame(d,columns=feat_cols)
df['y'] = GMM_labels
df['label'] = df['y'].apply(lambda i: str(i))
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("PCA + GMM")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

###DAE +GMM######

time_start = time.time()

#d_DAE = pd.read_csv("datasets/mnist_ae_DEC.csv")
#y_DAE = pd.read_csv("datasets/mnist_label.csv")
#d_DAE = pd.read_csv("datasets/DAE_FMNIST10000.csv")
#d_DAE = pd.read_csv("datasets/DAE_FMNIST10000_tEpochs1000.csv")
d_DAE = pd.read_csv("datasets/usps_ae_DEC.csv")
y_DAE = pd.read_csv("datasets/USPS_y.csv")
#y_DAE = y[1:]
d_DAE=np.array(d_DAE)
y_DAE=np.array(y_DAE)
#d_DAE=d_DAE[1:,:]
y_DAE=y_DAE[:,0]
print(np.shape(d_DAE))
print(np.shape(y_DAE))
embedding = pca.fit_transform(d_DAE)
GMM_labels = GaussianMixture(n_components=10).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y_DAE, GMM_labels),5)
print("DAE + GMM Accuracy: ",acc)
print("DAE + GMM NMI: ",normalized_mutual_info_score(y_DAE, GMM_labels))
print('DAE + GMM runtime: {} seconds'.format(time_end-time_start))

"""embedding = pca.fit_transform(d_DAE)
feat_colss = [ 'pixel'+str(i) for i in range(embedding.shape[1])]
dff = pd.DataFrame(embedding,columns=feat_colss)
dff['y'] = GMM_labels
dff['embedding-one'] = embedding[:,0]
dff['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("DAE + GMM")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=dff,
    alpha=0.3
)
plt.show();"""

###CAE + GMM#####

time_start = time.time()
#d_CAE = pd.read_csv("datasets/mnist_CAE.csv")
#y_CAE = pd.read_csv("datasets/mnist_label.csv")
#d_CAE = pd.read_csv("datasets/fmnist10000_CAE.csv")
#d_CAE = pd.read_csv("datasets/fmnist10000_CAE_tEpochs1000.csv")
d_CAE = pd.read_csv("datasets/usps_ae_CAE.csv")
y_CAE = pd.read_csv("datasets/USPS_y.csv")
#y_CAE = y[1:]
d_CAE=np.array(d_CAE)
y_CAE=np.array(y_CAE)
y_CAE=y_CAE[:,0]
embedding = pca.fit_transform(d_CAE)
GMM_labels = GaussianMixture(n_components=10).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y_CAE, GMM_labels),5)
print("CAE + GMM Accuracy: ",acc)
print("CAE + GMM NMI: ",normalized_mutual_info_score(y_CAE, GMM_labels))
print('CAE + GMM runtime: {} seconds'.format(time_end-time_start))

"""embedding = pca.fit_transform(d_CAE)
dff['y'] = GMM_labels
dff['embedding-one'] = embedding[:,0]
dff['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("CAE + GMM")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=dff,
    alpha=0.3
)
plt.show();"""

###ISOMAP + GMM##
isomap = Isomap(n_components=3)
embedding = isomap.fit_transform(d)
time_start = time.time()
GMM_labels = GaussianMixture(n_components=10).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, GMM_labels),5)
print("ISOMAP + GMM Accuracy: ",acc)
print("ISOMAP + GMM NMI: ",normalized_mutual_info_score(y, GMM_labels))
print('ISOMAP + GMM runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = GMM_labels
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("ISOMAP + GMM")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

###t-SNE + GMM###
tsne = TSNE(n_components=3, perplexity=10, n_iter=300)
embedding = tsne.fit_transform(d)
time_start = time.time()

GMM_labels = GaussianMixture(n_components=10).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, GMM_labels),5)
print("t-SNE + GMM Accuracy: ",acc)
print("t-SNE + GMM NMI: ",normalized_mutual_info_score(y, GMM_labels))
print('t-SNE + GMM runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = GMM_labels
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("t-SNE + GMM")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

###UMAP + GMM####
embedding = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.0).fit_transform(d)
time_start = time.time()

GMM_labels = GaussianMixture(n_components=10).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, GMM_labels),5)
print("UMAP+GMM Accuracy: ",acc)
print("UMAP+GMM NMI: ",normalized_mutual_info_score(y, GMM_labels))
print('UMAP+GMM runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = GMM_labels
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("UMAP + GMM")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

#print(adjusted_rand_score(d.iloc[:4000, 784], kmeans_labels),adjusted_mutual_info_score(d.iloc[:4000, 784], kmeans_labels))
