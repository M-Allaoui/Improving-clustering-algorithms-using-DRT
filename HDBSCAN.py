import hdbscan
import matplotlib.pyplot as plt
import time

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
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

###Only HDBSCAN####
feat_cols = [ 'pixel'+str(i) for i in range(d.shape[1])]
df = pd.DataFrame(d,columns=feat_cols)
pca = PCA(n_components=0.99)

time_start = time.time()
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=90).fit_predict(d)
time_end = time.time()
acc = np.round(cluster_acc(y, HDBSCAN_labels),5)
print("HDBSCAN Accuracy: ",acc)
print("HDBSCAN NMI: ",normalized_mutual_info_score(y, HDBSCAN_labels))
print('HDBSCAN runtime: {} seconds'.format(time_end-time_start))


###PCA + HDBSCAN#####
embedding = pca.fit_transform(d)
time_start = time.time()
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=90).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, HDBSCAN_labels),5)
print("PCA + HDBSCAN Accuracy: ",acc)
print("PCA + HDBSCAN NMI: ",normalized_mutual_info_score(y, HDBSCAN_labels))
print('PCA + HDBSCAN runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = HDBSCAN_labels
df['label'] = df['y'].apply(lambda i: str(i))
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("PCA + HDBSCAN")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

###DAE +HDBSCAN######

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
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=90).fit_predict(d_DAE)
time_end = time.time()

acc = np.round(cluster_acc(y_DAE, HDBSCAN_labels),5)
print("DAE + HDBSCAN Accuracy: ",acc)
print("DAE + HDBSCAN NMI: ",normalized_mutual_info_score(y_DAE, HDBSCAN_labels))
print('DAE + HDBSCAN runtime: {} seconds'.format(time_end-time_start))

"""embedding = pca.fit_transform(d_DAE)
feat_colss = [ 'pixel'+str(i) for i in range(embedding.shape[1])]
dff = pd.DataFrame(embedding,columns=feat_colss)
dff['y'] = HDBSCAN_labels
dff['embedding-one'] = embedding[:,0]
dff['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("DAE + HDBSCAN")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=dff,
    alpha=0.3
)
plt.show();"""

###CAE + HDBSCAN#####

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
#d_CAE = d_CAE[:10000,:]
y_CAE = y_CAE[:,0]
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=90).fit_predict(d_CAE)
time_end = time.time()

acc = np.round(cluster_acc(y_CAE, HDBSCAN_labels),5)
print("CAE + HDBSCAN Accuracy: ",acc)
print("CAE + HDBSCAN NMI: ",normalized_mutual_info_score(y_CAE, HDBSCAN_labels))
print('CAE + HDBSCAN runtime: {} seconds'.format(time_end-time_start))

"""embedding = pca.fit_transform(d_CAE)
dff['y'] = HDBSCAN_labels
dff['embedding-one'] = embedding[:,0]
dff['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("CAE + HDBSCAN")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=dff,
    alpha=0.3
)
plt.show();"""

###ISOMAP + HDBSCAN##
isomap = Isomap(n_components=2)
embedding = isomap.fit_transform(d)
time_start = time.time()
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=80).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, HDBSCAN_labels),5)
print("ISOMAP + HDBSCAN Accuracy: ",acc)
print("ISOMAP + HDBSCAN NMI: ",normalized_mutual_info_score(y, HDBSCAN_labels))
print('ISOMAP + HDBSCAN runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = HDBSCAN_labels
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("ISOMAP + HDBSCAN")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

###t-SNE + HDBSCAN###
# fmnist TSNE(perplexity=30) HDBSCAN(min_samples=2, min_cluster_size=90) Accuracy:  0.3739 NMI:  0.5846
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
embedding = tsne.fit_transform(d)
time_start = time.time()
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=90).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, HDBSCAN_labels),5)
print("t-SNE + HDBSCAN Accuracy: ",acc)
print("t-SNE + HDBSCAN NMI: ",normalized_mutual_info_score(y, HDBSCAN_labels))
print('t-SNE + HDBSCAN runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = HDBSCAN_labels
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("t-SNE + HDBSCAN")
sns.scatterplot(
    x="embedding-one", y="embedding-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    legend="full",
    data=df,
    alpha=0.3
)
plt.show();"""

###UMAP + HDBSCAN####
# fmnist UMAP(n_neighbors=20) HDBSCAN(min_samples=10, min_cluster_size=120) Accuracy:  0.5219 NMI:  0.5744
embedding = umap.UMAP(n_neighbors=20, min_dist=0.0).fit_transform(d)
time_start = time.time()
HDBSCAN_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=120).fit_predict(embedding)
time_end = time.time()

acc = np.round(cluster_acc(y, HDBSCAN_labels),5)
print("UMAP + HDBSCAN Accuracy: ",acc)
print("UMAP + HDBSCAN NMI: ",normalized_mutual_info_score(y, HDBSCAN_labels))
print('UMAP + HDBSCAN runtime: {} seconds'.format(time_end-time_start))

"""df['y'] = HDBSCAN_labels
df['embedding-one'] = embedding[:,0]
df['embedding-two'] = embedding[:,1]
plt.figure(figsize=(10,6))
plt.title("UMAP + HDBSCAN")
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
