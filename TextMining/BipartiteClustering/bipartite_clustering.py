from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import numpy as np
import argparse
import random
import math
import time

# Assume this code is in the same folder as the document vector file
# baseline algorithm: python bipartite_clustering.py -f dev.docVectors
# custom algorithm : python bipartite_clustering.py --custom -f dev.docVectors -df dev.df

# global parameters
maxIter = 20      # maximum iteration times in K-means
nDCluster = 100   # initial # of cluster for documents
nWCluster = 200   # initial # of cluster for words
metric = 'cosine'

def main():
    # main routine for the bipartite_clustering
    args = parse_args()

    time_start = time.time()
    # read document vector from file
    with open(args.file) as f:
        doc_vec = map(lambda line : line.split(), f.readlines())
                
    # load data in sparse format
    row_ind=[]
    col_ind=[]
    data=[]
    for doc_id, doc in enumerate(doc_vec):
        for term in doc:
            word_id, tf = map(int, term.split(':'))
            row_ind.append(doc_id)
            col_ind.append(word_id)
            data.append(tf)
    total_docs = doc_id + 1
    total_words = max(col_ind) + 1
    
    # read document frequency df for each word (custom algorithm)
    if args.custom:
        idf = []
        with open(args.df) as f:
            for line in f.readlines():
                idx, val = line.strip().split(':')
                i_df = math.log(total_docs / float(val))
                idf.append(i_df)
        for idx in xrange(len(data)):
            data[idx] *= idf[col_ind[idx]]
        #print data

    M_d2w = csr_matrix((data, (row_ind, col_ind)), dtype=float,
                       shape=(total_docs, total_words))
    M_w2d = M_d2w.T.tocsr()
    
    # run the bipartite clustering algorithm
    DCluster, WCluster = bipartite_clustering(M_d2w, M_w2d, nDCluster, nWCluster)
    
    # write cluster info to the file for F1 score eval
    with open('doc_cluster.out', 'w') as f:
        for id, cluster in enumerate(DCluster):
            f.write('%i %i\n' % (id, cluster))
    
    with open('word_cluster.out', 'w') as f:
        for id, cluster in enumerate(WCluster):
            f.write('%i %i\n' % (id, cluster))

    # print out stats info
    print '\nIniti # of document cluster: ', nDCluster
    print 'Final # of document cluster: ', np.amax(DCluster) + 1
    print 'Initi # of word cluster: ', nWCluster
    print 'Final # of word cluster: ', np.amax(WCluster) + 1
    print 'Sum of cosine similarity for document cluster: ', calc_cosine_sum(M_d2w, DCluster)
    print 'Sum of cosine similarity for word cluster: ', calc_cosine_sum(M_w2d, WCluster)
    print 'Total running time: ', time.time() - time_start

def parse_args():
    parser = argparse.ArgumentParser(description='Bipartite Clustering')
    parser.add_argument('-f', '--file', type=str, metavar='FILENAME',
                        help='input file of document vectors', required=True)
    parser.add_argument('--custom', help='run with custom settings', action='store_true')
    parser.add_argument('-df', type=str, metavar='FILENAME', help='document frequency file')
    return parser.parse_args()

def assign_clusters(Cluster):
    # remove the empty cluster
    B = np.bincount(Cluster)
    valid = B > 0
    if valid.all():
        return Cluster
    B[valid] = 1
    B = np.cumsum(B) - 1
    return B[Cluster]

def get_centroids(X, XCluster):
    nCluster = np.amax(XCluster) + 1
    C = lil_matrix((nCluster, X.shape[1]))
    for i in xrange(nCluster):
        C[i] = X[XCluster == i].mean(axis=0)
    return csr_matrix(C)    

def do_kmeans(X, k, C=None):
    # Note: K-means clustering is implemented for row vectors of X
    
    # if initial centroids is None, then select randomly
    if C is None:
        random.seed()
        R = np.zeros(k, dtype=np.int)
        for i in xrange(k):
            R[i] = random.randint(0, X.shape[0]-1)
        C = X[R]

    nIter = 0
    G_old = None
    while True:
        DIS = pairwise_distances(X, C, metric=metric)
        G = DIS.argmin(axis=1) # G for group
        G = assign_clusters(np.asarray(G).reshape(-1))
        C = get_centroids(X, G)
        # stopping criteria
        if nIter > maxIter or ((G_old is not None) and (np.subtract(G, G_old).sum() == 0)):
            print '.',
            return G
        
        G_old = G
        nIter += 1

def bipartite_clustering(X2Y, Y2X, nXCluster, nYCluster):
    # number if clusters might be reduced
    XCluster = None
    YCluster = do_kmeans(Y2X, nYCluster)
    nYCluster = np.amax(YCluster) + 1
    
    for iter in xrange(20):
        W = 1 - pairwise_distances(Y2X, get_centroids(Y2X, YCluster), metric=metric)
        X2YC = X2Y.dot(W)
        
        if XCluster is None:
            XCluster = do_kmeans(X2Y, nXCluster)
        else:
            XCluster = do_kmeans(X2YC, nXCluster, get_centroids(X2YC, XCluster))
        nXCluster = np.amax(XCluster) + 1
        
        W = 1- pairwise_distances(X2Y, get_centroids(X2Y, XCluster), metric=metric)
        Y2XC = Y2X.dot(W)
        YCluster = do_kmeans(Y2XC, nYCluster, get_centroids(Y2XC, YCluster))
        nYCluster = np.amax(YCluster) + 1
    
    return XCluster, YCluster

def calc_cosine_sum(X, XCluster):
    C = get_centroids(X, XCluster)
    # corresponding center to each sample
    V = normalize(C)[XCluster]
    # sum of all cosine similarities
    return normalize(X).multiply(V).sum()
    
if __name__ == '__main__':
    main()