# Collaborative Filtering
# Jiachen Li (jiachenl)

import sys
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, pairwise_distances
from sklearn.preprocessing import normalize
import numpy as np
import random
import heapq
import time
import math

# README
# command line
# python CF.py <mode:0-5> <training set> <test set> <output file>
# mode: 0-Basic Statistics, 1-User-user similarity 2-Movie-movie similarity
#       3-Movie-rating/user-rating normalization 4-Bipartite User-user similarity
#       5-Bipartite Movie-movie similarity
#
# Set the experiment parameters here
weight = 'weight' # mean or weight
metric = 'cosine' # cosine or dotproduct
K = 10

# Run bipartite clustering separately!!
run_bipartite_clustering = False
maxIter = 10      # maximum iteration times in K-means
nUCluster = 2000  # initial # of cluster for users
nMCluster = 1000   # initial # of cluster for movies

def main(args):
    
    mode = args[1]
    evalout_name = args[4]
    
    time_start = time.time()
    trainingset = load_csvdata(args[2])
    
    M_m2u = training_matrix(trainingset) # movie to user sparse matrix        
    M_u2m = M_m2u.T.tocsr()

    # testset sample format: (movieID, userID)
    testset = load_testdata(args[3])
    
    if run_bipartite_clustering:
        print 'using bipartite clustering...'
        # bipartite clustring
        UCluster, MCluster = bipartite_clustering(M_u2m, M_m2u, nUCluster, nMCluster)
        Mc_u2m = get_centroids(M_u2m, UCluster).todense().tolist()
        write_matrix(Mc_u2m, 'Mc_u2m.matrix')
        Mc_m2u = get_centroids(M_m2u, MCluster).todense().tolist()
        write_matrix(Mc_m2u, 'Mc_m2u.matrix')
        print time.time() - time_start
        return
        
    if mode == '0':
        get_stats(trainingset, M_u2m, M_m2u)
        return
    elif mode == '1':
        # task 1 user-user similarity
        if metric == 'dotproduct':
            A = M_u2m.dot(M_m2u).todense().tolist()
        elif metric == 'cosine':
            Mn_u2m = normalize(M_u2m)
            A = Mn_u2m.dot(Mn_u2m.T.tocsr()).todense().tolist()
        ratings = u2u_recommend(testset, A, M_u2m.todense(), k=K, weight=weight)
    elif mode == '2':
        # task 2 movie-movie similarity A = X^T*X
        if metric == 'dotproduct':
            A = M_m2u.dot(M_u2m).todense().tolist()
        elif metric == 'cosine':
            Mn_m2u = normalize(M_m2u)
            A = Mn_m2u.dot(Mn_m2u.T.tocsr()).todense().tolist()
        ratings = m2m_recommend(testset, A, M_m2u.todense(), k=K, weight=weight)
    elif mode == '3':
        # standardization for user rating of a movie
        Ms_m2u, mean_std = standardization(M_m2u)
        if metric == 'dotproduct':
            A = Ms_m2u.dot(Ms_m2u.T).tolist()
        elif metric == 'cosine':
            A = Ms_m2u.dot(Ms_m2u.T).tolist()
        ratings = m2m_recommend_std(testset, A, Ms_m2u, mean_std, k=K, weight=weight)
    elif mode == '4':
        # bipartite mode - user 2 user recommend
        #Mc_m2u = load_matrix('Mc_m2u.matrix')
        Mc_u2m = load_matrix('Mc_u2m.matrix')
        if metric == 'dotproduct':
            A = M_u2m.todense().dot(Mc_u2m.T).tolist()
        elif metric == 'cosine':
            A = normalize(M_u2m).todense().dot(normalize(Mc_u2m).T).tolist()
        ratings = u2u_recommend(testset, A, Mc_u2m, k=K, weight=weight)
    elif mode == '5':
        # bipartite mode - movie 2 movie recommend
        Mc_m2u = load_matrix('Mc_m2u.matrix')
        #Mc_u2m = load_matrix('Mc_u2m.matrix')
        if metric == 'dotproduct':
            A = M_m2u.todense().dot(Mc_m2u.T).tolist()
        elif metric == 'cosine':
            A = normalize(M_m2u).todense().dot(normalize(Mc_m2u).T).tolist()
        ratings = m2m_recommend(testset, A, Mc_m2u, k=K, weight=weight)   

    #print ratings
    print time.time() - time_start

    write_eval_result(testset, ratings, evalout_name)

def write_matrix(M, filename):
    with open(filename, 'w') as f:
        for i in xrange(len(M)):
            for j in xrange(len(M[i])):
                f.write(str(M[i][j]) + ' ')
            f.write('\n')
            
def load_matrix(filename):
    M = []
    with open(filename) as f:
        for line in map(lambda l:l.strip().split(' '), f.readlines()):
            row = []
            for val in line:
                row.append(float(val))
            M.append(row)
    return np.matrix(M)

def standardization(M):
    # M is csr_sparse matrix
    mean = np.zeros(M.shape[0])
    std = np.zeros(M.shape[0])
    for i in xrange(M.shape[0]):
        if M[i].size:
            mean[i] = np.mean(M[i].data)
            std[i] = math.sqrt(np.var(M[i].data))
            M.data[M.indptr[i]:M.indptr[i+1]] -= mean[i]
            if std[i] > 0:
                M.data[M.indptr[i]:M.indptr[i+1]] /= std[i]
    mean_std = []
    for idx in xrange(M.shape[0]):
        mean_std.append((mean[idx], std[idx]))

    return M.todense(), mean_std

    
def bipartite_clustering(X2Y, Y2X, nXCluster, nYCluster):
    # number if clusters might be reduced
    XCluster = None
    YCluster = do_kmeans(Y2X, nYCluster)
    nYCluster = np.amax(YCluster) + 1
    iter = 0
    for iter in xrange(20):
        iter += 1
        print 'Bipartite clustering iter:', iter
        W = 1 - pairwise_distances(Y2X, get_centroids(Y2X, YCluster), metric='cosine')
        X2YC = X2Y.dot(W)
        
        if XCluster is None:
            XCluster = do_kmeans(X2Y, nXCluster)
        else:
            XCluster = do_kmeans(X2YC, nXCluster, get_centroids(X2YC, XCluster))
        nXCluster = np.amax(XCluster) + 1
        
        W = 1- pairwise_distances(X2Y, get_centroids(X2Y, XCluster), metric='cosine')
        Y2XC = Y2X.dot(W)
        YCluster = do_kmeans(Y2XC, nYCluster, get_centroids(Y2XC, YCluster))
        nYCluster = np.amax(YCluster) + 1
    
    return XCluster, YCluster
    
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
        DIS = pairwise_distances(X, C, metric='cosine')
        G = DIS.argmin(axis=1) # G for group
        G = assign_clusters(np.asarray(G).reshape(-1))
        C = get_centroids(X, G)
        # stopping criteria
        if nIter > maxIter or ((G_old is not None) and (np.subtract(G, G_old).sum() == 0)):
            print '.',
            return G
        G_old = G
        nIter += 1

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

def write_eval_result(testset, ratings, filename):
    with open(filename, 'w') as f:
        for sample in testset:
            f.write(str(ratings[sample] + 3.0) + '\n')

def m2m_recommend_std(testset, A, Ms_m2u, mean_std, k, weight):
    Matrix_m2u = Ms_m2u.tolist()
    ratings = {}
    # group users to each movies, mid = [uid1, uid2, ...]
    movie_map = {}
    for (mid, uid) in testset:
        if mid not in movie_map:
            movie_map[mid] = []
        movie_map[mid].append(uid)
    n = 0
    # find k nearest neighbors for movie
    for mid, uids in movie_map.iteritems():
        # (mid, metrics)
        kNN = get_kNN_for_movie(mid, A, k)
        for uid in uids:
            n += 1
            print n
            l_rate = []
            for idx in xrange(len(kNN)):
                nn_rating = Matrix_m2u[kNN[idx][0]][uid]
                l_rate.append((nn_rating, kNN[idx][1]))
            
            # combine the score
            weighted_rate = 0.0
            if weight == 'weight':
                norm = 0.0
                for rate in l_rate:
                    weighted_rate += rate[0] * rate[1]
                    norm += abs(rate[1])
                if norm > 0:
                    weighted_rate /= norm
            else: # mean
                for rate in l_rate:
                    weighted_rate += rate[0]
                weighted_rate /= len(l_rate)
            ratings[(mid, uid)] = weighted_rate * mean_std[mid][1] + mean_std[mid][0]
    return ratings

            
def m2m_recommend(testset, A, M_m2u, k, weight):
    Matrix_m2u = M_m2u.tolist()
    ratings = {}
    # group users to each movies, mid = [uid1, uid2, ...]
    movie_map = {}
    for (mid, uid) in testset:
        if mid not in movie_map:
            movie_map[mid] = []
        movie_map[mid].append(uid)
    n = 0
    # find k nearest neighbors for movie
    for mid, uids in movie_map.iteritems():
        # (mid, metrics)
        kNN = get_kNN_for_movie(mid, A, k)
        for uid in uids:
            n += 1
            print n
            l_rate = []
            for idx in xrange(len(kNN)):
                nn_rating = Matrix_m2u[kNN[idx][0]][uid]
                l_rate.append((nn_rating, kNN[idx][1]))
            
            # combine the score
            weighted_rate = 0.0
            if weight == 'weight':
                norm = 0.0
                for rate in l_rate:
                    weighted_rate += rate[0] * rate[1]
                    norm += abs(rate[1])
                if norm > 0:
                    weighted_rate /= norm
            else: # mean
                for rate in l_rate:
                    weighted_rate += rate[0]
                weighted_rate /= len(l_rate)
            ratings[(mid, uid)] = weighted_rate
    return ratings

def get_kNN_for_movie(mid, A, k):
    l_dis = list(enumerate(A[mid]))
    # sort the list by evaluation metrics
    kNN = heapq.nlargest(k+1, l_dis, key=lambda x:x[1])
    for idx in xrange(len(kNN)):
        if kNN[idx][0] == mid:
            del kNN[idx]
            break
    if len(kNN) > k:
        del kNN[-1]
    return kNN

def get_kNN_for_user(uid, A, k):
    l_dis = list(enumerate(A[uid]))
    # sort the list by evaluation metrics
    kNN = heapq.nlargest(k+1, l_dis, key=lambda x:x[1])
    for idx in xrange(len(kNN)):
        if kNN[idx][0] == uid:
            del kNN[idx]
            break
    if len(kNN) > k:
        del kNN[-1]
    return kNN
    
def u2u_recommend(testset, A, M_u2m, k, weight):
    Matrix_u2m = M_u2m.tolist()
    ratings = {}
    # group movies to each user, uid = [mid1, mid2, ...]
    user_map = {}
    for (mid, uid) in testset:
        if uid not in user_map:
            user_map[uid] = []
        user_map[uid].append(mid)

    # estimate rating using k-NN
    n = 0
    for uid, mids in user_map.iteritems():
        # (userID, metrics)
        kNN = get_kNN_for_user(uid, A, k)
        for mid in mids:
            n += 1
            print n
            l_rate = []
            for idx in xrange(len(kNN)):
                nn_rating = Matrix_u2m[kNN[idx][0]][mid]
                l_rate.append((nn_rating, kNN[idx][1]))
            
            # how to combine the score
            weighted_rate = 0.0
            if weight == 'weight':
                norm = 0.0
                for rate in l_rate:
                    weighted_rate += rate[0] * rate[1]
                    norm += rate[1]
                if norm > 0:
                    weighted_rate /= norm
            else:
                for rate in l_rate:
                    weighted_rate += rate[0]
                weighted_rate /= len(l_rate)
            ratings[(mid, uid)] = weighted_rate

    return ratings

def get_kNN_for_M(uid, M_u2m, k, metric):
    user = M_u2m.getrow(uid)
    l_dis = []
    if metric == 'cosine':
        dis = cosine_similarity(user, M_u2m)
    elif metric == 'dotproduct':
        dis = linear_kernel(user, M_u2m)
    l_dis = list(enumerate(dis[0]))
    # sort the list by evaluation metrics
    kNN = heapq.nlargest(k+1, l_dis, key=lambda x:x[1])
    for idx in xrange(len(kNN)):
        if kNN[idx][0] == uid:
            del kNN[idx]
            break
    return kNN


def heapSearch( bigArray, k ):
    heap = []
    # Note: below is for illustration. It can be replaced by 
    # heapq.nlargest( bigArray, k )
    for item in bigArray:
        # If we have not yet found k items, or the current item is larger than
        # the smallest item on the heap,
        if len(heap) < k or item > heap[0]:
            # If the heap is full, remove the smallest element on the heap.
            if len(heap) == k: heapq.heappop( heap )
            # add the current element as the new smallest.
            heapq.heappush( heap, item )
    return heap

def training_matrix(data):
    # build sparse matrix from the given data
    values = []
    row_ind = []
    col_ind = []

    for sample in data:
        row_ind.append(sample[0])
        col_ind.append(sample[1])
        values.append(sample[2]-3.0)

    n_movies = max(row_ind) + 1
    n_users = max(col_ind) + 1
    matrix = csr_matrix((values, (row_ind, col_ind)), dtype=float,
                        shape=(n_movies, n_users))
    return matrix


def get_stats(data, M_u2m, M_m2u):
    # find the nearest neighbors
    ids = get_kNN_for_M(4321, M_u2m, 5, 'dotproduct')
    print 'Top 5 NNs of user 4321 in terms of dot product similarity:', ids

    ids = get_kNN_for_M(4321, M_u2m, 5, 'cosine')
    print 'Top 5 NNs of user 4321 in terms of cosine similarity:', ids

    ids = get_kNN_for_M(3, M_m2u, 5, 'dotproduct')
    print 'Top 5 NNs of movie 3 in terms of dot product similarity:', ids

    ids = get_kNN_for_M(3, M_m2u, 5, 'cosine')
    print 'Top 5 NNs of movie 3 in terms of coisne similarity:', ids


    # "MovieID","UserID","Rating","RatingDate"
    unique_movie = set()
    unique_user = set()

    print 'For whole dataset:'
    n_rate_1 = 0
    n_rate_3 = 0
    n_rate_5 = 0
    sum_rate = 0
    for sample in data:
        unique_movie.add(sample[0])
        unique_user.add(sample[1])
        rate = sample[2]
        if rate == 1:
            n_rate_1 += 1
        elif rate == 3:
            n_rate_3 += 1
        elif rate == 5:
            n_rate_5 += 1
        sum_rate += rate

    print '# of movies:', max(unique_movie)+1
    print '# of users:', max(unique_user)+1
    print "the number of times any movie was rated '1':", n_rate_1
    print "the number of times any movie was rated '3':", n_rate_3
    print "the number of times any movie was rated '5':", n_rate_5
    print 'average ranking across all users and movies:', float(sum_rate) / len(data)

    print 'For user ID 4321'
    n_rated = 0
    n_rate_1 = 0
    n_rate_3 = 0
    n_rate_5 = 0
    sum_rate = 0
    for sample in data:
        if sample[1] != 4321:
            continue
        n_rated += 1
        rate = sample[2]
        if rate == 1:
            n_rate_1 += 1
        elif rate == 3:
            n_rate_3 += 1
        elif rate == 5:
            n_rate_5 += 1
        sum_rate += rate

    print 'the number of movies rated:', n_rated
    print "the number of times the user gave a '1' rating:", n_rate_1
    print "the number of times the user gave a '3' rating:", n_rate_3
    print "the number of times the user gave a '5' rating:", n_rate_5
    print 'the average movie rating for this user:', float(sum_rate) / n_rated

    print 'For movie ID 3'
    n_rated = 0
    n_rate_1 = 0
    n_rate_3 = 0
    n_rate_5 = 0
    sum_rate = 0
    for sample in data:
        if sample[0] != 3:
            continue
        n_rated += 1
        rate = sample[2]
        if rate == 1:
            n_rate_1 += 1
        elif rate == 3:
            n_rate_3 += 1
        elif rate == 5:
            n_rate_5 += 1
        sum_rate += rate

    print 'the number of users rating this moive:', n_rated
    print "the number of times the user gave a '1' rating:", n_rate_1
    print "the number of times the user gave a '3' rating:", n_rate_3
    print "the number of times the user gave a '5' rating:", n_rate_5
    print 'the average rating for this movie:', float(sum_rate) / n_rated


def load_csvdata(filename):
    data = []
    # csv format
    with open(filename) as f:
        for line in map(lambda l: l.strip().split(','), f.readlines()):
            data.append((int(line[0]), int(line[1]), int(line[2]), line[3]))
    return data

def load_testdata(filename):
    data = []
    with open(filename) as f:
        for line in map(lambda l: l.strip().split(','), f.readlines()):
            data.append((int(line[0]), int(line[1])))
    return data


if __name__ == '__main__':
    main(sys.argv)