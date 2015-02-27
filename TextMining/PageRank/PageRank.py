# Link Analysis - PageRank
# Author: Jiachen Li

import sys
import os
import math
import time
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

M = {} # transition matrix
stop_criteria = 0.0001 # stopping criteria for iteration

def main(args):
    mode = args[1]
    start_time = time.time()
    # load probabilistic transition matrix M
    n = load_transit_matrix('./transition.txt') # 1 - 81433
    if mode == '1': # Global PageRank (GPR)
        PR = get_GPR(n)
        eval('GPR', PR)
        with open('jiachenl-GPR-10.txt', 'w') as f:
            for idx in xrange(len(PR)):
                f.write(str(idx+1) + '-' + str(PR[idx]) + '\n')
    elif mode == '2': # Query-based Topic Sensitive PageRank (QTSPR)
        PR = get_QTSPR(n)
        eval('QTSPR', PR)
        with open('jiachenl-QTSPR-U2Q1-10.txt', 'w') as f:
            for idx in xrange(len(PR['2-1'])):
                f.write(str(idx+1) + '-' + str(PR['2-1'][idx]) + '\n')
    elif mode == '3': # Personalized Topic Sensitive PageRank (PTSPR)
        PR = get_PTSPR(n)
        eval('PTSPR', PR)
        with open('jiachenl-PTSPR-U2Q1-10.txt', 'w') as f:
            for idx in xrange(len(PR['2-1'])):
                f.write(str(idx+1) + '-' + str(PR['2-1'][idx]) + '\n')
    print('done! --- %s seconds ---' % str(time.time() - start_time))

def eval(mode_str, PR):
    print 'Evaluating...'
    # weight for Page Rank
    weight = -100.0
    filelist = os.listdir('./indri-lists')
    outNS = open(mode_str + '-NS-result.txt', 'w')
    outWS = open(mode_str + '-WS-result.txt', 'w')
    outCM = open(mode_str + '-CM-result.txt', 'w')
    for file in filelist:
        QryID = file.split('.')[0]
        ranklist = [] # [[docID, score]....]
        with open('./indri-lists/' + file) as f:
            for seg in map(lambda line:line.split(), f.readlines()):
                # Query-ID Q0 DocID Rank Score RunID
                ranklist.append([int(seg[2]), float(seg[4])])
        # re-rank with three different method
        NS = {}
        WS = {}
        CM = {}
        if mode_str == 'GPR':
            PR_list = PR
        else:
            PR_list = PR[QryID]
        # covert list to dict
        Rank = {}
        for i in xrange(len(PR_list)):
            Rank[i] = PR_list[i]
        for pair in ranklist:
            #NS[pair[0]] = Rank[pair[0]-1]
            WS[pair[0]] = pair[1] + weight * Rank[pair[0]-1]
            CM[pair[0]] = math.exp(pair[1]) - Rank[pair[0]-1] / 10
            # math.exp(pair[1]) - Rank[pair[0]-1] / 10
        # sort the dict by scores and write to results
        n = 1
        for (k,v) in sorted(Rank.iteritems(), key=lambda x:x[1], reverse=True):
            outNS.write(trec_eval_format(QryID, k, v, n))
            n += 1
            if n > len(ranklist):
                break
        n = 1
        for (k,v) in sorted(WS.iteritems(), key=lambda x:x[1], reverse=True):
            outWS.write(trec_eval_format(QryID, k, v, n))
            n += 1
        n = 1
        for (k,v) in sorted(CM.iteritems(), key=lambda x:x[1], reverse=True):
            outCM.write(trec_eval_format(QryID, k, v, n))
            n += 1
    outNS.close()
    outWS.close()
    outCM.close()

def trec_eval_format(QryID, docID, score, rank):
    return str(QryID) + ' Q0 ' + str(docID) + ' ' + str(rank) + ' ' + str(score) + ' run\n'

def get_PTSPR(n):
    ranks = get_TSPR(n)
    N_topic = len(ranks)
    print 'Calculating score of Peronalized Topic Sensitive Page Rank...'
    # read query topic distribution
    P_rank = {}
    with open('user-topic-distro.txt') as f:
        for line in map(lambda l : l.split(), f.readlines()):
            usr = line[0]
            qry = line[1]
            prob = [None] * N_topic
            for i in xrange(2, 14):
                k, v = line[i].split(':')
                prob[int(k)-1] = float(v)
            rank_p = np.array([0] * n, dtype=float)
            for j in xrange(N_topic):
                rank_p += prob[j] * ranks[j]
            P_rank[usr + '-' + qry] = rank_p    
    return P_rank
        
def get_QTSPR(n):
    ranks = get_TSPR(n)
    N_topic = len(ranks)
    print 'Calculating score of Query-based Topic Sensitive Page Rank...'
    # read query topic distribution
    Q_rank = {}
    with open('query-topic-distro.txt') as f:
        for line in map(lambda l : l.split(), f.readlines()):
            usr = line[0]
            qry = line[1]
            prob = [None] * N_topic
            for i in xrange(2, 14):
                k, v = line[i].split(':')
                prob[int(k)-1] = float(v)
            rank_q = np.array([0] * n, dtype=float)
            for j in xrange(N_topic):
                rank_q += prob[j] * ranks[j]
            Q_rank[usr + '-' + qry] = rank_q    
    return Q_rank

def get_TSPR(n):
    print 'Calculating Topic Sensitive Page Rank...'
    P_t, N_topic = load_doc_topics('./doc_topics.txt', n)
    # initial rank
    ranks = np.array([[1.0 / n] * n] * N_topic)
    
    n_ranks = update_rank_TSPR(ranks, P_t)
    l1_dis = 0
    for idx in xrange(len(ranks)):
        l1_dis = max(l1_dis, pairwise_distances(ranks[idx], n_ranks[idx], metric='l1')[0])
    # update rule
    while l1_dis > stop_criteria:
        ranks = n_ranks
        n_ranks = update_rank_TSPR(ranks, P_t)
        l1_dis = 0
        for idx in xrange(len(ranks)):
            l1_dis = max(l1_dis, pairwise_distances(ranks[idx], n_ranks[idx], metric='l1')[0])
        print 'max l1_dis:', l1_dis
    return ranks

def update_rank_TSPR(ranks, P_t):
    # parameters
    alpha = 0.5 # for M
    beta = 0.4  # for TS
    gamma = 1.0 - alpha - beta # for Teleportation
    N_topic = len(ranks)
    n = len(ranks[0])
    n_ranks = np.array([[1.0 / n] * n] * N_topic) * gamma + beta * np.array(P_t)
    # hash map based matrix-vector multiplication
    for key, val in M.iteritems():
        for idx in xrange(len(n_ranks)):
            n_ranks[idx][key[1]-1] += alpha * val * ranks[idx][key[0]-1]
    return n_ranks
    
def load_doc_topics(filename, n):
    topic = {}
    doc_topic = []
    with open(filename) as f:
        for p, t in map(lambda line : line.split(), f.readlines()):
            num_t = int(t)
            doc_topic.append((int(p), num_t))
            if num_t in topic:
                topic[num_t] += 1
            else:
                topic[num_t] = 1
    N_topic = len(topic)
    P_t = np.array([[0.0] * n] * N_topic) # row i is a topic vector for topic i-1
    
    for (d, t) in doc_topic:
        P_t[t-1][d-1] = 1.0 / topic[t]
    return P_t, N_topic
    
def get_GPR(n):
    print 'Calculating Global Page Rank...'
    # initial rank
    rank = np.array([1.0 / n] * n)
    n_rank = update_rank_GPR(rank)
    l1_dis = pairwise_distances(rank, n_rank, metric='l1')[0]
    # update rule
    while l1_dis > stop_criteria:
        rank = n_rank
        n_rank = update_rank_GPR(rank)
        l1_dis = pairwise_distances(rank, n_rank, metric='l1')[0]
        print 'l1_dis:', l1_dis
    return rank
    
def update_rank_GPR(rank):
    alpha = 0.15
    n = len(rank)
    n_rank = np.array([alpha / n] * n)
    # hash map based matrix-vector multiplication
    for key, val in M.iteritems():
        n_rank[key[1]-1] += (1-alpha) * val * rank[key[0]-1]
    return n_rank
    
def load_transit_matrix(filename):
    degree = {}
    n = 0
    with open(filename) as f:
        for i, j, k in map(lambda line : line.split(), f.readlines()):
            num_i = int(i)
            M[(num_i, int(j))] = int(k)
            if num_i in degree:
                degree[num_i] += 1
            else:
                degree[num_i] = 1
    for key in M.iterkeys():
        M[key] = 1.0 / degree[key[0]]
        n = max(n, key[0], key[1])
    return n

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python PageRank.py <mode: 1(GPR), 2(QTSPR), 3(PTSPR)>'
    main(sys.argv)