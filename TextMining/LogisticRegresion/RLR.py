# Regularized Logistic Regression
# Jiachen Li (jiachenl)

import sys, os, time
import math
import numpy as np
from scipy.special import expit as sigmoid


maxIter = 52
learning_rate = 0.1
# const
nfeat = 14601

def main(train, test, C):
    global learning_rate
    X, y, labels = load_data(train)
    b = np.array([0.0] * len(labels))
    W = np.array([[1.0 / nfeat] * nfeat] * len(labels)) # initial weights 17 * nfeat

    # training model
    start_time = time.time()
    learning_rate = min(learning_rate, learning_rate * 10 / C)
    print 'actual lr:', learning_rate
    W, b = train_RLR(W, b, X, y, labels, learning_rate, C)
    end_time = time.time()
    # evaluation
    if test == None:
        return
    X_test, y_test, dummy = load_data(test)
    #print X_test.shape
    eval_RLR(W, b, X_test, y_test)
    print 'Training time:', end_time - start_time

def eval_RLR(W, b, X, y):
    print 'evaluate RLR...'
    Z = W.dot(X.T) + np.array([b] * len(y)).T
    SCORE = sigmoid(Z).T
    predict = SCORE.argmax(axis=1)

    with open('results.txt', 'w') as f:
        for (y_hat, y_true) in zip(predict, y):
            f.write('{} {}\n'.format(y_hat, y_true))

    print os.system('./eval.out ./results.txt')
    count = (predict == y).sum()
    print 'simple accuracy:', float(count) / len(y)

def train_RLR(W, b, X, y, labels, alpha, C):
    print 'trainig RLR...'
    last_ll = calc_avg_loglikelihood(W, b, X, y, labels)
    print 'initial ll:', last_ll
    for t in xrange(maxIter):
        alpha_t = alpha / math.log(t + 2)
        W, b = update_weights(W, b, X, y, labels, alpha_t, C)
        ll = calc_avg_loglikelihood(W, b, X, y, labels)
        print 'iter', t, 'll:', ll
        if ll > last_ll and (ll - last_ll) < 0.5:
            break # early stop
        else:
            last_ll = ll
        
    return W, b

def update_weights(W, b, X, y, labels, alpha, C):
    # compute the Z matrix
    Z = W.dot(X.T) + np.array([b] * len(y)).T
    # compute sigma(Z)
    SIGMA = sigmoid(Z)
    # update W and b w.r.t. each label
    for label in labels:
        t = np.array(y==label, dtype=float) - SIGMA[label,:]
        W[label,:] = (1 - alpha * C) * W[label,:] + alpha * t.dot(X)
        b[label] += alpha * t.sum() * 0.001
    return W, b

def calc_avg_loglikelihood(W, b, X, y, labels):
    ll = [0] * len(labels)
    Z = W.dot(X.T) + np.array([b] * len(y)).T
    SIGMA = sigmoid(Z)

    for label in labels:
        ll[label] = np.log(SIGMA[label, y==label]).sum() \
            + np.log(1-SIGMA[label, y!=label]).sum()
    return np.sum(ll) / len(ll)

def load_data(filename):
    ndocs = 0
    labels = set()
    raw_data = []
    Y = []
    # load raw data, get labels and find the feature dimension
    with open(filename) as f:
        for line in map(lambda l:l.strip().split(' '), f.readlines()):
            label = int(line[0])
            Y.append(label-1)
            labels.add(label-1)

            feats = []
            for i in xrange(1,len(line)):
                seg = line[i].split(':')
                idx = int(seg[0])
                feats.append((idx, float(seg[1])))
            raw_data.append(feats)
    ndocs = len(Y)
    # build training data matrix
    X = np.zeros((ndocs,nfeat), dtype=float)
    for i in xrange(len(raw_data)):
        for (idx, val) in raw_data[i]:
            X[i][idx-1] = val
    y = np.array(Y, dtype=int)

    return X, y, list(labels)

if __name__ == '__main__':
    train = sys.argv[1]
    test = sys.argv[2]
    C = float(sys.argv[3])

    main(train, test, C)