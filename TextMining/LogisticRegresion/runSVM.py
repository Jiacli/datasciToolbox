# script to run SVM
# Jiachen Li (jiachenl)

# svm_learn [option] example_file model_file
# svm_classify [options] example_file model_file output_file

# very simple interface
# python runSVM.py <C>

import sys, os, time
import numpy as np

SVM_LEARN_PATH = './svm/svm_learn'
SVM_CLASSIFY_PATH = './svm/svm_classify'
TRAINING_FILE_PREFIX = './data/citeseer.train.ltc.svm-'
OUTDIR = './svm_model/'
MODEL_PREFIX = './svm_model/svm_model.'
SVM_RESULTS = './svm_results/svm_result.'
TEST_FILE_PATH = './data/citeseer.test.ltc.svm'

def main(args):
    # convert_training_data(args[1]) # only need to call at the first time
    start_time = time.time()
    train_model(args[1])
    end_time = time.time()
    classify(TEST_FILE_PATH)
    svm_eval()
    print 'Training time:', end_time - start_time

def svm_eval():
    # load true label
    y = load_label(TEST_FILE_PATH)
    #print y_true

    # load results
    results = []
    for i in xrange(17):
        idx = i + 1
        with open(SVM_RESULTS + str(idx)) as f:
            result = map(lambda x:float(x.strip()), f.readlines())
            results.append(result)
    SCORE = np.array(results)
    #print SCORE
    predict = SCORE.argmax(axis=0) + 1
    #print y_hat

    with open('results.txt', 'w') as f:
        for (y_hat, y_true) in zip(predict, y):
            f.write('{} {}\n'.format(y_hat, y_true))
    print os.system('./eval.out ./results.txt')
    count = (predict == y).sum()
    print 'simple accuracy:', float(count) / len(y)

def classify(testfile):
    for i in xrange(17):
        idx = i + 1
        cmd = SVM_CLASSIFY_PATH + ' ' + testfile + ' ' + MODEL_PREFIX \
             + str(idx) + ' ' + SVM_RESULTS + str(idx)
        #print cmd
        os.system(cmd)


def train_model(c=0):
    for i in xrange(17):
        idx = i + 1
        cmd = SVM_LEARN_PATH + ' -c ' + c + ' ' + TRAINING_FILE_PREFIX \
             + str(idx) + ' ' + OUTDIR + 'svm_model.' + str(idx)
        #print cmd
        os.system(cmd)

def convert_training_data(filename):
    for i in xrange(17):
        idx = i + 1
        with open(filename) as f, open(filename+'-'+str(idx), 'w') as g:
            for line in map(lambda l:l.strip().split(' '), f.readlines()):
                if line[0] == str(idx):
                    line[0] = '1'
                else:
                    line[0] = '-1'
                g.write(' '.join(line) + '\n')


def load_label(filename):
    ndocs = 0
    labels = set()
    raw_data = []
    Y = []
    # load raw data, get labels and find the feature dimension
    with open(filename) as f:
        for line in map(lambda l:l.strip().split(' '), f.readlines()):
            label = int(line[0])
            Y.append(label)
    y = np.array(Y, dtype=int)
    return y


if __name__ == '__main__':
    main(sys.argv)