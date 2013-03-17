from sys import argv, exit
import sys
import numpy as np
from os import path
from scipy.sparse import csr_matrix
from cPickle import dump

sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))

from LDA import TCVB0

#this demo is assumed to work with famous 20-newsgroups-dataset
#get this version http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz


def show_usage():
    print 'demo1.py path_to_dataset {alpha beta}'
    exit(1)

if len(argv) < 2:
    print 'Well, at least path to 20-newsgroups-dataset must be provided'
    show_usage()

input_path = argv[1]

if len(argv) < 4:
    print 'We need to know alpha and beta parameters for topic proportions \
        and topic smooting respectively'
    show_usage()

K = 20

alpha = np.array([float(argv[2])] * K)
beta = float(argv[3])


def yopen(filename):
    return open(path.join(input_path, filename), "r")


def read_labels(filename):
    with yopen(filename) as f:
        labels = f.read().rstrip().split('\n')
        return np.array(map(float, labels), dtype=int)

print 'reading labels...'
train_labels, test_labels = map(read_labels, ["train.label", "train.label"])


def read_data(filename):
    data = []
    rows = []
    cols = []
    with yopen(filename) as f:
        for line in f:
            i, j, x = map(float, line.split())
            data.append(x)
            rows.append(i - 1)
            cols.append(j - 1)
    return csr_matrix((data, (rows, cols)))

print 'reading data...'
train_data, test_data = map(read_data, ["train.data", "test.data"])

print 'training model...'
z = TCVB0(train_data[:1000], alpha, beta)

print 'model trained'
with open('LDA.dump', 'w') as output:
    dump(z, output)
