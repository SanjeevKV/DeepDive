import numpy as np
import os
# import gzip
# import pickle
# import argparse
import sys
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

def get_vocab(h2s_path, dataset):
    data = pd.read_csv(os.path.join(h2s_path, dataset+'.csv'), delimiter='\t')
    data['WORDS'] = data['SENTENCE'].apply(lambda x: [x for x in word_tokenize(x.lower()) if x.lower() not in stopwords.words('english') and x.replace("'", '').isalnum()])
    # data['VOCAB'] = data['WORDS'].apply(set)
    sets = data['WORDS'].tolist()
    words = set()
    vocab = dict()
    for s in sets:
        words.update(s)
        for w in s:
            if w not in vocab.keys():
                vocab[w] = 0
            vocab[w] += 1
    print('For', dataset, '::', len(words))
    return vocab, data

def plot_datalen_vs_thresh(vocab, data):
    # thresh = 1
    def can_drop(x):
        for w in x:
            if w in drop_set:
                return True
        return False
    y,x = [len(data)], [0]
    for thresh in range(1, 11):
        drop_set = set()
        for word in vocab:
            if vocab[word]<=thresh:
                drop_set.add(word)
        data['DROPPED'] = data['WORDS'].apply(can_drop)
        y.append(len(data[~data['DROPPED']]))
        x.append(thresh)
        # print('After dropping instances with words with freq <=',thresh,'dataset size:', len(data[~data['DROPPED']]), '/',len(data))
    plt.plot(x, y)
    plt.savefig('datalenVSthresh.png')

def plot_vocab_counts(vocab, name):
    vocab = sorted(vocab.items(), key=lambda x: -x[-1])
    # vocab = vocab[:5]
    x, y = zip(*vocab)
    plt.plot(x, y)
    plt.xticks(rotation = 90)
    plt.savefig(name)

def main():
    h2s_path = '/scratch2/maiyaupp/how2sign'
    # dev = get_vocab(h2s_path, 'dev')
    # test = get_vocab(h2s_path, 'test')
    train, data = get_vocab(h2s_path, 'train')
    plot_datalen_vs_thresh(train, data)
    # total = set()
    # total.update(dev.keys())
    # total.update(test.keys())
    # total.update(train.keys())
    # print('For total ::',len(total))
    # print('For dev ::', plot_vocab_counts(dev, 'dev.png'))
    # print('For test ::', plot_vocab_counts(test, 'test.png'))
    train = {x:train[x] for x in train if train[x]>200}

    # print('For train ::', plot_vocab_counts(train, 'train_nostp_200.png'))

if __name__ == '__main__':
    # ap = argparse.ArgumentParser("DeepDive")
    # ap.add_argument("--datapath", help="Which dataset to run on")
    # args = ap.parse_args()
    main()
# For dev :: 3196
# For test :: 3662
# For train :: 15655
# For total :: 16539
# For train :: None