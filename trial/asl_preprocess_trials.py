import numpy as np
import os
import gzip
import pickle
# import argparse
import sys
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from img2vec import write_pickle_file
from time import time
import contractions

def preprocess_text(in_file, out_file, delimiter='\t'):
    def preprocess(text):
        text = contractions.fix(text.lower()).lower()
        text = ' '.join([x for x in word_tokenize(text) if x.isalnum()])
        return text
    print('Starting text preprocessing...')
    print('Reading data from', in_file)
    data = pd.read_csv(in_file, delimiter=delimiter)
    data['name'] = data['SENTENCE_NAME']
    data['text'] = data['SENTENCE'].apply(preprocess)
    data = data[ ['name', 'text'] ]
    # data.to_csv(out_file, index=False)
    print('Text preprocess completed')
    return data

def filter_data(in_path, out_path, batch_size, keep_set, data=None):
    def drop_cross(other, text): #for test, dev datasets
        words = [x for x in word_tokenize(text.lower()) if x.lower() not in stopwords.words('english') and x.replace("'", '').isalnum()]
        for word in words:
            if word not in keep_set:
                return True
        return False
    
    def drop_same(name, other): #for train dataset
        ans = data[data['SENTENCE_NAME']==name]['DROPPED'].item()
        return ans
    
    drop = drop_same if type(data)!=type(None) else drop_cross

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print('Starting filter process...')
    print('Reading data from', in_path)
    in_files = sorted(os.listdir(in_path))
    batch = []
    ind = 0
    name = '.'.join(in_files[0].split('.')[:-1])
    for fi in in_files:
        with gzip.open(os.path.join(in_path, fi)) as f:
            fi = pickle.load(f)
        for d in fi:
            if not drop(d['name'], d['text']):
                batch.append(d)
                ind += 1
            if len(batch)==batch_size:
                print('Writing batch', (ind-1))
                towri = os.path.join(out_path, name+'.'+str(ind-1).zfill(6))
                write_pickle_file(towri, batch)
                batch = []
    if len(batch)>0:
        print('Writing batch', (ind-1))
        towri = os.path.join(out_path, name+'.'+str(ind-1).zfill(6))
        write_pickle_file(towri, batch)
    print('Filter Process Completed')

def get_all_ids(path):
    print('Reading data from', path)
    in_files = sorted(os.listdir(path))
    ids = set()
    for fi in in_files:
        with gzip.open(os.path.join(path, fi)) as f:
            fi = pickle.load(f)
        for d in fi:
            ids.add(d['name'])
    print('No of IDs found', len(ids))
    return ids

def get_vocab(h2s_path, dataset, data = None):
    if type(data)==type(None):
        data = pd.read_csv(os.path.join(h2s_path, dataset+'.csv'), delimiter='\t')
    data['WORDS'] = data['SENTENCE'].apply(lambda y: [x for x in word_tokenize(y.lower()) if x.lower() not in stopwords.words('english') and x.replace("'", '').isalnum()])
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
        drop_set = {x for x in vocab if vocab[x]<=thresh}
        data['DROPPED'] = data['WORDS'].apply(can_drop)
        y.append(len(data[~data['DROPPED']]))
        x.append(thresh)
        # print('After dropping instances with words with freq <=',thresh,'dataset size:', len(data[~data['DROPPED']]), '/',len(data))
    plt.plot(x, y)
    plt.savefig('datalenVSthresh.png')

def add_drop_column(vocab, data, thresh):
    def can_drop(x):
        for w in x:
            if w in drop_set:
                return True
        return False
    drop_set = {x for x in vocab if vocab[x]<=thresh}
    data['DROPPED'] = data['WORDS'].apply(can_drop)
    print('After dropping instances with words with freq <=',thresh,'dataset size:', len(data[~data['DROPPED']]), '/', len(data))

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
    # train, data = get_vocab(h2s_path, 'train')
    # plot_datalen_vs_thresh(train, data)
    # total = set()
    # total.update(dev.keys())
    # total.update(test.keys())
    # total.update(train.keys())
    # print('For total ::',len(total))
    # print('For dev ::', plot_vocab_counts(dev, 'dev.png'))
    # print('For test ::', plot_vocab_counts(test, 'test.png'))
    # train = {x:train[x] for x in train if train[x]>200}

    thresh = 40
    input_main_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'
    print('START :: FILTERING BASED ON TRAIN VOCABULARY')
    output_folder = 'filtered_thresh40'

    train, data = get_vocab(h2s_path, 'train')
    add_drop_column(train, data, thresh)
    batch_size = 100
    keep_set = {x for x in train if train[x]>thresh}
    print('Keeping data with word frequency above', thresh, 'and using batch size', batch_size)
    in_path = input_main_path + 'train'
    out_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'+output_folder+'/train'
    filter_data(in_path, out_path, batch_size, keep_set, data=data)
    in_path = input_main_path + 'test'
    out_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'+output_folder+'/test'
    filter_data(in_path, out_path, batch_size, keep_set)
    in_path = input_main_path + 'dev'
    out_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'+output_folder+'/dev'
    filter_data(in_path, out_path, batch_size, keep_set)

    print('START :: FILTERING BASED ON COMBINED (TRAIN+DEV+TEST) VOCABULARY')
    output_folder = 'filtered_mix_thresh40'

    tr = pd.read_csv(os.path.join(h2s_path, 'train.csv'), delimiter='\t')
    tr['DATASET'] = 'TRAIN'
    te = pd.read_csv(os.path.join(h2s_path, 'test.csv'), delimiter='\t')
    te['DATASET'] = 'TEST'
    de = pd.read_csv(os.path.join(h2s_path, 'dev.csv'), delimiter='\t')
    de['DATASET'] = 'DEV'
    data = pd.concat([tr, te, de])
    vocab, data = get_vocab(h2s_path, 'combined', data=data)
    add_drop_column(vocab, data, thresh)
    batch_size = 100
    keep_set = {x for x in vocab if vocab[x]>thresh}
    print('Keeping data with word frequency above', thresh, 'and using batch size', batch_size)
    in_path = input_main_path + 'train'
    out_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'+output_folder+'/train'
    filter_data(in_path, out_path, batch_size, keep_set, data=data[data['DATASET']=='TRAIN'])
    in_path = input_main_path + 'test'
    out_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'+output_folder+'/test'
    filter_data(in_path, out_path, batch_size, keep_set, data=data[data['DATASET']=='TEST'])
    in_path = input_main_path + 'dev'
    out_path = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/'+output_folder+'/dev'
    filter_data(in_path, out_path, batch_size, keep_set, data=data[data['DATASET']=='DEV'])

    # ids = get_all_ids(out_path)
    # sent_ids = set(data[~data['DROPPED']]['SENTENCE_NAME'].unique())
    # f = open('stats.txt', 'w')
    # temp = sent_ids - ids
    # f.write('In csv, not in folder:\n\n')
    # f.write( '\n'.join(temp) )
    # f.close()
    # print('For train ::', plot_vocab_counts(train, 'train_nostp_200.png'))

def main_text_preprocess():
    out_file = '/scratch2/maiyaupp/how2sign/text_preprocessed/combined.csv'
    
    in_file = '/scratch2/maiyaupp/how2sign/dev.csv'
    dev = preprocess_text(in_file, out_file, delimiter='\t')
    
    in_file = '/scratch2/maiyaupp/how2sign/test.csv'
    test = preprocess_text(in_file, out_file, delimiter='\t')
    
    in_file = '/scratch2/maiyaupp/how2sign/train.csv'
    train = preprocess_text(in_file, out_file, delimiter='\t')

    combined = pd.concat([dev, test, train])
    combined.to_csv(out_file, index=False)

if __name__ == '__main__':
    start = time()
    # ap = argparse.ArgumentParser("DeepDive")
    # ap.add_argument("--datapath", help="Which dataset to run on")
    # args = ap.parse_args()
    # main()
    main_text_preprocess()
    print('Completed in:', (time() - start))
# For dev :: 3196
# For test :: 3662
# For train :: 15655
# For total :: 16539
# For train :: None
