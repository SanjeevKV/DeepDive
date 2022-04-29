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
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import matplotlib.pyplot as plt
import gzip, pickle

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
    return vocab

def get_words_below_threshold(vocab, threshold = 1):
    drop_set = set()
    for word in vocab:
        if vocab[word] <= threshold:
            drop_set.add(word)
    return drop_set

def plot_vocab_counts(vocab, name):
    vocab = sorted(vocab.items(), key=lambda x: -x[-1])
    # vocab = vocab[:5]
    x, y = zip(*vocab)
    plt.plot(x, y)
    plt.xticks(rotation = 90)
    plt.savefig(name)

def get_pos_statistics(h2s_path, drop_set):
    f = open(os.path.join(h2s_path, 'train.csv'))
    anno = f.read().strip()
    f.close()
    anno = anno.split('\n')
    anno = anno[1:]

    num_rows = len(anno)
    pos_dict = dict()

    for i in range(num_rows):
        csv_line = anno[i].split('\t')
        s = csv_line[6]
        text = word_tokenize(s)
        pos_tags = nltk.pos_tag(text)
        for tag in pos_tags:
            if tag[0].lower() in drop_set:
                if tag[1] == 'DT':
                    print(tag[0])
                if tag[1] in pos_dict:
                    pos_dict[tag[1]] += 1
                else:
                    pos_dict[tag[1]] = 1

    return pos_dict

def write_pickle_file(filename, dataset):
	out = gzip.compress(pickle.dumps(dataset))
	f = open(filename, 'wb')
	f.write(out)
	f.close()

def generate_new_embeddings(h2s_path, drop_set):
    out_folder = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/pos_thresh2'
    for subset in ['dev', 'test', 'train']:
        folder_path = os.path.join(h2s_path, 'how2sign_vitb16', subset)
        out_file = os.path.join(out_folder, subset)
        
        count = 0
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            gzip_file = gzip.open(file_path, 'rb')
            pickle_file = pickle.load(gzip_file)

            dataset = []

            for video in pickle_file:
                string = video['text']
                tokenized_text = word_tokenize(string)
                pos_tags = nltk.pos_tag(tokenized_text)
                # new_string = ''
                new_tokens = []
                for tag in pos_tags:
                    if tag[0].lower() in drop_set:
                        # new_string += tag[1]
                        new_tokens.append(tag[1])
                    else:
                        # new_string += tag[0]
                        new_tokens.append(tag[0])
                
                new_string = TreebankWordDetokenizer().detokenize(new_tokens)
                video['text'] = new_string
                
                dataset.append(video)
            
            if len(dataset) > 0:
                write_pickle_file(os.path.join(out_file, file), dataset)
            

def main():
    h2s_path = '/scratch2/maiyaupp/how2sign'
    train = get_vocab(h2s_path, 'train')
    drop_set = get_words_below_threshold(train, 2)

    # pos_dict = get_pos_statistics(h2s_path, drop_set)

    generate_new_embeddings(h2s_path, drop_set)

if __name__ == '__main__':
    main()