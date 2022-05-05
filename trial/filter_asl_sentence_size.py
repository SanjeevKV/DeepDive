import os
import gzip, pickle
import pandas as pd

def write_pickle_file(filename, dataset):
	out = gzip.compress(pickle.dumps(dataset))
	f = open(filename, 'wb')
	f.write(out)
	f.close()
    
def generate_new_embeddings(h2s_path, preprocessed_path):
    df = pd.read_csv(preprocessed_path)
    skipped_count = [0,0,0]
    new_count = [0,0,0]
    out_folder = '/scratch2/maiyaupp/how2sign/how2sign_vitb16/filter_sentence_size_30'
    for i,subset in enumerate(['train', 'dev', 'test']): 
        folder_path = os.path.join(h2s_path, 'how2sign_vitb16', subset)
        out_file = os.path.join(out_folder, subset)
        
        count = 0
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            gzip_file = gzip.open(file_path, 'rb')
            pickle_file = pickle.load(gzip_file)

            dataset = []

            for video in pickle_file:
                # string = video['text']
                string = df[df['name'] == video['name']]['text'].values[0]
                # print(string, type(string))
                if len(string.split(" ")) <= 30:              
                    dataset.append(video)
                    new_count[i] += 1
                else:
                    skipped_count[i] += 1
            
            if len(dataset) > 0:
                write_pickle_file(os.path.join(out_file, file), dataset)
    print("Skipped count for train, dev, test:", skipped_count, "\nNew count for train, dev, test:", new_count)


if __name__ == '__main__':
    h2s_path = '/scratch2/maiyaupp/how2sign'
    preprocessed_path = '/scratch2/maiyaupp/how2sign/text_preprocessed/combined.csv'
    generate_new_embeddings(h2s_path, preprocessed_path)