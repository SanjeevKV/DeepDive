from alexnet_tf2 import AlexNet
from cv2 import imread, resize
import numpy as np
import tensorflow as tf
import os
import gzip
import pickle
import argparse
import sys

#Phoenix fields
name_field = 'name'
signer_field = 'speaker'
gloss_field = 'orth'
text_field = 'translation'
has_gloss = True

#device = "/gpu:0"
device = 'cpu'
keep_rate = 0.8

#Phoenix data files
#main_path = '/scratch2/ssnair/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/'
#video_path = main_path + 'features/fullFrame-210x260px/'
#out_file = '/scratch2/ssnair/data/PHOENIX-2014-T-DD/'

def preprocess(args):
	main_path = args.base_folder
	video_path = os.path.join(main_path, 'features', 'fullFrame-210x260px')
	out_file = args.out_folder
	if not os.path.exists(out_file):
		os.mkdir(out_file)
	out_file = os.path.join(out_file, 'phoenix14t.dd')

	for subset in ['dev', 'train', 'test']:
		anno_path = os.path.join('annotations', 'manual', 'PHOENIX-2014-T.' + subset + '.corpus.csv')#'annotations/manual/PHOENIX-2014-T.' + subset + '.corpus.csv'

		f = open(os.path.join(main_path, anno_path))
		anno = f.read()
		f.close()

		anno = anno.split('\n')
		ind = { x[1]:x[0] for x in enumerate(anno[0].split('|')) }
		#print(ind)
		#sys.exit()
		anno = anno[1:]
		dataset = []

		for i, csv_line in enumerate(anno):
			print(f'Running subset: {subset}')
			print(f'Current file number: {i}')
			line = csv_line.split('|')
			#print(line)
			#sys.exit()
			res = dict()
			res[name_field] = line[ ind[name_field] ]
			res['signer'] = line[ ind[signer_field] ]
			res['gloss'] = line[ ind[gloss_field] ] if has_gloss else ''
			res['text'] = line[ ind[text_field] ]
			curr_path = os.path.join(video_path, subset, res['name'])
			files = sorted([x for x in os.listdir(curr_path) if '.png' in x ])
			im_vs = []
			for fil in files:
				filepath = os.path.join(curr_path, fil)
				img = imread(filepath)
				img = resize(img, (227, 227))
				img = img.reshape(1, *img.shape)
				img = img.astype(np.float32)
				img = tf.convert_to_tensor(img)
				cnn = AlexNet(img, keep_rate, device)
				out = cnn.output
				im_vs.append(tf.reshape(out, out.shape[1:]))
				#print(out.shape)
			res['sign'] = tf.concat(im_vs, axis=0)
			print(res['sign'].shape)
			#sys.exit()
			dataset.append(res)
		out = gzip.compress(pickle.dumps(dataset))

		f = open(out_file+'.'+subset, 'wb')
		f.write(out)
		f.close()

def main():
	 ap = argparse.ArgumentParser("Joey NMT")
	 ap.add_argument("base_folder", help="Base folder for all the data")
	 ap.add_argument("out_folder", help="Base folder to write the output features")
	 args = ap.parse_args()
	 preprocess(args)

if __name__ == "__main__":
    main()