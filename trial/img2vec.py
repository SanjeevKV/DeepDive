from alexnet_tf2 import AlexNet
from cv2 import imread, resize
import numpy as np
import tensorflow as tf
import os
import gzip
import pickle

#Phoenix fields
name_field = 'name'
signer_field = 'speaker'
gloss_field = 'orth'
text_field = 'translation'
has_gloss = True

device = "/gpu:0"
#device = 'cpu'
keep_rate = 0.8

#Phoenix data files
main_path = '/scratch2/ssnair/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/'
video_path = main_path + 'features/fullFrame-210x260px/'
out_file = '/scratch2/ssnair/data/PHOENIX-2014-T-DD/'

if not os.path.exists(out_file):
	os.mkdir(out_file)
out_file += '/phoenix14t.dd'

for subset in ['dev', 'train', 'test']:
	anno_path = 'annotations/manual/PHOENIX-2014-T.' + subset + '.corpus.csv'

	f = open(main_path+anno_path)
	anno = f.read()
	f.close()

	anno = anno.split('\n')
	ind = { x[1]:x[0] for x in enumerate(anno[0].split('|')) }
	anno = anno[1:]
	dataset = []

	for line in anno:
		line = line.split('\t')
		res = dict()
		res['name'] = line[ ind[name_field] ]
		res['signer'] = line[ ind[signer_field] ]
		res['gloss'] = line[ ind[gloss_field] ] if has_gloss else ''
		res['text'] = line[ ind[text_field] ]
		curr_path = video_path+subset+'/'+res['name']
		files = sorted([x for x in os.listdir(curr_path) if '.png' in x ])
		im_vs = []
		for fil in files:
			filepath = curr_path + fil
			img = imread(filepath)
			img = resize(img, (227, 227))
			img = img.reshape(1, *img.shape)
			img = img.astype(np.float32)
			img = tf.convert_to_tensor(img)
			cnn = AlexNet(img, keep_rate, device)
			out = cnn.output
			im_vs.append(tf.reshape(out, out.shape[1:]))
		res['sign'] = tf.concat(im_vs, axis=0)
		dataset.append(res)
	out = gzip.compress(pickle.dumps(dataset))

	f = open(out_file+'.'+subset, 'wb')
	f.write(out)
	f.close()

