from cv2 import imread, resize
import numpy as np
import tensorflow as tf
import os
import gzip
import pickle
import argparse
import sys
import torch
import torchvision.models as models

# Dataset fields
datasets = {
	'Phoenix': {
		'name_field' : 'name',
		'signer_field' : 'speaker',
		'gloss_field' : 'orth',
		'text_field' : 'translation',
		'has_gloss' : True,
		'has_signer_info' : True,
		'delimiter' : '|'
	},
	'How2Sign': {
		'name_field' : 'SENTENCE_NAME',
		'signer_field' : 'speaker',
		'gloss_field' : 'orth',
		'text_field' : 'SENTENCE',
		'has_gloss' : False,
		'has_signer_info' : False,
		'delimiter' : '\t'
	}
}

#device = "/gpu:0"
device = 'cpu'
keep_rate = 0.8

def prepare_image(fp):
	img = imread(fp)
	img = resize(img, (227, 227))
	img = img.reshape(1, *img.shape)
	img = img.astype(np.float32)
	img = torch.tensor(img)
	img = torch.transpose(img, 1, 3)
	assert img.shape == (1, 3, 227, 227)
	return img

def preprocess(args, model, device):
	sign_dataset = args.dataset
	main_path = args.base_folder
	if sign_dataset == 'How2Sign':
		video_path = main_path
	elif sign_dataset == 'Phoenix':
		video_path = os.path.join(main_path, 'features', 'fullFrame-210x260px')
	
	out_file = args.out_folder
	if not os.path.exists(out_file):
		os.mkdir(out_file)
	out_file = os.path.join(out_file, sign_dataset + '.dd')

	for subset in ['dev', 'train', 'test']:
		if sign_dataset == 'How2Sign':
			anno_path = subset + '.csv'
		elif sign_dataset == 'Phoenix':
			anno_path = os.path.join('annotations', 'manual', 'PHOENIX-2014-T.' + subset + '.corpus.csv')

		f = open(os.path.join(main_path, anno_path))
		anno = f.read()
		f.close()

		anno = anno.split('\n')
		delimiter = datasets[sign_dataset]['delimiter']
		ind = { x[1]:x[0] for x in enumerate(anno[0].split(delimiter)) }
		anno = anno[1:]
		dataset = []

		for i, csv_line in enumerate(anno):
			print(f'Running subset: {subset}')
			print(f'Current file number: {i}')
			line = csv_line.split(delimiter)
			
			res = dict()
			
			fields = datasets[sign_dataset]
			res['name'] = line[ ind[ fields['name_field']] ]
			res['signer'] = line[ ind[ fields['signer_field'] ] ] if fields['has_signer_info'] else ''
			res['gloss'] = line[ ind[ fields['gloss_field'] ] ] if fields['has_gloss'] else ''
			res['text'] = line[ ind[ fields['text_field'] ] ]
			
			curr_path = os.path.join(video_path, subset + ('_images' if sign_dataset == 'How2Sign' else ''), res['name'])
			
			if not os.path.isdir(curr_path):
				continue
			try:
				files = sorted([x for x in os.listdir(curr_path) if '.png' in x ])
			except:
				continue
			im_vs = []
			
			for fil in files:
				print(fil)
				filepath = os.path.join(curr_path, fil)
				img = prepare_image(filepath)
				im_vs.append(img)
			
			img = torch.cat(im_vs, dim=0)
			img = img.to(device)
			res['sign'] = model(img)
			dataset.append(res)
			break
		
		out = gzip.compress(pickle.dumps(dataset))
		f = open(out_file+'.'+subset, 'wb')
		f.write(out)
		f.close()

def main():
	ap = argparse.ArgumentParser("Joey NMT")
	ap.add_argument("dataset", help="Which dataset to run on")
	ap.add_argument("base_folder", help="Base folder for all the data")
	ap.add_argument("out_folder", help="Base folder to write the output features")
	args = ap.parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	model = models.alexnet(pretrained=True)
	model.classifier = model.classifier[:5]
	model = model.to(device)
	preprocess(args, model, device)

if __name__ == "__main__":
    main()