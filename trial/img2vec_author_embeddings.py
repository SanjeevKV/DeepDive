# from cv2 import imread, resize
from PIL import Image
import numpy as np
import os
import gzip
import pickle
import argparse
import sys
import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
import copy
import torch.optim as optim
import logging
import math
import random

logging.basicConfig(filename='convnext_with_dev_full_size_with_shuffle.log', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')

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

torch_preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def prepare_image(fp):
	# img = imread(fp)
	# img = resize(img, (227, 227))
	# img = img.reshape(1, *img.shape)
	# img = img.astype(np.float32)
	# img = torch.tensor(img)
	# img = torch.transpose(img, 1, 3)
	# img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
	img = Image.open(fp)
	img = torch_preprocess(img)
	img = img.unsqueeze(0)
	assert img.shape == (1, 3, 224, 224), 'Image shape: '+str(img.shape)
	return img

def prepare_video(files, curr_path):
	im_vs = []
	for fil in files:
		filepath = os.path.join(curr_path, fil)
		img = prepare_image(filepath)
		im_vs.append(img)
	vid = torch.cat(im_vs, dim=0)
	return vid

def prepare_video_from_folder(curr_path):
	files = sorted([x for x in os.listdir(curr_path) if '.png' in x ])
	return prepare_video(files, curr_path)

def write_pickle_file(filename, dataset):
	out = gzip.compress(pickle.dumps(dataset))
	f = open(filename, 'wb')
	f.write(out)
	f.close()

def preprocess(args, model, device):
	sign_dataset = args.dataset
	main_path = args.base_folder
	subset = args.subset
	batch_size = int(args.batch_size)

	if sign_dataset == 'How2Sign':
		video_path = main_path
		anno_path = subset + '.csv'
	elif sign_dataset == 'Phoenix':
		video_path = os.path.join(main_path, 'features', 'fullFrame-210x260px')
		anno_path = os.path.join('annotations', 'manual', 'PHOENIX-2014-T.' + subset + '.corpus.csv')
	else:
		raise Exception('Error in dataset name.')
	
	out_file = args.out_folder
	if not os.path.exists(out_file):
		os.mkdir(out_file)
	out_file = os.path.join(out_file, sign_dataset + '.dd')

	try:
		f = open(os.path.join(main_path, anno_path))
		anno = f.read().strip()
		f.close()

		anno = anno.split('\n')
		delimiter = datasets[sign_dataset]['delimiter']
		ind = { x[1]:x[0] for x in enumerate(anno[0].split(delimiter)) }
		anno = anno[1:]

		i = int(args.start_ind)
		end_ind = int(args.end_ind)
		if end_ind == -1:	# using -1 to default to full length
			end_ind = len(anno)-1	# inclusive range
		
		dataset = []

		print(f'Running subset: {subset}')

		while i<=end_ind:
			print(f'Current file number: {i}')
			csv_line = anno[i]
			line = csv_line.split(delimiter)
			
			res = dict()
			
			fields = datasets[sign_dataset]
			res['name'] = line[ ind[ fields['name_field']] ]
			res['signer'] = line[ ind[ fields['signer_field'] ] ] if fields['has_signer_info'] else ''
			res['gloss'] = line[ ind[ fields['gloss_field'] ] ] if fields['has_gloss'] else ''
			res['text'] = line[ ind[ fields['text_field'] ] ]
			
			curr_path = os.path.join(video_path, subset + ('_images' if sign_dataset == 'How2Sign' else ''), res['name'])
			
			if not os.path.isdir(curr_path):
				print('Skipping ::',i,'::',curr_path,'\nNot a directory.')
				i += 1
				continue
			try:
				files = sorted([x for x in os.listdir(curr_path) if '.png' in x ])
			except Exception as e:
				print('Skipping file at:',curr_path,'Due to exception.',sep='\n')
				i += 1
				continue

			vid = prepare_video(files, curr_path)
			vid = vid.to(device)
			with torch.no_grad():
				out = model(vid)
			res['sign'] = out.cpu().detach().numpy()
			dataset.append(res)
			if len(dataset)==batch_size:
				print('Writing pickle at file no:', i)
				write_pickle_file(out_file+'.'+subset+'.'+str(i).zfill(6), dataset)
				dataset = []		# pickle filename will contain the number of the last video completed
			i += 1
		if len(dataset)>0:
			write_pickle_file(out_file+'.'+subset+'.'+str(i-1).zfill(6), dataset)
	except Exception as e:
		print('Aborted preprocessing due to:\n', str(e))


def get_model(device):
	model = models.convnext_large(pretrained = True)
	l = nn.Linear(1536, 1024)#nn.Linear(4096, 1024) #1536
	model.classifier[2] = l
	for cn, child in enumerate(model.children()):
		if cn > 0:
			for param in child.parameters():
				param.requires_grad = True
		else:
			for gcn, grand_child in enumerate(child.children()):
				if gcn > 5:
					for param in grand_child.parameters():
						param.requires_grad = True
				else:
					for param in grand_child.parameters():
						param.requires_grad = False
	
	# model.classifier[6].weight.requires_grad = True
	# model.classifier[6].bias.requires_grad = True
	model = model.to(device)
	return model

def get_data_dict(file_loc):
	data = load_dataset_file(file_loc)
	data_dict = dict(map( lambda x : (x['name'], x), data) )
	return data_dict

def get_vid_name(name):
	return name.split('/')[-1]

def get_vid_names_from_keys(names):
	return list(map(get_vid_name, names))

def train_one_epoch(data_dict, part_folder, raw_names, model, optimizer, device, bs = 8):
	logging.info(f'Starting train with batch size: {bs}')
	n = 0
	total_loss = 0
	prev_loss = float('inf')
	model.train()

	for i, rn in enumerate(raw_names):
		#logging.info(f'Training video number {i}, {device}')
		n += 1
		optimizer.zero_grad()
		cur_vid = prepare_video_from_folder(os.path.join(part_folder, get_vid_name(rn))).to(device)
		out_ref = data_dict[rn]['sign'].to(device)

		o = model(cur_vid)

		loss = nn.MSELoss()

		target = out_ref#torch.randn(1,1024)

		output = loss(o, target)
		total_loss += output.item()
		output.backward()
		optimizer.step()
		if n%bs == 0 or i == len(raw_names)-1:
			logging.info(f'Batch loss: {total_loss/n}')
			if total_loss/n - prev_loss < 0:
				logging.info('Saving model due to loss improvement in batch.')
				model_scripted = torch.jit.script(model)
				model_scripted.save('model_scripted_batch.pt')
				logging.info(f'New best batch model saved')
				prev_loss = total_loss/n

			n = 0
			total_loss = 0

def evaluate_model(dev_dict, dev_folder, dev_raw_names, model, optimizer, device):
	logging.info('Evaluating model')
	model.eval()
	total_loss = 0
	for i, rn in enumerate(dev_raw_names):
		#logging.info(f'Training video number {i}, {device}')
		optimizer.zero_grad()
		cur_vid = prepare_video_from_folder(os.path.join(dev_folder, get_vid_name(rn))).to(device)
		out_ref = dev_dict[rn]['sign'].to(device)

		o = model(cur_vid)

		loss = nn.MSELoss()

		target = out_ref#torch.randn(1,1024)

		output = loss(o, target)
		total_loss += output.item()

	avg_loss = total_loss/len(dev_raw_names)
	logging.info(f'Average loss: {avg_loss}')
	return avg_loss

def main():
	# ap = argparse.ArgumentParser("DeepDive")
	# ap.add_argument("--dataset", help="Which dataset to run on")
	# ap.add_argument("--base_folder", help="Base folder for all the data")
	# ap.add_argument("--out_folder", help="Base folder to write the output features")
	# ap.add_argument("--subset", default='dev', help="Subset of data (of train, dev, test). Defaults to dev.")
	# ap.add_argument("--start_ind", default=0, help="Entry in csv to start from (0 indexed, inclusive range)")
	# ap.add_argument("--end_ind", default=-1, help="Entry in csv to end with (0 indexed, inclusive range)")
	# 				# Using -1 to default to the end of the csv
	# ap.add_argument("--batch_size", default=20, help="No of videos pickled in on batch.")
	# args = ap.parse_args()
	author_train_embeddings_path = '/scratch1/maiyaupp/phoenix/author_embeddings/phoenix14t.pami0.train'
	author_dev_embeddings_path = '/scratch1/maiyaupp/phoenix/author_embeddings/phoenix14t.pami0.dev'
	base_images_folder = '/scratch1/maiyaupp/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
	train_folder = os.path.join(base_images_folder, 'train')
	dev_folder = os.path.join(base_images_folder, 'dev')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Running on: {device}')
	
	# Model initialization
	model = get_model(device)

	optimizer = optim.Adam(model.parameters())
	optimizer.zero_grad()
	# Load author embeddings
	data_dict = get_data_dict(author_train_embeddings_path)
	vid_names = get_vid_names_from_keys(data_dict.keys())
	raw_names = list(data_dict.keys())

	dev_dict = get_data_dict(author_dev_embeddings_path)
	dev_raw_names = list(dev_dict.keys())


	file_names = raw_names#[:4]
	#file_names.append('train/09August_2011_Tuesday_heute-2641')
	least_loss = float('inf')
	for i in range(1000):
		logging.info(f'Training epoch: {i}')
		random.shuffle(file_names)
		train_one_epoch(data_dict, train_folder, file_names, model, optimizer, device, 100)
		cur_dev_loss = evaluate_model(dev_dict, dev_folder, dev_raw_names, model, optimizer, device)
		if cur_dev_loss < least_loss:
			least_loss = cur_dev_loss
			model_scripted = torch.jit.script(model)
			model_scripted.save('model_scripted.pt')
			logging.info(f'New best model saved')


	logging.info(f'---------------------------')

if __name__ == "__main__":
    main()