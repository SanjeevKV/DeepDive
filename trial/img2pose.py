# from cv2 import imread, resize
#from PIL import Image
import numpy as np
import os
import gzip
import pickle
import argparse
import sys
# import torch
# import torchvision.models as models
# from torchvision import transforms

sys.path.append('/home1/svadiraj/projects/pytorch-openpose')
import openpose_image as oi
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

# torch_preprocess = transforms.Compose([
#     transforms.Resize((227,227)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

def prepare_image(in_file, out_file):
	oi.img_to_openpose(in_file, out_file)


def prepare_video(files, curr_path, out_path):
	im_vs = []
	for fil in files:
		in_file = os.path.join(curr_path, fil)
		out_file = os.path.join(out_path, fil)
		prepare_image(in_file, out_file)


def write_pickle_file(filename, dataset):
	out = gzip.compress(pickle.dumps(dataset))
	f = open(filename, 'wb')
	f.write(out)
	f.close()

def preprocess(args):
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
	
	out_path = args.out_folder
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	#out_file = os.path.join(out_file, sign_dataset + '.dd')

	out_root = out_path
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
			print(f'Cur name: {res["name"]}')
			
			curr_path = os.path.join(video_path, subset + ('_images' if sign_dataset == 'How2Sign' else ''), res['name'])
			out_path = os.path.join(out_root, subset + ('_images' if sign_dataset == 'How2Sign' else ''), res['name'])

			if not os.path.exists(out_path):
				os.makedirs(out_path)

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

			vid = prepare_video(files, curr_path, out_path)
			i += 1

	except Exception as e:
		print('Aborted preprocessing due to:\n', str(e))

def main():
	ap = argparse.ArgumentParser("DeepDive")
	ap.add_argument("--dataset", help="Which dataset to run on")
	ap.add_argument("--base_folder", help="Base folder for all the data")
	ap.add_argument("--out_folder", help="Base folder to write the output features")
	ap.add_argument("--subset", default='dev', help="Subset of data (of train, dev, test). Defaults to dev.")
	ap.add_argument("--start_ind", default=0, help="Entry in csv to start from (0 indexed, inclusive range)")
	ap.add_argument("--end_ind", default=-1, help="Entry in csv to end with (0 indexed, inclusive range)")
					# Using -1 to default to the end of the csv
	ap.add_argument("--batch_size", default=20, help="No of videos pickled in on batch.")
	args = ap.parse_args()
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#print('Running on:',device)
	#model = models.alexnet(pretrained=True)
	#model.classifier = model.classifier[:6]
	#model = models.vgg16(pretrained = True)
	#model = model.to(device)
	preprocess(args)

if __name__ == "__main__":
    main()