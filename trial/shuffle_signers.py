# from torchtext import data
# from torchtext.data import Field, RawField
# from typing import List, Tuple
import pickle
import gzip
import operator
# import torch
import pandas as pd

# Signer 2, 6, 8 

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        # print(type(loaded_object))
        return loaded_object


def write_pickle_file(filename, dataset):
	out = gzip.compress(pickle.dumps(dataset))
	f = open(filename, 'wb')
	f.write(out)
	f.close()


def count_signers(obj, _type):
    dict_signers = {}
    for item in obj:
        signer = item["signer"]
        if signer in dict_signers:
            dict_signers[signer][0] += 1
        else:
            dict_signers[signer] = [1]

    pd_signers = pd.DataFrame.from_dict(dict_signers)
    cols = ["Signer01", "Signer02", "Signer03", "Signer04", "Signer05", "Signer06", "Signer07", "Signer08"]
    pd_signers = pd_signers.reindex(columns = cols)
    print("\n", pd_signers)
    _sum = 0
    for item in dict_signers.values():
        _sum += item[0] 
    print("Total=", _sum)  # 7096, 642, 519
    # return pd_signers


train_file = "/scratch1/maiyaupp/phoenix/author_embeddings/phoenix14t.pami0.train"
dev_file = "/scratch1/maiyaupp/phoenix/author_embeddings/phoenix14t.pami0.dev"
test_file = "/scratch1/maiyaupp/phoenix/author_embeddings/phoenix14t.pami0.test"

train_obj = load_dataset_file(train_file)
dev_obj = load_dataset_file(dev_file)
test_obj = load_dataset_file(test_file)


# count_signers(train_obj, "train")
# count_signers(dev_obj, "dev")
# count_signers(test_obj, "test")

new_train = []
new_dev = []
new_test = []

len_half_signer08 = 400
count = 0

for i,obj in enumerate(train_obj):

    if obj["signer"] == "Signer02":
        new_test.append(obj)
    elif obj["signer"] == "Signer06":
        new_dev.append(obj)
    elif obj["signer"] == "Signer08":
        count += 1
        if count <= len_half_signer08:
            new_dev.append(obj)
        else:
            new_test.append(obj)
    else:
        new_train.append(obj)


for i,obj in enumerate(dev_obj):

    if obj["signer"] ==  "Signer06" or obj["signer"] ==  "Signer08":
        new_dev.append(obj)
    elif obj["signer"] ==  "Signer02":
        new_test.append(obj)
    else:
        new_train.append(obj)

for i,obj in enumerate(test_obj):
    if obj["signer"] ==  "Signer02" or obj["signer"] ==  "Signer08":
        new_test.append(obj)
    elif obj["signer"] ==  "Signer06":
        new_dev.append(obj)
    else:
        new_train.append(obj)

# print("\n\n\n")
# count_signers(new_train, "train")
# count_signers(new_dev, "dev")
# count_signers(new_test, "test")

# write_pickle_file("/scratch1/maiyaupp/phoenix/author_embeddings_signers_shuffled/phoenix14t.pami0.dev", new_dev)
# write_pickle_file("/scratch1/maiyaupp/phoenix/author_embeddings_signers_shuffled/phoenix14t.pami0.test", new_test)
print("Starting to upload file...")
write_pickle_file("/scratch1/maiyaupp/phoenix/author_embeddings_signers_shuffled/phoenix14t.pami0.train", new_train)
print("end")












