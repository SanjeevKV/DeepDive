import os
import sys
import pickle
import gzip
from img2vec import write_pickle_file

in_files = sys.argv[1]
out_path = sys.argv[2]
if not os.path.exists(out_path):
    os.mkdir(out_path)

bsize = int(sys.argv[3])
# start_fno = int(sys.argv[4]) # ONLY considering nums in filenames (COMPLETE FILES ONLY)
# end_fno = int(sys.argv[5])
add_path = ''

if os.path.isdir(in_files):
    add_path = in_files
    in_files = os.listdir(in_files)
else:
    in_files = [in_files]

for fi in in_files:
    _, name = os.path.split(fi)
    name, num = name.rsplit('.', 1)
    num = int(num)
    with gzip.open(os.path.join(add_path, fi)) as f:
        fi = pickle.load(f)
    start = num - len(fi)
    while len(fi)>0:
        temp = fi[:bsize]
        fi = fi[bsize:]
        start += bsize
        towri = os.path.join(out_path, name+'.'+str(start).zfill(6))
        write_pickle_file(towri, temp)