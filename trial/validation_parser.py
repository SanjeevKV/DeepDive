# from cv2 import imread, resize
import os
import argparse
import sys
from matplotlib import pyplot as plt
import pandas as pd

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def process_line(line):
    split_line = line.split('\t')
    space_rem = list(map(lambda x : x.replace(' ','') if ':' in x else x.replace(' ',':') , split_line) )
    spl_rem = list(map(lambda x : x.replace(',','').replace('(','').replace(')', '') , space_rem) )
    spl_filt = list(filter(lambda x : '*' not in x and '\n' not in x, spl_rem))
    tuples = list(map(lambda x : x.split(':'), spl_filt))
    typecast_tuples = list(map(lambda x : (x[0], float(x[1]) if isfloat(x[1]) else x[1] ), tuples))
    return dict(typecast_tuples)

def process_file(file_path):
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
    lines_as_dict = list(map(lambda x : process_line(x), lines))
    return lines_as_dict

def get_df(file_path):
    lines_as_dict = process_file(file_path)
    return pd.DataFrame(lines_as_dict)

def main():
    ap = argparse.ArgumentParser("DeepDive-ValidationParser")
    ap.add_argument("--val_file", help="Path to validation file")
    args = ap.parse_args()
    get_df(args.val_file)

if __name__ == "__main__":
    main()