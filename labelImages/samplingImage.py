#
# For DIGITS Detectnet, you shuld use 'samplingImage.py'
# Detectnet uses sampled images and all labels
#

import numpy as np
import time
from itertools import repeat
import random
import multiprocessing as mp
import shutil
import os
import sys
import argparse

def get_image_file_list(image_root_dir):
    max_images_per_image = 1224
    target_image_list = ['yoshiki','weiliang','soga','shudo','ohnishi','nakajima','ken','aoki','kanda','iwasaki','matsumura']
    #target_image_list = ['all']
    # Make target file name set
    image_file_set = set()
    for target_image in target_image_list :
        count = max_images_per_image
        file_names = set(os.listdir(os.path.join(image_root_dir, target_image)))
        while(count > 0 and len(file_names) > 0):
            count -= 1
            #image_file_set.add(target_image + '/' + file_names.pop())
            image_file_set.add(file_names.pop())
    # Make target file name list
    image_file_list = []
    for image_file in image_file_set:
        i = 0
        while(not image_file in os.listdir(image_root_dir + '/' + target_image_list[i])):
            i += 1
        image_file_list.append(image_root_dir + '/' + target_image_list[i] + '/' + image_file)
    return image_file_list

def copy_image_files(image_file_list, target_dir):
    for image_file in image_file_list:
        shutil.copy(image_file, target_dir)
    return len(image_file_list)

def partition_to_N(src_list, N):
        return [ src_list[int(len(src_list)*i/N):int(len(src_list)*(i+1)/N)] for i in range(N)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sampling image files and copy them')
    parser.add_argument('image_root',      type=str, help='e.g. [image_root]/[label]/xxx.png.Labels are sepecified on cord.')
    parser.add_argument('--num_threads',      type=int, default=8)
    parser.add_argument('--training_ratio',      type=float, default=0.9)
    parser.add_argument('--output_folder', type=str, default='sampled-images/')
    args = parser.parse_args()

    ts1 = time.time()
    image_file_list = get_image_file_list(args.image_root)
    random.shuffle(image_file_list)
    partitioned_file_list_for_train = partition_to_N(image_file_list[:int(len(image_file_list)*args.training_ratio)], args.num_threads)
    partitioned_file_list_for_validate = partition_to_N(image_file_list[int(len(image_file_list)*args.training_ratio):], args.num_threads)
    ts2 = time.time()
    # multi process run
    with mp.Pool(processes=args.num_threads) as pool:
        num_copy_file_for_train = pool.starmap(copy_image_files, 
            zip(partitioned_file_list_for_train,    repeat(os.path.join(args.output_folder, 'train'))))
        print('Finished coping train data')
        num_copy_file_for_validate = pool.starmap(copy_image_files,
            zip(partitioned_file_list_for_validate, repeat(os.path.join(args.output_folder, 'validate'))))
        print('Finished coping validate data')
        ts3 = time.time()
        print('# all file : ' + str(len(image_file_list)) + '# for train : ' + \
            str(num_copy_file_for_train) + '# for validate : ' + str(num_copy_file_for_validate))
        print('pre process time :' + str(ts2 - ts1) + ', copy time :' + str(ts3 - ts2))

