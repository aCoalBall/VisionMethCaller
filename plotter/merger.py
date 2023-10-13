import numpy as np
import os

train_path = '/home/coalball/projects/methBert2/toys/new_data/train/'
val_path = '/home/coalball/projects/methBert2/toys/new_data/val/'
test_path = '/home/coalball/projects/methBert2/toys/new_data/test/'

val_files_path = os.listdir(val_path)
merged_val_file_path = '/home/coalball/projects/methBert2/toys/new_data/val/val.npy'
merge_val = open(merged_val_file_path, 'wb')
for p in val_files_path:
    abs_path = '/home/coalball/projects/methBert2/toys/new_data/val/' + p
    chunk = open(abs_path, 'rb')
    while True:
        try:
            x = np.load(chunk, allow_pickle=True)
            np.save(merge_val, x, allow_pickle=True)
        except:
            break
    chunk.close()
merge_val.close()
print('val set merged')

test_files_path = os.listdir(test_path)
merged_test_file_path = '/home/coalball/projects/methBert2/toys/new_data/test/test.npy'
merge_test = open(merged_test_file_path, 'wb')
for p in test_files_path:
    abs_path = '/home/coalball/projects/methBert2/toys/new_data/test/' + p
    chunk = open(abs_path, 'rb')
    while True:
        try:
            x = np.load(chunk, allow_pickle=True)
            np.save(merge_test, x, allow_pickle=True)
        except:
            break
    chunk.close()
merge_test.close()
print('test set merged')


train_files_path = os.listdir(train_path)
merged_train_file_path = '/home/coalball/projects/methBert2/toys/new_data/train/train.npy'
merge_train = open(merged_train_file_path, 'wb')
for p in train_files_path:
    abs_path = '/home/coalball/projects/methBert2/toys/new_data/train/' + p
    chunk = open(abs_path, 'rb')
    while True:
        try:
            x = np.load(chunk, allow_pickle=True)
            np.save(merge_train, x, allow_pickle=True)
        except:
            break
    chunk.close()
merge_train.close()
print('training set merged')
print('done')