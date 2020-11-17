import os
import random
import shutil
import numpy as np
import pandas as pd

folder_name = 'n50'
df = pd.read_csv('/data/FixRes-master/image/{}/home/rstudio/work/ecsite/drive/{}/{}.csv'.format(folder_name,folder_name,folder_name), encoding = "utf-8")
df1 = pd.DataFrame(columns=['filename','label'])
df1['filename'] = df['filename']
df1['label'] = df['l1'].astype(str) + ">" + df['l2'].astype(str) + ">" + df['l3'].astype(str)

# Remove images with same category to folder
def category_folder(SOURCE, df1):
    for filename in os.listdir(SOURCE):
        this_file = SOURCE + filename
        for i in range(len(df1)):
            df1['label'][i] = df1['label'][i].replace('>nan','')
            df1['filename'][i] = df1['filename'][i].replace(':','_')
            if filename == df1['filename'][i]:
                path = '/data/FixRes-master/image/img_folder/'+ str(df1['label'][i]).replace('/', ',')
                destination = path + '/' + filename
                print(path)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                shutil.copyfile(this_file, destination)
    try:
        shutil.rmtree("/data/FixRes-master/image/img_folder/nan")
    except:
        pass

# delete folders which number of images < n
def delete_folder(SOURCE):
    for foldername in os.listdir(SOURCE):
        list = os.listdir(SOURCE + '/' + foldername)
        if len(list) < 5:
            shutil.rmtree(SOURCE + '/' + foldername)

# split each folder in to train and val set
def split_data(SOURCE, TRAINING, VALIDATION, TRAIN_PERCENT, VALID_PERCENT):
    files = []

    for filename in os.listdir(SOURCE):
        file = SOURCE + '/' + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            pass

    training_length = int(len(files) * TRAIN_PERCENT)
    valid_length = int(len(files) * VALID_PERCENT)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    valid_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE + '/' + filename
        destination = TRAINING + '/' + filename
        shutil.copyfile(this_file, destination)

    for filename in valid_set:
        this_file = SOURCE + '/' + filename
        destination = VALIDATION + '/' + filename
        shutil.copyfile(this_file, destination)

def create_train_val_folder(SOURCE, DESTINATION):
    for foldername in os.listdir(SOURCE):
        path_train = DESTINATION+ 'train/' + foldername
        path_val = DESTINATION + 'val/' + foldername
        for path in [path_train, path_val]:
            try:
                os.mkdir(path)
            except OSError:
                pass
        source = SOURCE + foldername
        split_data(source, path_train, path_val, 0.8, 0.2)

source_img='/data/FixRes-master/image/{}/home/rstudio/work/ecsite/drive/{}/'.format(folder_name,folder_name)
source_folder='/data/FixRes-master/image/img_folder/'
destination='/data/FixRes-master/image_filter/'

# category_folder(source_img, df1)
# delete_folder(source_folder)

create_train_val_folder(source_folder, destination)


