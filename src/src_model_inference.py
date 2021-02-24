from fastai.text.all import *
import boto3
import os

def get_s3_bucket(bucket, aws_file, l_file):
    """
    Pass AWS info to download file
    :param bucket: name of S£ bucket
    :param aws_file: path to bucket file
    :param l_file: path to loccl file
    :return: None
    """
    for f in os.listdir():
        # Check if file in local dir
        if l_file != f:
            s3 = boto3.client(
                's3',
                'aws_access_key_id',
                'aws_secret_access_key'
            )
            s3.download_file(bucket, aws_file, l_file)


l_file='text_classifier.pth'

get_s3_bucket('jack-ml-models', 'models/text_classifier.pth', l_file)

path = untar_data(URLs.IMDB)

# Text classifier DataLoaders
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path), CategoryBlock),
    get_y=parent_label,
    get_items=partial(get_text_files, folders=['train', 'test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path, path=path, bs=128, seq_len=72)

# Create text model 
l = text_classifier_learner(dls_clas, 
                            AWD_LSTM, 
                            drop_mult=0.5, 
                            metrics=accuracy)#.to_fp16()


# Load trained model
l = l.load(l_file)

# predict
print(l.predict("That was such a good movie"))
