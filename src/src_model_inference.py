from fastai.text.all import *
import urllib

path = untar_data(URLs.IMDB)

# Create language model DataLoaders
# get_imdb = partial(get_text_files, folders=['train', 'test', 'unsup'])
# dls_lm = DataBlock (
#     blocks=TextBlock.from_folder(path, is_lm=True), # other option is .from_df
#     get_items=get_imdb,
#     splitter=RandomSplitter(0.1)
# ).dataloaders(path, path=path, bs=128, seq_len=80)

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
                            metrics=accuracy).to_fp16()

# Get S3 bucket
# myfile = opener.open(myurl)

# def get_s3_bucket(bucket_n):
#     response = s3.list_objects_v2(Bucket=bucket_n )
#     for content in response['Contents']:
#         obj_dict = s3.get_object(Bucket=bucket_n, Key=content['Key'])
#         contents = obj_dict['Body'].read().decode('utf-8')
#     return obj_dict, content
#     # opener = urllib.URLopener()
#     # myurl = "s3://jack-ml-models/models/"

# myfile = get_s3_bucket('')

# Load trained model
l = l.load('/content/gdrive/MyDrive/Colab Notebooks/FastAI/models/text_classifier')

# predict
l.predict(my_str)