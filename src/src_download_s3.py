
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
            s3 = boto3.client('s3')
            s3.download_file(bucket, aws_file, file)


l_file='text_classifier.pth'

get_s3_bucket('jack-ml-models', 'models/text_classifier.pth', l_file)