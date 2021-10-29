import boto3
from boto3.session import Session

########## READ READ READ #######################

## model image 의 경우 /home/ubuntu/Desktop/data-conversion/RefineNetwork/data/raw_data/p001/c001/images/1.jpg 로 저장 바랍
## clothes image 의 경우 /home/ubuntu/Desktop/data-conversion/RefineNetwork/data/dataset/clothes/base/c001.jpg 로 저장 바람

########## READ READ READ #######################

def download_file_s3(access_key, secret_key, bucket_name, image_file, file_path):

    ACCESS_KEY = access_key
    SECRET_KEY = secret_key

    session = Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    bucket.download_file(image_file, file_path)

download_file_s3(access_key='ACCESS_KEY', secret_key='SECRET_KEY',
                 bucket_name='bucket_name', image_file='image_name.jpg',
                 file_path='/home/ubuntu/Desktop/data-conversion/RefineNetwork/data/test.jpg')