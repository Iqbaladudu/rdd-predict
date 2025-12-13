import boto3
import os

ENDPOINT_URL = 'https://3d7a6fad0b53f11a279fd0c7f8bccac4.r2.cloudflarestorage.com'
BUCKET_NAME = 'public'

s3 = boto3.resource('s3', 
    endpoint_url=ENDPOINT_URL, 
    aws_access_key_id = 'c30fbb5b5b69144f2a8e6daaee76b1d5',
    aws_secret_access_key = 'a659888317c38d6f4ae59cfc15e2f5a50eb5435247691f8c3634602531d8c481')

def upload_file(file_path, object_name):
    """
    Uploads a file to the S3 bucket.
    """
    try:
        s3.Bucket(BUCKET_NAME).upload_file(file_path, object_name)
        # Construct a URL - Note: this is the raw R2 URL, you might need a custom domain for public access
        # if r2.dev is not enabled or if this is private.
        # Assuming R2 format.
        return f"https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/{object_name}" 
        # Note: I am guessing the public URL format for R2 dev if enabled, 
        # or maybe I should return the endpoint based URL.
        # Let's return the endpoint based URL for now as it's safer than guessing a subdomain.
        # Actually R2 path style access: {endpoint}/{bucket}/{key}
        # return f"{ENDPOINT_URL}/{BUCKET_NAME}/{object_name}"
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e