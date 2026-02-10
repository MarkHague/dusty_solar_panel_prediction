import os
import boto3
from dotenv import load_dotenv
from PIL import Image
import imagehash
from pathlib import Path
import glob

def get_s3_client() -> None:

    """
    Get an AWS S3 client to retrieve training data.
    """
    load_dotenv()

    return boto3.client(
        "s3",
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')

    )

