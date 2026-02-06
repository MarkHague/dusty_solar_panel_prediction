import os
import boto3
from dotenv import load_dotenv
from PIL import Image
import imagehash

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

# TODO add test for find_exact_duplicates()
def find_exact_duplicates(image_file_paths: list[str]) -> list[str]:
    """
    Finds exact image duplicates using imagehash library.
    The duplicates are chosen based on the dimensions of the image, with smaller images labelled as duplicates.
    Only .jpg, .jpeg, .webp and .png files are allowed.

    Args:
        image_file_paths: List of image file paths.

    Returns:
        List of image files paths which are duplicates.
    """

    img_hashes = {}
    duplicate_images = []

    for image_fn in sorted(image_file_paths):

        image = Image.open(image_fn)
        _hash = imagehash.average_hash(image)

        if _hash in img_hashes:
            base_path, file_name = os.path.split(image_fn)
            base_path, ori_file_name = os.path.split(img_hashes[_hash])

            print('{} duplicate of {}'.format(file_name, ori_file_name))

            image_ori = Image.open(img_hashes[_hash])
            image_ori_size = image_ori.size[0] * image_ori.size[1]
            image_size = image.size[0] * image.size[1]

            # keep image with larger size, save duplicate file names
            if image_ori_size >= image_size:
                duplicate_images.append(image_fn)
                print(f'Image {file_name} is a duplicate')
            else:
                duplicate_images.append(img_hashes[_hash])
                print(f'Image {ori_file_name} is a duplicate')
                # update file name of new largest image
                img_hashes[_hash] = image_fn

        else:
            img_hashes[_hash] = image_fn

    return duplicate_images

# TODO add checking if images are valid for not - currently in Colab notebook