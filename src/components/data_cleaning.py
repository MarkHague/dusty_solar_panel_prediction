from src.logger import logging
import os
from PIL import Image
from PIL import UnidentifiedImageError
import imagehash
from pathlib import Path
import glob
from src.settings import *
from dataclasses import dataclass
from src.exception import CustomException
import sys

@dataclass
class DataCleanConfig:
    accepted_image_types: tuple[str] = ("BMP", "GIF", "JPEG", "PNG")

class DataCleaning:

    def __init__(self):
        self.data_clean_config = DataCleanConfig()

    def find_exact_duplicates(self,image_file_paths: list[str]) -> list[str]:
        """
        Finds exact image duplicates using the imagehash library.
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

            try:
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

            except Exception as e:
                raise CustomException(e, sys)

        return duplicate_images

    def get_unique_extensions(self, file_paths: list[str]) -> list[str]:
        """
        Returns a list of unique file extensions given the input file paths.
        """
        extensions = [os.path.splitext(f)[1] for f in file_paths]
        return list(set(extensions))

    def check_if_images_valid(self, data_dir: str = None, recursive_search: bool = False) -> (list[str], list[str]):
        """
        Check validity of images in a directory for use in tensorflow.

        Args:
            data_dir: Directory where images are located.
            recursive_search: Search recursively or not from data_dir

        Returns:
            not_valid: List of image file paths where the image cannot be read (image type None).
            not_supported: List of image file paths where image type is not supported by Tensorflow.
        """

        img_type_accepted_by_tf = list(self.data_clean_config.accepted_image_types)
        # search_path = os.path.join(data_dir, "*")
        # image_extensions = self.get_unique_extensions(glob.glob(search_path, recursive=recursive_search) )
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico'}

        not_valid = []
        not_supported =[]

        for filepath in glob.glob(os.path.join(data_dir,"*"), recursive=recursive_search ):
            try:
                img = Image.open(filepath)
                img_type = img.format
                print(f"FILE: {filepath}")
                if img_type is None:
                    print(f"{filepath} is not an image")
                    not_valid.append(filepath)
                elif img_type not in img_type_accepted_by_tf:
                    print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                    not_supported.append(filepath)
            except (UnidentifiedImageError, FileNotFoundError, OSError):
                # check if file has an image extension
                root, ext = os.path.splitext(filepath)
                if ext.lower() in image_extensions:
                    not_valid.append(filepath)
                else:
                    print(f"Skipping: {filepath}, can't be read by Pillow ")
                continue


        return [str(f) for f in not_valid], [str(f) for f in not_supported]

    def convert_webp_to_jpeg(self, file_list: list[str] = None, delete_original: bool = True) -> None:
        """
        Convert WEBP and MPO files to JPEG so they can be read by tensorflow.
        Original image files are deleted by default.

        """
        for file in file_list:
            root, ext = os.path.splitext(file)
            with Image.open(file) as img:
                img_type = img.format

                if img_type == "WEBP" or img_type == "MPO":
                    if ext.lower() != ".webp":
                        # rename so we can differentiate between old and new files
                        os.rename(file, root + ".webp")
                    try:
                        img.save(root + ".jpg", "JPEG")
                        logging.info(f"Converting {file} to JPEG ...")
                        if delete_original:
                            os.remove(file)
                    except Exception as e:
                        logging.info(f'Unable to convert {file}')
                        raise CustomException(e, sys)
                else:
                    raise ValueError("Only WEBP and MPO files accepted")

    def remove_files(self, file_list: list[str]) -> None:
        """
        Delete files in a file list
        """
        for file in file_list:
            os.remove(file)

    def correct_file_extensions(self, data_dir: str) -> None:
        """
        Correct the file extension if it does not reflect the image type.
        """

        files = os.listdir(data_dir)
        for file in files:
            with Image.open(file) as img:
                img_type = img.format
                root, ext = os.path.splitext(file)

                if ext.lower() != img_type.lower():
                    os.rename(file, root + img_type.lower())