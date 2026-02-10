import logging
import os
from PIL import Image
import imagehash
from pathlib import Path
import glob
from src.settings import *
from dataclasses import dataclass
from src.exception import CustomException
import sys

@dataclass
class DataCleanConfig:
    accepted_image_types: list[str] = ("BMP", "GIF", "JPEG", "PNG")

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

    def check_if_images_valid(self, data_dir: str) -> (list[str], list[str]):
        """
        Check validity of images for use in tensorflow.

        Args:
            data_dir: Directory where images are located

        Returns:
            not_valid: List of image file paths where the image cannot be read (image type None).
            not_supported: List of image file paths where image type is not supported by Tensorflow.
        """

        img_type_accepted_by_tf = list(self.data_clean_config.accepted_image_types)
        search_path = os.path.join(data_dir, "*")
        image_extensions = self.get_unique_extensions(glob.glob(search_path) )

        not_valid = []
        not_supported =[]

        for filepath in Path(data_dir).rglob("*"):
          if filepath.suffix.lower() in image_extensions:
              img = Image.open(filepath)
              img_type = img.format
              if img_type is None:
                  print(f"{filepath} is not an image")
                  not_valid.append(filepath)
              elif img_type not in img_type_accepted_by_tf:
                  print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                  not_supported.append(filepath)

        return [str(f) for f in not_valid], [str(f) for f in not_supported]

    def convert_webp_to_jpeg(self, file_list: list[str]) -> None:
        """
        Convert WEBP files to JPEG so they can be read by tensorflow.
        Original image files are deleted.

        """
        for file in file_list:
            root, ext = os.path.splitext(file)
            with Image.open(file) as img:
                img_type = img.format
                if img_type == "WEBP":
                    try:
                        img.save(root + ".jpg", "JPEG")
                        logging.info(f"Converting {file} to JPEG ...")
                        os.remove(file)
                    except Exception as e:
                        logging.info(f'Unable to convert {file}')
                        raise CustomException(e, sys)
                else:
                    raise ValueError("Only WEBP files accepted")

    def remove_files(self, file_list: list[str]) -> None:
        """
        Delete files in a file list
        """
        for file in file_list:
            os.remove(file)