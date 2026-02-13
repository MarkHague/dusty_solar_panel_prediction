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
    # these need to be immutable, so tuples are good
    accepted_image_types: tuple[str] = ('BMP', 'GIF', 'JPEG', 'PNG')
    image_extensions: set[str] = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico', '.mpo')

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

            except UnidentifiedImageError:
                print(f"Cannot open {image_fn}, skipping ...")
                continue

            except Exception as e:
                raise CustomException(e, sys)

        return duplicate_images

    def get_unique_extensions(self, file_paths: list[str]) -> list[str]:
        """
        Returns a list of unique file extensions given the input file paths.
        """
        extensions = [os.path.splitext(f)[1] for f in file_paths]
        return list(set(extensions))

    def check_if_images_valid(self, data_dir: str = None, recursive_search: bool = True) -> (list[str], list[str]):
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

        not_valid = []
        not_supported =[]

        path_check = Path(data_dir)
        glob_method = path_check.rglob if recursive_search else path_check.glob

        for filepath in glob_method("*"):
            try:
                with Image.open(filepath) as img:
                    img_type = img.format
                    if img_type is None:
                        print(f"{filepath} is not an image")
                        not_valid.append(filepath)
                    elif img_type not in img_type_accepted_by_tf:
                        print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                        not_supported.append(filepath)
            except (UnidentifiedImageError, FileNotFoundError, OSError):
                # check if file has an image extension
                root, ext = os.path.splitext(filepath)
                if ext.lower() in self.data_clean_config.image_extensions:
                    not_valid.append(filepath)
                else:
                    print(f"Skipping: {filepath}, can't be read by Pillow ")
                continue


        return [str(f) for f in not_valid], [str(f) for f in not_supported]

    def convert_image_to_jpeg(self, file_list: list[str] = None) -> None:
        """
        Convert WEBP and MPO files to JPEG so they can be read by tensorflow.
        Original image files are deleted by default.

        Args:
            file_list: list of file paths.
            delete_original: If True, delete the original file.

        """
        for file in file_list:
            root, ext = os.path.splitext(file)
            with Image.open(file) as img:
                try:
                    img.save(root + ".jpg", "JPEG")
                    logging.info(f"Converting {file} to JPEG ...")
                    if ext != ".jpg":
                        os.remove(file)
                except UnidentifiedImageError:
                    print(f"Skipping: {file}, can't be read by Pillow ")
                    continue
                except Exception as e:
                    logging.info(f'Unable to convert {file}')
                    raise CustomException(e, sys)

    def remove_files(self, file_list: list[str]) -> None:
        """
        Delete files in a file list.
        """
        for file in file_list:
            os.remove(file)

    def correct_file_extensions(self, data_dir: str, recursive_search: bool = True) -> None:
        """
        Correct the file extension if it does not reflect the image type.

        Args:
            data_dir: Directory to search for incorrect file extensions
            recursive_search: Search recursively or not from data_dir

        """

        path = Path(data_dir)
        glob_method = path.rglob if recursive_search else path.glob
        files = glob_method("*")

        for file in files:
            try:
                with Image.open(file) as img:
                    img_type = img.format
                    root, ext = os.path.splitext(file)

                    if ext.lower() != img_type.lower():
                        ext_str = "."+img_type.lower()
                        print(f"Renaming {file} to {root + ext_str}")
                        os.rename(file, root + ext_str)
            except (UnidentifiedImageError, FileNotFoundError, OSError):
                print(f"Skipping {file}, not readable by Pillow")
                continue
                # TODO add test for this

    def run_cleaning_steps(self, data_source: str = None, recursive_search: bool = True) -> None:
        """
        Run the following data cleaning steps on image data, before passing to tensorflow:
            1. Correct the file extension if it does not reflect the image type.
            2. Remove exact duplicates (keeping the larger images).
            3. Remove invalid images (e.g. corrupted).
            4. Convert unsupported image type to jpeg.

        Args:
            data_source: Directory where image data is stored
            recursive_search: Search recursively or not from data_source
        """
        # 1. correct file extensions
        print("Correcting file extensions ...")
        self.correct_file_extensions(data_source, recursive_search=recursive_search)

        # 2. remove duplicates
        print("Removing duplicates...")
        path = Path(data_source)
        glob_method = path.rglob if recursive_search else path.glob

        file_types = ["*" + file_type for file_type in self.data_clean_config.image_extensions]
        image_file_names = []
        for files in file_types:
            image_file_names.extend(glob_method(files) )
        image_file_names = [str(f) for f in image_file_names]

        duplicates = self.find_exact_duplicates(image_file_names)
        self.remove_files(duplicates)

        # 3. remove invalid images
        print("Removing invalid images...")
        not_valid, not_supported = self.check_if_images_valid(data_source, recursive_search=recursive_search)
        self.remove_files(not_valid)
        # 4. convert not supported to jpeg
        print("Converting not supported to jpeg...")
        self.convert_image_to_jpeg(file_list=not_supported)



if __name__ == "__main__":
    data_cleaning = DataCleaning()
    data_dir = "../../../solar_dust_detection/Detect_solar_dust_original"
    data_cleaning.run_cleaning_steps(data_source=data_dir)
