import pandas as pd

from src.logger import logging
import os
from PIL import Image
from PIL import UnidentifiedImageError
import imagehash
from pathlib import Path
from src.settings import *
from dataclasses import dataclass
from src.exception import CustomException
import sys
from src.utils import get_model_predictions

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

                    # print('{} duplicate of {}'.format(file_name, ori_file_name))

                    image_ori = Image.open(img_hashes[_hash])
                    image_ori_size = image_ori.size[0] * image_ori.size[1]
                    image_size = image.size[0] * image.size[1]

                    # keep image with larger size, save duplicate file names
                    if image_ori_size >= image_size:
                        duplicate_images.append(image_fn)
                        logging.info(f'Image {file_name} is a duplicate')
                    else:
                        duplicate_images.append(img_hashes[_hash])
                        logging.info(f'Image {ori_file_name} is a duplicate')
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
                        logging.info(f"{filepath} is not an image")
                        not_valid.append(filepath)
                    elif img_type not in img_type_accepted_by_tf:
                        logging.info(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                        not_supported.append(filepath)
            except (UnidentifiedImageError, FileNotFoundError, OSError):
                # check if file has an image extension
                root, ext = os.path.splitext(filepath)
                if ext.lower() in self.data_clean_config.image_extensions:
                    not_valid.append(filepath)
                else:
                    logging.info(f"Skipping: {filepath}, can't be read by Pillow ")
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
                    logging.info(f"Skipping: {file}, can't be read by Pillow ")
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
                    img_ext = "." + img_type.lower()
                    root, ext = os.path.splitext(file)

                    if ext.lower() == '.jpg' and img_ext == '.jpeg':
                        pass
                    elif ext.lower() != img_ext:
                        logging.info(f"Renaming {file} to {root + img_ext}")
                        os.rename(file, root + img_ext)
            except (UnidentifiedImageError, FileNotFoundError, OSError):
                logging.info(f"Skipping {file}, not readable by Pillow")
                continue

    def relabel_images(self,
                       model_path: str = None,
                       image_dataset_path: str = None,
                       conf_low: float = 0.15,
                       conf_high: float = 0.95,
                       dir_class_0: str = None,
                       dir_class_1: str = None):
        """
        Relabel newly extracted images based on a pretrained model. Currently only works for binary labels.

            Args:
                model_path: File path to the .keras saved model.
                image_dataset_path: File path to the newly extracted images.
                conf_high: Confidence threshold above which we will move images to class 1.
                conf_low: Confidence threshold below which we will move images to class 0.
                dir_class_0: Directory path containing images currently in class 0 (alphabetically ordered).
                dir_class_1: Directory path containing images currently in class 1 (alphabetically ordered).

            Notes:
                Confidence thresholds (conf_high and conf_low) refer to the trained model's confidence of its prediction.

                If the model outputs a probability of 0.98, and conf_high = 0.95, then the image will be relabeled
                from class 0 to class 1 (assuming its original label was 0). If conf_high = 0.99, the image is not relabeled.
                Likewise, if the model outputs a probability of 0.1, and conf_low = 0.15, then the image will be relabeled
                from class 1 to class 0 (assuming its original label was 1). If conf_low = 0.05, the image is not relabeled.

        """

        # get the model predictions
        df_preds = get_model_predictions(model_path=model_path, image_dataset_path=image_dataset_path)

        for index, row in df_preds.iterrows():
            file_name = os.path.basename(row["image_filepath"])

            if row["predicted_label"] != row["expected_label"]:
                model_prob = row["model_prob"]
                if model_prob < conf_low:
                    logging.info(f"Moving image {row['image_filepath']}, with probability {model_prob}")

                    os.rename(os.path.join(dir_class_1, file_name), os.path.join(dir_class_0, file_name) )
                elif model_prob > conf_high:
                    logging.info(f"Moving image {row['image_filepath']}, with probability {model_prob}")

                    os.rename(os.path.join(dir_class_0, file_name), os.path.join(dir_class_1, file_name))
                else:
                    logging.info(f"Mismatched labels, but confidence too low for {row['image_filepath']}")
            else:
                pass


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
        logging.info("\n")
        logging.info("RUNNING DATA CLEANING ...")
        # 1. correct file extensions
        logging.info("Correcting file extensions ...")
        self.correct_file_extensions(data_source, recursive_search=recursive_search)

        # 2. remove duplicates
        logging.info("\n\n")
        logging.info("Removing duplicates...")
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
        logging.info("\n\n")
        logging.info("Removing invalid images...")
        not_valid, not_supported = self.check_if_images_valid(data_source, recursive_search=recursive_search)
        self.remove_files(not_valid)
        # 4. convert not supported to jpeg
        logging.info("\n\n")
        logging.info("Converting not supported to jpeg...")
        self.convert_image_to_jpeg(file_list=not_supported)



if __name__ == "__main__":
    data_cleaning = DataCleaning()
    data_dir = "../../../solar_dust_detection/Detect_solar_dust_original"
    data_cleaning.run_cleaning_steps(data_source=data_dir)
