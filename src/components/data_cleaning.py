import pandas as pd
from src.logger import logging
import os
from PIL import Image
from PIL import UnidentifiedImageError
import imagehash
from pathlib import Path
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

        """
        for file in file_list:
            root, ext = os.path.splitext(file)
            try:
                with Image.open(file) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
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

    def convert_to_rgb(self, file_list: str = None) -> None:
        """
        Convert and save images to RGB.

        Args:
            file_list: list of file paths.
        """
        for file in file_list:

            try:
                with Image.open(file) as img:

                    if img.mode != "RGB":
                        img.convert("RGB")
                        img.save(file)
                        logging.info(f"Converting {file} to RGB ...")

            except UnidentifiedImageError:
                logging.info(f"Skipping: {file}, can't be read by Pillow ")
                continue
            except OSError as e:
                logging.info(f"Unable to convert {file}, with exception {e}, removing ...")
                os.remove(file)
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
                       conf_low: float = 0.1,
                       conf_high: float = 0.99,
                       perform_image_move: bool = False) -> pd.DataFrame:
        """
        Relabel newly extracted images based on a pretrained model. Currently only works for binary labels.

            Args:
                model_path: File path to the .keras saved model.
                image_dataset_path: File path to the newly extracted images. Must contain both classes as sub-directories.
                conf_high: Confidence threshold above which we will move images to class 1.
                conf_low: Confidence threshold below which we will move images to class 0.
                perform_image_move: If True, move image files to the model predicted class label.
                                    If False, only updates the labels in the dataframe for review.

            Notes:
                Confidence thresholds (conf_high and conf_low) refer to the trained model's confidence of its prediction.

                If the model outputs a probability of 0.98, and conf_high = 0.95, then the image will be relabeled
                from class 0 to class 1 (assuming its original label was 0). If conf_high = 0.99, the image is not relabeled.
                Likewise, if the model outputs a probability of 0.1, and conf_low = 0.15, then the image will be relabeled
                from class 1 to class 0 (assuming its original label was 1). If conf_low = 0.05, the image is not relabeled.

        """

        # get the model predictions
        df_preds = get_model_predictions(model_path=model_path, image_dataset_path=image_dataset_path)
        # get the class names
        names = df_preds["expected_label"].unique()
        class_0_name, class_1_name = names[0], names[1]

        # create a dataframe to track which images are being relabeled
        df_relabelled = df_preds.copy()
        df_relabelled["relabelled_by_model"] = "N"
        # we update this with the model predictions
        df_relabelled["true_label"] = df_relabelled["expected_label"]

        for index, row in df_preds.iterrows():
            file_name = os.path.basename(row["image_filepath"])

            if row["predicted_label"] != row["expected_label"]:
                model_prob = row["model_prob"]
                if model_prob < conf_low:
                    logging.info(f"Image {row['image_filepath']} flagged for relabelling with probability {model_prob}")
                    df_relabelled.loc[index, "relabelled_by_model"] = "Y"
                    df_relabelled.loc[index, "true_label"] = class_0_name

                    if perform_image_move:
                        os.rename(os.path.join(image_dataset_path, class_1_name, file_name),
                                  os.path.join(image_dataset_path, class_0_name, file_name) )
                        df_relabelled.loc[index, "image_filepath"] = os.path.join(image_dataset_path, class_0_name, file_name)

                elif model_prob > conf_high:
                    logging.info(f"Image {row['image_filepath']} flagged for relabelling with probability {model_prob}")
                    df_relabelled.loc[index, "relabelled_by_model"] = "Y"
                    df_relabelled.loc[index, "true_label"] = class_1_name

                    if perform_image_move:
                        os.rename(os.path.join(image_dataset_path, class_0_name, file_name),
                                  os.path.join(image_dataset_path, class_1_name, file_name) )
                        df_relabelled.loc[index, "image_filepath"] = os.path.join(image_dataset_path, class_1_name,
                                                                                  file_name)
                else:
                    logging.info(f"Mismatched labels, but confidence too low for {row['image_filepath']}")
            else:
                pass

        return df_relabelled


    def run_cleaning_steps(self, data_source: str = None,
                           recursive_search: bool = True,) -> None:

        """
        Run the following data cleaning steps on image data, before passing to tensorflow:
            1. Correct the file extension if it does not reflect the image type.
            2. Convert images that are not RGB, to RGB.
            3. Remove invalid images (e.g. corrupted).
            4. Convert unsupported image type to jpeg.
            5. Remove exact duplicates (keeping the larger images).

        Args:
            data_source: Directory where image data is stored
            recursive_search: Search recursively or not from data_source
        """
        logging.info("\n")
        logging.info("RUNNING DATA CLEANING ...")
        # 1. correct file extensions
        logging.info("Correcting file extensions ...")
        self.correct_file_extensions(data_source, recursive_search=recursive_search)

        # 2. convert images to RGB, if not already
        path = Path(data_source)
        all_files = [p for p in path.rglob("*") if p.is_file()]
        self.convert_to_rgb(file_list=all_files)

        # 3. remove invalid images
        logging.info("\n\n")
        logging.info("Removing invalid images...")
        not_valid, not_supported = self.check_if_images_valid(data_source, recursive_search=recursive_search)
        self.remove_files(not_valid)
        # 4. convert not supported to jpeg
        logging.info("\n\n")
        logging.info("Converting not supported to jpeg...")
        self.convert_image_to_jpeg(file_list=not_supported)

        # 5. remove duplicates
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

    def run_cleaning_steps_model_labelling(self, data_source: str = None,
                                           recursive_search: bool = True,
                                           model_path: str = None,
                                           image_dataset_path: str = None,
                                           **kwargs
                                           ) -> pd.DataFrame:
        """
        Chain together standard cleaning steps with label checking using an already trained model.

        Args:
            data_source: Directory where image data is stored
            recursive_search: Search recursively or not from data_source
            model_path: File path to the .keras saved model.
            image_dataset_path: File path to the newly extracted images.
            **kwargs: additional arguments forwarded to relabel_images method

        """
        # 1. perform standard cleaning
        self.run_cleaning_steps(data_source=data_source, recursive_search=recursive_search)

        # 2. relabel results using trained model
        df_relabel = self.relabel_images(model_path=model_path,
                                         image_dataset_path=image_dataset_path,
                                         **kwargs)
        return df_relabel


if __name__ == "__main__":
    data_cleaning = DataCleaning()
    data_dir = "../../artifacts/data/images/"
    trained_model_path = "../../artifacts/training/model_mobnet_curated_data_extended.keras"
    # data_cleaning.run_cleaning_steps(data_source=data_dir)
    df_relabel = data_cleaning.run_cleaning_steps_model_labelling(data_source=data_dir,
                                                     model_path=trained_model_path,
                                                     image_dataset_path=data_dir,
                                                     perform_image_move = True
                                                                     )
    df_relabel.to_csv("../../artifacts/data/test_relabelling_with_model.csv")