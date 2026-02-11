from src.components.data_cleaning import DataCleaning
import os
data_cleaning = DataCleaning()

class TestDataCleaning:
    """Test common data cleaning functions """

    def test_find_exact_duplicates(self) -> None:


        image_paths = ['image_1.jpg','image_2.jpg','image_3.jpg']
        # image_paths = [os.path.join(base_path+"/tests",f) for f in image_paths ]

        duplicates = data_cleaning.find_exact_duplicates(image_paths)

        assert duplicates[0] == image_paths[0]


    def test_check_if_images_valid(self) -> None:
        data_dir = "./"

        not_valid, not_supported = data_cleaning.check_if_images_valid(data_dir)

        assert not_valid[0] == "./image_corrupted.jpg"
        assert not_supported[0] == "./image_webp.webp"