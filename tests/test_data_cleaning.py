from src.components.data_cleaning import DataCleaning
data_cleaning = DataCleaning()


class TestDataCleaning:
    """Test common data cleaning functions """

    def test_find_exact_duplicates(self, test_images) -> None:

        image_paths = test_images.glob('image_?.jpg')
        # image_paths = [os.path.join(base_path+"/tests",f) for f in image_paths ]

        duplicates = data_cleaning.find_exact_duplicates(image_paths)

        assert str(duplicates[0]) == str(test_images / "image_1.jpg")


    def test_check_if_images_valid(self, test_images) -> None:

        not_valid, not_supported = data_cleaning.check_if_images_valid(test_images)

        assert not_valid[0] == str(test_images / "image_corrupted.jpg")
        assert not_supported[0] == str(test_images / "image_webp.webp")

    def test_correct_file_extensions(self, test_images):

        assert (test_images / "image_incorrect_extension.jpg.webp").exists()
        data_cleaning.correct_file_extensions(test_images)

        assert (test_images / "image_incorrect_extension.jpg.png").exists()