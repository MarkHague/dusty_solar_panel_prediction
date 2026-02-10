from src.utils import find_exact_duplicates
import os

class TestDataCleaning:
    """Test common utils functions """

    def test_find_exact_duplicates(self) -> None:

        base_path = os.getcwd()
        image_paths = ['image_1.jpg','image_2.jpg','image_3.jpg']
        image_paths = [os.path.join(base_path+"/tests",f) for f in image_paths ]

        duplicates = find_exact_duplicates(image_paths)

        # TODO add checking if input is invalid - should print warning

        assert duplicates[0] == image_paths[0]