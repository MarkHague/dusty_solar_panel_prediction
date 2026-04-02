from src.components.data_extractor import DataExtractor
data_extractor = DataExtractor()


def make_json(organic_results, start_page=1, label="dirty"):
    """
    Make example valid-looking Oxylabs API response dict.
    """
    return {
        "job": {
            "created_at": "2024-01-01T00:00:00Z",
            "query": "dusty solar panels",
            "start_page": start_page,
        },
        "results": [{"content": {"results": {"organic": organic_results}}}],
    }

class TestDataExtract:
    """
    Test data extraction methods.
    """

    def test_convert_json_to_df_happy_path(self):
        organic = [
            {"high_res_image": "http://example.com/img1.jpg", "domain": "example.com", "title": "A solar panel"},
            {"high_res_image": "http://example.com/img2.jpg", "domain": "example.com", "title": "Another solar panel"},
        ]
        json_data = make_json(organic, start_page=2)
        df = data_extractor.convert_json_to_df(json_data=json_data, expected_label="dirty")

        expected_columns = ['date_extracted', 'label', 'domain', 'image_url', 'image_title', 'page_number', 'query']
        assert len(df) == 2
        assert df.columns.tolist() == expected_columns
        assert df.iloc[0]["image_url"] == "http://example.com/img1.jpg"
        assert df.iloc[0]["domain"] == "example.com"
        assert df.iloc[0]["image_title"] == "A solar panel"
        assert df.iloc[0]["label"] == "dirty"
        assert df.iloc[0]["query"] == "dusty solar panels"
        assert df.iloc[0]["date_extracted"] == "2024-01-01T00:00:00Z"
        assert df.iloc[0]["page_number"] == 2


