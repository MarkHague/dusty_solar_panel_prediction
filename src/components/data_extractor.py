import pandas as pd
import os
import requests
from PIL import Image, UnidentifiedImageError
import io
import random

from dataclasses import dataclass
from src.exception import CustomException
import sys
from src.logger import logging
from dotenv import load_dotenv
load_dotenv()

@dataclass
class DataExtractConfig:
    USERNAME: str = os.getenv('OXYLABS_USERNAME')
    PASSWORD: str = os.getenv('OXYLABS_PASSWORD')
    API_URL: str = 'https://realtime.oxylabs.io/v1/queries'

    DATA_DIR : str = os.path.normpath(
        os.path.join( os.path.dirname(__file__),'..', '..', 'artifacts', 'data')
    )

class DataExtractor:

    def __init__(self):
        self.data_extract_config = DataExtractConfig()


    def get_image_data_df(self, query: str = None, pages: int = 1,
                          expected_label: str = "dirty") -> pd.DataFrame:
        """
        Retrieve image data from Oxylabs API given search params.

        Args:
            query: Search query for Google image search.
            pages: Number of pages to scrape. Each page is 100 images.
            expected_label: The label you expect given the query to Oxylabs image search API.
        Returns:
            A pandas DataFrame with the image urls and other useful metadata related to each image.

        """

        data_df = pd.DataFrame()

        for page in range(pages):
            # Structure payload.
            payload = {
                'source': 'google_search',
                'query': query,
                'parse': True,
                'pages': pages,
                'start_page': page+1,
                'context': [
                    {'key': 'tbm', 'value': 'isch'},
                ],
            }

            # Get response.
            response = requests.post(
                self.data_extract_config.API_URL,
                auth=(self.data_extract_config.USERNAME, self.data_extract_config.PASSWORD),
                json=payload,
            )

            if response.status_code == 200:
                data_json = response.json()

                df = self.convert_json_to_csv(json_data= data_json, expected_label=expected_label)
                data_df = pd.concat([data_df, df])

                data_df.index = range(len(data_df))

            else:
                logging.info(f"ERROR. Image data extraction failed with response code {response.status_code}, on page {page}")

        if len(data_df) > 0:
            return data_df.drop_duplicates(subset=["image_url"])


    def convert_json_to_csv(self, json_data : dict = None, expected_label: str = "dirty") -> pd.DataFrame:
        """
        Convert json image data scraped from Oxylabs API to csv.

        Args:
            json_data: Response from Oxylabs API call.
            expected_label: The label you expect given the query to Oxylabs image search API.
        """
        date_extracted = json_data["job"]["created_at"]
        query = json_data["job"]["query"]
        label = expected_label
        page_num = json_data["job"]['start_page']

        json_data_res = json_data["results"][0]["content"]["results"]["organic"]
        column_names = ['date_extracted', 'label', 'domain', 'image_url', 'image_title', 'page_number',
                        'query']

        df_out = pd.DataFrame(columns=column_names)

        try:
            for img in json_data_res:
                new_row = {'date_extracted': date_extracted,
                           'label': label,
                           'domain': img["domain"],
                           'image_url': img["high_res_image"],
                           'image_title': img["title"],
                           'page_number': page_num,
                           'query': query
                           }

                df_out.loc[len(df_out)] = new_row
        except Exception as e:
            raise CustomException(e, sys)


        return df_out

    def save_images(self, image_data: pd.DataFrame = None) -> pd.DataFrame | None:
        """
        Saves image files to disk using an input dataframe containing the image urls.

        Args:
            image_data: pandas Dataframe created by get_image_data_df method.

        Returns:
            pandas Dataframe containing any abnormal response codes (!= 200), and associated data.
            If no abnormal response codes are present returns None.
        """
        cols = ["image_name","image_url", "response_code"]
        abnormal_response = pd.DataFrame(columns=cols)

        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36'
        ]

        for index, row in image_data.iterrows():

            headers = {
                "User-Agent": random.choice(user_agents),
                "Accept": "image/png, image/jpeg, image/*,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": row['image_url'],
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": 'document',
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Site": "none",
            }

            response = requests.get(row['image_url'], headers=headers, stream=True)

            if response.status_code == 200:

                try:
                    img = Image.open(io.BytesIO(response.content))
                    img_ext = "." + img.format.lower()

                    filepath = os.path.join(self.data_extract_config.DATA_DIR, "images", row['label'],
                                            row['query'] + "_" + str(index) + img_ext)
                    # make base save path if not already done
                    base_path = os.path.join(self.data_extract_config.DATA_DIR, "images", row['label'])
                    if not os.path.isdir(base_path):
                        os.makedirs(base_path)

                    with open(filepath, 'wb') as file:
                        file.write(response.content)
                        logging.info(f"Writing to disk: {file}")
                except UnidentifiedImageError:
                    logging.info(f'Invalid image {row["image_url"]}')
            else:
                print(f'Image not retrievable with status code: {response.status_code}')
                error_data = {"image_name": row['query'] + "_" + str(index),
                              "image_url": row['image_url'],
                              "response_code": response.status_code}
                abnormal_response[len(abnormal_response)] = error_data

        if len(abnormal_response) > 0:
            return abnormal_response
        else:
            return None

    def save_df_to_csv(self, dataframe: pd.DataFrame = None, file_name: str = "data.csv") -> None:
        file_path = os.path.join(self.data_extract_config.DATA_DIR, file_name)
        dataframe.to_csv(file_path)

    def run_extraction(self, queries: dict[list] = None, pages_per_query = 2,
                       filename_output_csv_data: str = 'data.csv'):
        """
        Run all steps required to extract image data using the Oxylabs API.

        Args:
            queries: A dict where each label is a key, and its values are a list of search queries.
            pages_per_query: Number of pages per query. Each page contains 100 images.
            filename_output_csv_data: File name of the output data containing all the image urls and meta-data.
        """
        logging.info(f"RUNNING IMAGE EXTRACTION WITH QUERY DICT: {queries}")
        # 1. Get the image urls and other meta-data for each query and combine
        df_list = []
        for label, query_list in queries.items():
            for qry in query_list:
                df_list.append(self.get_image_data_df(query = qry, pages=pages_per_query, expected_label=label) )

        df_all_data = pd.concat(df_list).drop_duplicates(subset=['image_url'] )
        df_all_data.index = range(len(df_all_data))
        # write data to DATA_DIR
        logging.info(f"Saving image data csv file as {filename_output_csv_data}")
        self.save_df_to_csv(df_all_data, file_name=filename_output_csv_data)

        # 2. Extract images from df_all_data
        abnormal_responses = self.save_images(df_all_data)

        return abnormal_responses









