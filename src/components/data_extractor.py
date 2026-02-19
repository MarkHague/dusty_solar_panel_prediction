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

    SAVE_PATH : str = '../../artifacts/data/'


class DataExtractor:

    def __init__(self):
        self.data_extract_config = DataExtractConfig()


    def get_image_data_df(self, query: str = None, pages: int = 1,
                          expected_label: str = "dirty") -> pd.DataFrame:
        """
        Returns a pd.Dataframe containing image urls from Oxylabs API given search params.

        Args:
            query: Search query for Google image search.
            pages: Number of pages to scrape. Each page is 100 images.
            expected_label: The label you expect given the query to Oxylabs image search API.

        """

        data_df = pd.DataFrame()

        for page in range(pages):
            # Structure payload.
            payload = {
                'source': 'google_search',
                'query': query,
                'parse': True,
                'pages': pages,
                'start_page': page,
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

            data_json = response.json()

            df = self.convert_json_to_csv(json_data= data_json, expected_label=expected_label)
            data_df = pd.concat([data_df, df])

        data_df.index = range(len(data_df))
        return data_df


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

        return df_out

    def extract_images(self, image_data: pd.DataFrame = None) -> pd.DataFrame | None:

        """
        Extracts (saves to disk) image files from a list of image urls.
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

                    filepath = os.path.join(self.data_extract_config.SAVE_PATH, "images", row['label'],
                                            row['query'] + "_" + str(index) + img_ext)
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







