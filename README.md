# Dirty Solar Panel Prediction
 
--This page is under construction--

## Overview
This project aims to automatically detect when detect when solar require cleaning using image classification. The build-up of dusty, pollen and other contaminants can significantly impact the performance of solar installations, making timely cleaning essential.

## Data
An initial model was trained on a curated subset of this [kaggle dataset](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection). This dataset was then extended by scraping images using the [Oxylabs API](https://developers.oxylabs.io/scraping-solutions/web-scraper-api/targets/google/search/image-search). 

## Model
In order to balance accuracy and inference performance, the [MobileNetV2 model](https://huggingface.co/papers/1801.04381) was used. 