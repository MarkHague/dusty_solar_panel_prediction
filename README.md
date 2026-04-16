# Dirty Solar Panel Prediction
 
--This page is under construction--

## Overview
This project aims to automatically detect when detect when solar require cleaning using image classification. The build-up of dust, pollen and other contaminants can significantly impact the performance of solar installations, making timely cleaning essential.

## Data
### Initial Dataset
An initial model was trained on a curated subset of this [kaggle dataset](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection). The original dataset did not seem to have been assessed for quality, and contained many images that had one or more of the following problems:
- Solar panel was too far away to see if its clean/dirty (if a human can't classify the image, how can we expect the ML algorithm to?).
- Unrealistic images of "clean" panels - usually promotional "photoshop-style" images that are very far from what the system would realistically see in production.
- Images where it was genuinely unclear to me if it was clean or dirty due to bad lighting, a poor angle of the photo etc.  

### Extended Dataset
Since I was not able to hit my target performance with the initial curated dataset, I set about gathering more data. Here I scraped images using the [Oxylabs API](https://developers.oxylabs.io/scraping-solutions/web-scraper-api/targets/google/search/image-search). This is implemented by `src/components/data_extractor.py`

### Image Data Cleaning
Apart from the manual cleaning (curation) I had to do initially, the system performs a number of automated data cleaning steps, including:
1. Correcting file extensions
2. Converting non-RGB images to RGB mode (needed for training)
3. Removing corrupted or otherwise unreadable images
4. Converting unsupported image types (e.g .webp) to .jpeg
5. Removing duplicates using the [imagehash library](https://pypi.org/project/ImageHash/)

## Model
In order to balance performance and inference latency, the [MobileNetV2 model](https://huggingface.co/papers/1801.04381) is the current default model. Using the model as a feature extractor, I started with the pre-trained (ImageNet) model, only training the top classification layers.

## Project Structure
At a high-level, the code is structured into two pipelines located in `src/pipeline`:
- train_pipeline - input data source --> output a trained model artifact.
- inference_pipeline - input model artifact --> output Flask API with a predict endpoint.  