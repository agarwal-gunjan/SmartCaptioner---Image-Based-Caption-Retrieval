# SmartCaptioner---Image-Based-Caption-Retrieval
A CLIP-based image caption retrieval system suggesting creative social media captions. Includes a Django web app and Kodular mobile app interface.

## Overview
- Built using a custom dataset consisting of 1600 images across 16 categories with total 8000 captions.
- Uses OpenAI's CLIP model for zero-shot image-text similarity.
- Suggests top 5 captions based on predicted image category

## Interfaces
- A Django web application for uploading and retrieving top 5 captions with option to regenerate captions depending upon the user expectations.
- A Kodular-based Android mobile app to generate captions on the go.

## Tech Stack
- Python, CLIP model, Django
- Kodular (for android app)
- Custom dataset creation and preprocessing

## Example Output
> Upload an  image -> Get 5 unique, social media worthy captions or regenerate based on the expectations.
