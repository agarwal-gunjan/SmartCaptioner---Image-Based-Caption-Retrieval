import os
import pandas as pd
import random
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
from torchvision import transforms

# Load CLIP model and processor
from transformers import CLIPProcessor, CLIPModel

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# Define paths to the dataset and folders
dataset_path = 'D:/Gunjan/minor project/smartcaption/clip_data/dataset2.csv'
mountain_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/mountain'
beach_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/beaches'
desi_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/desi'
baddie_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/baddie'
bday_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/bday'
Cafe_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Cafe'
Cats_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Cats'
CHRISTMAS_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/CHRISTMAS'
couples_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/couples'
diwali_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/diwali'
Dogs_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Dogs'
Friends_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Friends'
Mirror_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Mirror'
Monsoon_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Monsoon'
Sky_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Skyyy'
Sunkissed_folder = 'D:/Gunjan/minor project/smartcaption/clip_data/Sunkissed'
df = pd.read_csv(dataset_path)

# Prepare category-based caption mappings
categories = [
    'Baddie', 'bday', 'beaches', 'Cafe', 'Cats', 'CHRISTMAS',
    'couples', 'desi', 'diwali', 'Dogs', 'Mirror', 'Monsoon',
    'mountain', 'Skyyy', 'Sunkissed','Friends']
category_captions = {category: [] for category in categories}

category_prompts = {
    'Baddie': "A stylish baddie aesthetic",
    'bday': "A cheerful birthday celebration",
    'beaches': "A scenic beach view",
    'Cafe': "A cozy cafe scene",
    'Cats': "A cute cat moment",
    'CHRISTMAS': "A festive Christmas celebration",
    'couples': "A romantic couple moment",
    'desi': "A vibrant cultural desi experience",
    'diwali': "A sparkling Diwali celebration",
    'Dogs': "A playful dog moment",
    'Mirror': "A mirror selfie aesthetic",
    'Monsoon': "A beautiful monsoon scene",
    'mountain': "A breathtaking mountain view",
    'Skyyy': "A picturesque sky view",
    'Sunkissed': "A sunkissed golden hour moment",
    'Friends' : "A group of friends makes life better"
}

# List of categories and matching prompts
categories = list(category_prompts.keys())
text_prompts = list(category_prompts.values())

# Populate category captions from the dataset
for _, row in df.iterrows():
    image_path = row[0]
    captions = row[1:6].tolist()  

    # Determine category based on image path
    matched = False
    for category in categories:
        if category.lower() in image_path.lower():
            category_captions[category].extend(captions)
            matched = True
            break

# Function to preprocess an input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict the category of an image
def predict_category(image_path):
    image = Image.open(image_path).convert("RGB")

    # Process the image and text prompts with CLIP
    inputs = clip_processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    # Calculate similarity
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = torch.cosine_similarity(image_features, text_features)

    # Get most similar category
    predicted_idx = torch.argmax(similarity).item()
    predicted_category = categories[predicted_idx]

    print(f"Predicted category: {predicted_category} (Score: {similarity[predicted_idx].item():.4f})")
    return predicted_category

# Function to get 5 random captions from a category
def get_captions_for_category(category):
    if category in category_captions:
        captions = category_captions[category]
        if len(captions) >= 5:
            return random.sample(captions, 5)
        elif captions:
            return captions  # return all if less than 5
    print(f"No captions found for category '{category}'")
    return []

# Main function to process an input image
def generate_captions(image_path):
    category = predict_category(image_path)
    captions = get_captions_for_category(category)

    print(f"\nGenerated Captions for '{category}':")
    for idx, cap in enumerate(captions, 1):
        print(f"{idx}. {cap}")

    return category, captions

def get_category_and_captions(image_path):
    category, captions = generate_captions(image_path)
    return category, captions
