Food Allergy Risk Estimation using Food Image Recognition
Project Overview
----------------
This project is a web-based application that analyzes food images and estimates potential allergy risks. The system identifies ingredients present in the food using a trained deep learning model and checks them against known allergens selected by the user. The goal is to help users quickly understand whether a food item might trigger their allergies before consuming it.

Key Features
------------------
.Upload food images through a web interface
.Deep learning model for food/spice recognition
.User allergy selection system
.Ingredient–allergen matching engine
.Real-time allergy risk detection
.Web interface built with Flask

Technologies Used
--------------------
Python
Flask
TensorFlow / Keras
HTML, CSS
JSON databases (ingredients & allergens)
MobileNetV2 (for image classification)

How the System Works
--------------------
The user uploads an image of food.
The image is preprocessed (resized, normalized).
The trained deep learning model predicts the ingredients/spices present in the image.
The system compares detected ingredients with allergens selected by the user.
If a match is found, the system warns the user about the allergy risk.



How to Run the Project
------------------------
1. Clone the repository
git clone https://github.com/fai-404/food-allergy-risk-estimation.git
2. Install dependencies
pip install -r requirements.txt
3. Run the Flask server
python app.py

Machine Learning Model
The system uses a convolutional neural network based on MobileNetV2 architecture to classify food ingredients from images. The model was trained on a dataset of spice and ingredient images and exported as a .h5 weights file.


