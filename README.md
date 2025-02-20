# Skin & Hair Disease Prediction App

This project is a Streamlit-based web application that predicts hair and skin diseases using deep learning models. The app takes an image as input and provides disease predictions along with potential cures.

## Features
- **Hair Disease Prediction**: Uses a trained VGG19 model (`VGG19-Final.h5`) to classify hair diseases.
- **Skin Cancer Detection**: Uses a trained model (`skin_cancer_detection_model.h5`) to detect different types of skin cancer.
- **Image Upload**: Users can upload images in JPG, PNG, or JPEG format.
- **Cure Suggestions**: Provides potential treatment suggestions for detected hair diseases.

## Technologies Used
- Python
- Streamlit
- TensorFlow/Keras
- NumPy
- PIL (Pillow)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/skin-hair-disease-prediction.git
   cd skin-hair-disease-prediction
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Ensure the model files (`VGG19-Final.h5` and `skin_cancer_detection_model.h5`) are present in the project directory.

## Usage

Run the Streamlit application:
```sh
streamlit run app.py
```

### Steps to Use
1. Choose the disease type (Hair Disease or Skin Cancer).
2. Upload an image of the affected area.
3. View the predicted disease and suggested cure.

## Project Structure
```
skin-hair-disease-prediction/
│-- VGG19-Final.h5
│-- skin_cancer_detection_model.h5
│-- app.py
│-- requirements.txt
│-- README.md
```

## Requirements
Ensure you have the required Python libraries installed. The dependencies are listed in `requirements.txt`:
```sh
streamlit
tensorflow
numpy
pillow
```

## License
This project is licensed under the MIT License.

## Acknowledgments
- The deep learning models were trained using TensorFlow and Keras.
- Streamlit is used for creating the web interface.

## Author
Your Name – [Your GitHub](https://github.com/kshitij730)

