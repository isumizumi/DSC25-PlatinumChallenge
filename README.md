# Building a Sentiment Analysis Engine & API

## Challenge: Membuat API untuk Analisis Sentimen dan Laporan Analisis Data Berdasarkan Sentimen

### Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Reports](#reports)
  - [Manual Sentiment Calculation Report](#manual-sentiment-calculation-report)
  - [Data Analysis Report](#data-analysis-report)

---

## Project Overview
This project involves building a sentiment analysis API powered by Neural Network and LSTM (Long Short-Term Memory) deep learning model. The API processes text input and predicts its sentiment as `positive`, `neutral`, or `negative`. Additionally, it includes data analysis reports to provide insights into sentiment distribution and trends.

---

## Features
- **Sentiment Analysis API**:
  - Accepts text input via HTTP POST requests.
  - Provides sentiment predictions (Positive, Neutral, Negative).
- **Preprocessing Pipeline**:
  - Cleanses text using custom functions.
  - Normalizes text with slang correction and abusive word handling.
- **Data Augmentation**:
  - Replaces words with synonyms to improve model performance.
- **Custom Trained Model**:
  - LSTM model fine-tuned on a dataset with Indonesian text.
- **Visualization and Reports**:
  - Produces detailed analysis of sentiment distribution.

---

## Technologies Used
- **Frameworks & Libraries**:
  - [Flask](https://flask.palletsprojects.com/) for building the LSTM API.
  - [Gradio](https://www.gradio.app/) for building the RNN API.
  - [TensorFlow/Keras](https://www.tensorflow.org/) for machine learning models.
  - [Scikit-learn](https://scikit-learn.org/) for preprocessing and evaluation.
  - [Matplotlib & Seaborn](https://seaborn.pydata.org/) for data visualization.
- **Other Tools**:
  - [FastText](https://fasttext.cc/) for word embedding.
  - [Git](https://git-scm.com/) for version control.

---

## Project Team

### Contributors

1. **[Siti Aisyah Ramadhani](https://github.com/siti-aisyah19/platinum/tree/main/platinum)**  
   - Contributions: Data cleansing, Manual Sentiment Calculation, Statistical & EDA Methods and Visualization.

2. **[Nur'Adilah Firdaus](https://github.com/nuradilahf/feature-extraction-revision/tree/main)**  
   - Contributions: Feature Extraction.

3. **[Devan Alingga Abda](https://github.com/Devanaa1999/NeuralNetwork/tree/main)**   
   - Contributions: Training Neural Network Model, API for Neural Network Model

4. **[Isumi Karinaningsih](https://github.com/isumizumi/DSC25-PlatinumChallenge)**    
   - Contributions: Trello for project management, Training LSTM Model & Visualization, API & Swagger for LSTM Model, Github Documentation, Slide Presentation.

### Project Management

[Trello](https://trello.com/b/6o5SKrfP/datascienceplatinum-challenge)

---

## Installation
Follow these steps to set up the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-api.git
   cd sentiment-analysis-api

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Download pretrained embedding**:
   Download the Indonesian FastText embedding (cc.id.300.vec) and place it in the resources directory.

5. **Prepare data**:
   Ensure clean.csv, new_kamusalay.csv, and abusive.csv are in the project directory.

6. **Train the model**:
   ```bash
   python train_model.py

7. **Run the application**:
   ```bash
   python app.py

---

## Usage

### 1. LSTM Sentiment Prediction
You can use the LSTM API to predict the sentiment of a given text.

#### Predict Sentiment using LSTM (Text)
- **Endpoint:** `/api/predict/lstm`
- **Method:** POST  
- **Request Body:**
   ```json
   {
       "text": "Rasa syukur, cukup"
   }
- **Responses:**
    ```bash
    {
    "sentiment": "neutral",
    "confidence": 0.70
    }

#### Predict Sentiment using LSTM (File)
- **Endpoint:** `/api/predict-file-lstm`
- **Method:** POST  
- **Example Request:**
   1. Open a tool like Postman or Curl.
   2. Attach the file in the request body under the key `"file"`.

##### Curl Command:

```
curl -X POST -F "file=@your_file.txt" http://127.0.0.1:5000/predict-file-lstm
```

##### Postman:
1. Set the request type to POST.
2. Enter the URL http://127.0.0.1:5000/predict-file-lstm.
3. Go to the "Body" tab, choose "form-data".
4. Add a key named "file" and attach your .txt file (You can download this file from data folder: uji-teks.txt).

### 2. RNN Sentiment Prediction
RNN Sentiment Analysis API using Gradio by executing the `rnn-api.py` script.

    ```
    python rnn-api.py
    ```

---

## API Documentation 

LSTM Sentiment Analysis API using Swagger
You can access detailed API documentation through Swagger. After running the application, navigate to:

    ```
    http://127.0.0.1:5000/swagger
    ```

---

## Reports

### Manual Sentiment Calculation Report
You can access detailed calculation report:
[Manual Sentiment Calculation Report](https://docs.google.com/document/d/1X9n2kYE_QY9cRNHhnWrcg4FS_dxroT5l/edit?usp=sharing&ouid=101498873662196123612&rtpof=true&sd=true)

### Data Analysis Report
You can access detailed data analysis report:
[Data Analysis Report](https://colab.research.google.com/drive/1BMqlaSWA8RTgwbVfmQLimmnl7sMcBZff?usp=sharing)