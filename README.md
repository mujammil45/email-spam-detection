
Email Spam Detection

A Machine Learning project that detects whether an email/message is Spam or Not Spam.

Features

- Data preprocessing
- TF-IDF text vectorization
- Logistic Regression model
- Model evaluation
- Saved ML model
- Streamlit web application
- Spam/Not Spam prediction

Dataset

The project uses an SMS/Email spam dataset containing spam and normal (ham) messages.

Model

Logistic Regression is used for classification, with TF-IDF for converting text into numerical features.

Results

- Accuracy: 96.71%
- Precision: 98.99%
- Recall: 74.81%
- F1 Score: 85.22%
- ROC-AUC: 98.92%

How to Run

pip install -r requirements.txt
python train_model.py
streamlit run app.py

Then open the Streamlit URL shown in the terminal.

## ## 📸 Screenshots

### Spam Prediction

![Spam Prediction 1](screenshots/spam_prediction_1.png)

![Spam Prediction 2](screenshots/spam_prediction_2.png)

### Not Spam Prediction

![Not Spam Prediction 1](screenshots/not_spam_prediction_1.png)

![Not Spam Prediction 2](screenshots/not_spam_prediction_2.png)


