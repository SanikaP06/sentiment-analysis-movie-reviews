#  Sentiment Analyzer for Movie Reviews

An advanced NLP-powered web app to analyze movie reviews using multiple state-of-the-art transformer models. This project uses Hugging Face models (BERT, RoBERTa, and DistilRoBERTa) to provide in-depth sentiment analysis, star rating prediction, and emotion detection—all wrapped in a visually interactive Streamlit UI.

##  Features

-  **Single Review Analysis**: Get insights into the sentiment, star rating, and emotions of an individual movie review.
-  **Batch Review Analysis**: Analyze multiple reviews at once and get a detailed summary with visualizations.
-  **Model Comparison**: Compare the outputs of different models on the same input.
-  **Linguistic Feature Explorer**: Dive into key textual features like word count, exclamations, caps usage, and sentiment cues.
-  **Interactive Visualizations**: Dynamic Plotly charts for ratings, sentiment breakdown, emotions, and text feature radar.
-  **CSV Export**: Download batch results for further analysis or reporting.

##  Models Used

- **BERT** (`nlptown/bert-base-multilingual-uncased-sentiment`): Predicts star ratings from 1 to 5.
- **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`): Predicts sentiment as Positive, Neutral, or Negative.
- **DistilRoBERTa** (`j-hartmann/emotion-english-distilroberta-base`): Detects emotional tone (e.g., joy, sadness, anger, etc.)

  ## 📂 How to Run Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SanikaP06/sentiment-analysis-movie-reviews
2. **Install the dependencies**

   ```bash
   pip install -r requirements.txt


