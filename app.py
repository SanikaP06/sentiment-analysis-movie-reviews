import os
os.environ["TF_KERAS"] = "1"

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import re
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from io import BytesIO
import json

#Pageconfig
st.set_page_config(
    page_title=" Advanced Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Load multiple models for comparison
@st.cache_resource
def load_models():
    models = {}
    
    # Model 1: BERT-based sentiment (original)
    model_name_1 = "nlptown/bert-base-multilingual-uncased-sentiment"
    models['BERT_Sentiment'] = {
        'tokenizer': AutoTokenizer.from_pretrained(model_name_1),
        'model': AutoModelForSequenceClassification.from_pretrained(model_name_1),
        'type': 'rating'
    }
    
    # Model 2: RoBERTa for sentiment classification
    model_name_2 = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    models['RoBERTa_Sentiment'] = {
        'tokenizer': AutoTokenizer.from_pretrained(model_name_2),
        'model': AutoModelForSequenceClassification.from_pretrained(model_name_2),
        'type': 'sentiment'
    }
    
    # Model 3: Emotion classification
    models['Emotion_Pipeline'] = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        return_all_scores=True
    )
    
    return models

def analyze_sentiment_bert(text, model_data):
    """Analyze sentiment using BERT model"""
    inputs = model_data['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_data['model'](**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item() + 1
    
    return predicted_class, probs[0].numpy()

def analyze_sentiment_roberta(text, model_data):
    """Analyze sentiment using RoBERTa model"""
    inputs = model_data['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_data['model'](**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    
    labels = ['Negative', 'Neutral', 'Positive']
    scores = probs[0].numpy()
    predicted_class = np.argmax(scores)
    
    return labels[predicted_class], scores

def analyze_emotions(text, emotion_pipeline):
    """Analyze emotions in the text"""
    results = emotion_pipeline(text)
    return results[0]

def extract_features(text):
    """Extract linguistic features from text"""
    features = {}
    
    # Basic stats
    features['word_count'] = len(text.split())
    features['char_count'] = len(text)
    features['sentence_count'] = len(re.split(r'[.!?]+', text))
    features['avg_word_length'] = np.mean([len(word) for word in text.split()])
    
    # Sentiment indicators
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'awesome', 'brilliant']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'boring', 'stupid', 'annoying', 'anger','sad']
    
    text_lower = text.lower()
    features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
    features['negative_word_count'] = sum(1 for word in negative_words if word in text_lower)
    
    # Punctuation analysis
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return features

def create_sentiment_visualization(bert_probs, roberta_scores, emotions):
    """Create comprehensive sentiment visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BERT Rating Distribution', 'RoBERTa Sentiment', 'Emotion Analysis', 'Confidence Comparison'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # BERT Rating Distribution
    fig.add_trace(
        go.Bar(x=[f"{i+1} Star" for i in range(5)], y=bert_probs, name="BERT Rating",
               marker_color=['#ff4444', '#ff8800', '#ffaa00', '#88aa00', '#44aa00']),
        row=1, col=1
    )
    
    # RoBERTa Sentiment
    roberta_labels = ['Negative', 'Neutral', 'Positive']
    fig.add_trace(
        go.Bar(x=roberta_labels, y=roberta_scores, name="RoBERTa Sentiment",
               marker_color=['#ff4444', '#ffaa00', '#44aa00']),
        row=1, col=2
    )
    
    # Emotions
    emotion_labels = [item['label'] for item in emotions]
    emotion_scores = [item['score'] for item in emotions]
    fig.add_trace(
        go.Bar(x=emotion_labels, y=emotion_scores, name="Emotions",
               marker_color=px.colors.qualitative.Set3),
        row=2, col=1
    )
    
    # Confidence comparison
    max_bert = max(bert_probs)
    max_roberta = max(roberta_scores)
    max_emotion = max(emotion_scores)
    
    fig.add_trace(
        go.Bar(x=['BERT', 'RoBERTa', 'Emotion'], y=[max_bert, max_roberta, max_emotion],
               name="Max Confidence", marker_color=['#3498db', '#e74c3c', '#2ecc71']),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Sentiment Analysis")
    return fig

def create_feature_radar_chart(features):
    """Create radar chart for linguistic features"""
    categories = ['Word Count', 'Positive Words', 'Negative Words', 'Exclamations', 'Questions', 'Caps Ratio']
    
    # Normalize features for radar chart
    normalized_features = [
        min(features['word_count'] / 100, 1),  # Normalize to 0-1
        min(features['positive_word_count'] / 5, 1),
        min(features['negative_word_count'] / 5, 1),
        min(features['exclamation_count'] / 5, 1),
        min(features['question_count'] / 5, 1),
        features['caps_ratio']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=categories,
        fill='toself',
        name='Text Features',
        line_color='rgb(255,140,0)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Linguistic Feature Analysis"
    )
    
    return fig

def batch_analyze_reviews(reviews_text):
    """Analyze multiple reviews at once"""
    reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
    results = []
    
    models = st.session_state.models
    
    for i, review in enumerate(reviews):
        if len(review) < 5:  # Skip very short reviews
            continue
            
        # BERT Analysis
        bert_rating, bert_probs = analyze_sentiment_bert(review, models['BERT_Sentiment'])
        
        # RoBERTa Analysis
        roberta_sentiment, roberta_scores = analyze_sentiment_roberta(review, models['RoBERTa_Sentiment'])
        
        # Features
        features = extract_features(review)
        
        results.append({
            'Review': review[:100] + '...' if len(review) > 100 else review,
            'BERT_Rating': bert_rating,
            'RoBERTa_Sentiment': roberta_sentiment,
            'Word_Count': features['word_count'],
            'Positive_Words': features['positive_word_count'],
            'Negative_Words': features['negative_word_count'],
            'BERT_Confidence': max(bert_probs),
            'RoBERTa_Confidence': max(roberta_scores)
        })
    
    return pd.DataFrame(results)

# Initialize session state
if 'models' not in st.session_state:
    with st.spinner("Loading AI models... This may take a moment."):
        st.session_state.models = load_models()

# Main app
st.markdown('<h1 class="main-header">🎬 Advanced Movie Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎛️ Analysis Options")
analysis_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ["Single Review Analysis", "Batch Review Analysis", "Model Comparison", "Feature Explorer"]
)

# Main content tabs
if analysis_mode == "Single Review Analysis":
    st.header("📝 Single Review Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter your movie review:", height=150, placeholder="Type your movie review here...")
        
        if st.button("🔍 Analyze Review", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing your review..."):
                    models = st.session_state.models
                    
                    # BERT Analysis
                    bert_rating, bert_probs = analyze_sentiment_bert(user_input, models['BERT_Sentiment'])
                    
                    # RoBERTa Analysis
                    roberta_sentiment, roberta_scores = analyze_sentiment_roberta(user_input, models['RoBERTa_Sentiment'])
                    
                    # Emotion Analysis
                    emotions = analyze_emotions(user_input, models['Emotion_Pipeline'])
                    
                    # Feature Extraction
                    features = extract_features(user_input)
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("BERT Rating", f"{bert_rating} ⭐", f"{max(bert_probs):.2%} confidence")
                    
                    with col2:
                        st.metric("RoBERTa Sentiment", roberta_sentiment, f"{max(roberta_scores):.2%} confidence")
                    
                    with col3:
                        top_emotion = max(emotions, key=lambda x: x['score'])
                        st.metric("Dominant Emotion", top_emotion['label'], f"{top_emotion['score']:.2%}")
                    
                    with col4:
                        st.metric("Word Count", features['word_count'], f"Avg: {features['avg_word_length']:.1f} chars/word")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = create_sentiment_visualization(bert_probs, roberta_scores, emotions)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig_radar = create_feature_radar_chart(features)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Detailed analysis
                    st.subheader("🔍 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sentiment Breakdown:**")
                        st.write(f"- **BERT Rating:** {bert_rating}/5 stars")
                        st.write(f"- **RoBERTa Sentiment:** {roberta_sentiment}")
                        st.write(f"- **Top Emotion:** {top_emotion['label']} ({top_emotion['score']:.2%})")
                        
                        if bert_rating <= 2:
                            st.error("😞 Overall: Negative Review")
                        elif bert_rating == 3:
                            st.warning("😐 Overall: Neutral Review")
                        else:
                            st.success("😊 Overall: Positive Review")
                    
                    with col2:
                        st.write("**Linguistic Features:**")
                        st.write(f"- **Words:** {features['word_count']}")
                        st.write(f"- **Sentences:** {features['sentence_count']}")
                        st.write(f"- **Positive indicators:** {features['positive_word_count']}")
                        st.write(f"- **Negative indicators:** {features['negative_word_count']}")
                        st.write(f"- **Exclamations:** {features['exclamation_count']}")
                        st.write(f"- **Questions:** {features['question_count']}")
            else:
                st.warning("Please enter a review to analyze.")

elif analysis_mode == "Batch Review Analysis":
    st.header("📚 Batch Review Analysis")
    
    st.info("Enter multiple reviews (one per line) to analyze them all at once.")
    
    batch_input = st.text_area(
        "Enter multiple reviews (one per line):",
        height=200,
        placeholder="Review 1...\nReview 2...\nReview 3..."
    )
    
    if st.button("🔍 Analyze All Reviews", type="primary"):
        if batch_input.strip():
            with st.spinner("Analyzing all reviews..."):
                results_df = batch_analyze_reviews(batch_input)
                
                if not results_df.empty:
                    st.subheader("📊 Batch Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_rating = results_df['BERT_Rating'].mean()
                        st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
                    
                    with col2:
                        positive_ratio = (results_df['RoBERTa_Sentiment'] == 'Positive').sum() / len(results_df)
                        st.metric("Positive Reviews", f"{positive_ratio:.1%}")
                    
                    with col3:
                        avg_words = results_df['Word_Count'].mean()
                        st.metric("Avg Word Count", f"{avg_words:.0f}")
                    
                    with col4:
                        high_conf = (results_df['BERT_Confidence'] > 0.8).sum() / len(results_df)
                        st.metric("High Confidence", f"{high_conf:.1%}")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(results_df, x='BERT_Rating', title='Rating Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(results_df, names='RoBERTa_Sentiment', title='Sentiment Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No valid reviews found. Please check your input.")
        else:
            st.warning("Please enter some reviews to analyze.")

elif analysis_mode == "Model Comparison":
    st.header("🔬 Model Comparison")
    
    st.write("Compare how different AI models analyze the same review.")
    
    comparison_input = st.text_area("Enter a review for model comparison:", height=100)
    
    if st.button("🔍 Compare Models", type="primary"):
        if comparison_input.strip():
            with st.spinner("Running comparison analysis..."):
                models = st.session_state.models
                
                # Get results from all models
                bert_rating, bert_probs = analyze_sentiment_bert(comparison_input, models['BERT_Sentiment'])
                roberta_sentiment, roberta_scores = analyze_sentiment_roberta(comparison_input, models['RoBERTa_Sentiment'])
                emotions = analyze_emotions(comparison_input, models['Emotion_Pipeline'])
                
                # Create comparison table
                comparison_data = {
                    'Model': ['BERT (Rating)', 'RoBERTa (Sentiment)', 'Emotion Model'],
                    'Prediction': [f"{bert_rating} stars", roberta_sentiment, max(emotions, key=lambda x: x['score'])['label']],
                    'Confidence': [f"{max(bert_probs):.2%}", f"{max(roberta_scores):.2%}", f"{max(emotions, key=lambda x: x['score'])['score']:.2%}"],
                    'Model Type': ['BERT-based', 'RoBERTa-based', 'DistilRoBERTa-based']
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                st.subheader("📊 Model Comparison Results")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Detailed breakdown
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🎯 BERT Rating Model")
                    st.write("**Prediction:** {} stars".format(bert_rating))
                    st.write("**Confidence:** {:.2%}".format(max(bert_probs)))
                    st.bar_chart(pd.DataFrame({'Rating': [1,2,3,4,5], 'Probability': bert_probs}))
                
                with col2:
                    st.subheader("🎯 RoBERTa Sentiment Model")
                    st.write("**Prediction:** {}".format(roberta_sentiment))
                    st.write("**Confidence:** {:.2%}".format(max(roberta_scores)))
                    sentiment_df = pd.DataFrame({'Sentiment': ['Negative', 'Neutral', 'Positive'], 'Score': roberta_scores})
                    st.bar_chart(sentiment_df.set_index('Sentiment'))
                
                with col3:
                    st.subheader("🎯 Emotion Model")
                    top_emotion = max(emotions, key=lambda x: x['score'])
                    st.write("**Prediction:** {}".format(top_emotion['label']))
                    st.write("**Confidence:** {:.2%}".format(top_emotion['score']))
                    emotion_df = pd.DataFrame(emotions)
                    st.bar_chart(emotion_df.set_index('label'))
                
                # Agreement analysis
                st.subheader("🤝 Model Agreement Analysis")
                
                # Simple agreement logic
                bert_positive = bert_rating > 3
                roberta_positive = roberta_sentiment == 'Positive'
                emotion_positive = top_emotion['label'] in ['joy', 'love', 'optimism']
                
                agreements = [bert_positive, roberta_positive, emotion_positive]
                agreement_count = sum(agreements)
                
                if agreement_count == 3:
                    st.success("🎉 All models agree on the sentiment!")
                elif agreement_count == 2:
                    st.warning("⚠️ Two models agree, one disagrees")
                else:
                    st.error("❌ Models disagree on the sentiment")
        else:
            st.warning("Please enter a review to compare models.")

elif analysis_mode == "Feature Explorer":
    st.header("🔍 Feature Explorer")
    
    st.write("Explore the linguistic features that influence sentiment analysis.")
    
    explorer_input = st.text_area("Enter text to explore features:", height=100)
    
    if st.button("🔍 Explore Features", type="primary"):
        if explorer_input.strip():
            features = extract_features(explorer_input)
            
            st.subheader("📊 Feature Analysis")
            
            # Feature metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Word Count", features['word_count'])
                st.metric("Character Count", features['char_count'])
            
            with col2:
                st.metric("Sentences", features['sentence_count'])
                st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
            
            with col3:
                st.metric("Positive Words", features['positive_word_count'])
                st.metric("Negative Words", features['negative_word_count'])
            
            with col4:
                st.metric("Exclamations", features['exclamation_count'])
                st.metric("Questions", features['question_count'])
            
            # Feature explanations
            st.subheader("🎓 Feature Explanations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Textual Features:**")
                st.write("- **Word Count:** Number of words in the text")
                st.write("- **Character Count:** Total characters including spaces")
                st.write("- **Sentence Count:** Number of sentences")
                st.write("- **Avg Word Length:** Average characters per word")
            
            with col2:
                st.write("**Sentiment Indicators:**")
                st.write("- **Positive Words:** Count of positive sentiment words")
                st.write("- **Negative Words:** Count of negative sentiment words")
                st.write("- **Exclamations:** Number of exclamation marks")
                st.write("- **Questions:** Number of question marks")
            
            # Feature importance
            st.subheader("📈 Feature Impact on Sentiment")
            
            feature_impact = {
                'Feature': ['Positive Words', 'Negative Words', 'Exclamations', 'Word Count', 'Questions'],
                'Impact Score': [
                    features['positive_word_count'] * 0.3,
                    features['negative_word_count'] * -0.3,
                    features['exclamation_count'] * 0.1,
                    min(features['word_count'] / 100, 1) * 0.1,
                    features['question_count'] * -0.05
                ]
            }
            
            impact_df = pd.DataFrame(feature_impact)
            fig = px.bar(impact_df, x='Feature', y='Impact Score', title='Feature Impact on Sentiment')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter some text to explore features.")

