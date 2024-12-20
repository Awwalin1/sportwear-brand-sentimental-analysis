# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import defaultdict
import os
from typing import Dict, Union
import warnings
warnings.filterwarnings('ignore')

# Department Detection Keywords
DEPARTMENT_KEYWORDS = {
    'Customer Service': ['service', 'support', 'staff', 'representative', 'helpdesk', 'assistance'],
    'Product Quality': ['quality', 'material', 'durability', 'construction', 'craftsmanship'],
    'Design': ['design', 'style', 'look', 'aesthetic', 'appearance', 'color'],
    'Pricing': ['price', 'cost', 'expensive', 'cheap', 'value', 'affordable'],
    'Shipping': ['shipping', 'delivery', 'shipment', 'arrived', 'packaging'],
    'Size & Fit': ['size', 'fit', 'comfort', 'comfortable', 'tight', 'loose'],
    'Store Experience': ['store', 'shop', 'retail', 'location', 'shopping experience'],
    'Website/App': ['website', 'app', 'online', 'mobile', 'site', 'application'],
    'Marketing': ['advertisement', 'promotion', 'campaign', 'marketing', 'ads'],
    'Innovation': ['technology', 'innovation', 'feature', 'innovative', 'new']
}

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RobertaSentimentAnalyzer:
    def __init__(self, model_name='roberta-base', num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_labels = num_labels
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)

    @classmethod
    def load_model(cls, path='./saved_model'):
        try:
            config = torch.load(os.path.join(path, 'config.pt'))
            analyzer = cls(
                model_name=config['model_name'],
                num_labels=config['num_labels']
            )
            analyzer.model = AutoModelForSequenceClassification.from_pretrained(path)
            analyzer.tokenizer = AutoTokenizer.from_pretrained(path)
            analyzer.model.to(analyzer.device)
            return analyzer
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise

@st.cache_resource
def load_models():
    """Load both BERT and RoBERTa models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BERT
    bert_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_name).to(device)
    bert_model.eval()
    
    # Load RoBERTa
    roberta_analyzer = RobertaSentimentAnalyzer.load_model('./saved_model')
    
    return {
        'device': device,
        'bert': (bert_model, bert_tokenizer),
        'roberta': (roberta_analyzer.model, roberta_analyzer.tokenizer)
    }

def detect_departments(text: str) -> Dict[str, float]:
    """Detect mentioned departments and their sentiment context"""
    text_lower = text.lower()
    departments = {}
    
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        mention_count = sum(1 for keyword in keywords if keyword in text_lower)
        if mention_count > 0:
            departments[dept] = mention_count
            
    return departments

class ReviewAnalytics:
    def __init__(self):
        self.reviews_data = []
        
    def add_review(self, text: str, result: Dict, brand: str):
        departments = detect_departments(text)
        
        review_data = {
            'text': text,
            'brand': brand,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'departments': departments,
            'length': len(text.split()),
            'timestamp': pd.Timestamp.now()
        }
        self.reviews_data.append(review_data)
    
    def get_dataframe(self):
        return pd.DataFrame(self.reviews_data)

def analyze_text(text: str, brand: str, models: dict) -> Dict:
    """Analyze text sentiment for a specific brand"""
    device = models['device']
    bert_model, bert_tokenizer = models['bert']
    roberta_model, roberta_tokenizer = models['roberta']
    
    # Preprocess
    if len(text.split()) <= 2:
        text = f"This is {text}"
    if brand.lower() not in text.lower():
        text = f"{brand} product review: {text}"
    
    # BERT prediction
    bert_inputs = bert_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        bert_outputs = bert_model(**bert_inputs)
        bert_probs = torch.softmax(bert_outputs.logits, dim=1)[0].cpu().numpy()
        
    bert_sentiment = {
        'Negative': float(np.sum(bert_probs[0:2])),
        'Neutral': float(bert_probs[2]),
        'Positive': float(np.sum(bert_probs[3:5]))
    }
    
    # RoBERTa prediction
    dataset = SentimentDataset([text], [0], roberta_tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))
    
    with torch.no_grad():
        roberta_outputs = roberta_model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        )
        roberta_probs = torch.softmax(roberta_outputs.logits, dim=1)[0].cpu().numpy()
    
    roberta_sentiment = {
        'Negative': float(roberta_probs[0]),
        'Neutral': float(roberta_probs[1]),
        'Positive': float(roberta_probs[2])
    }
    
    # Combine predictions
    combined_probs = {
        key: 0.3 * bert_sentiment[key] + 0.7 * roberta_sentiment[key]
        for key in bert_sentiment.keys()
    }
    
    sentiment = max(combined_probs.items(), key=lambda x: x[1])[0]
    confidence = max(combined_probs.values())
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': combined_probs,
        'individual_predictions': {
            'bert': bert_sentiment,
            'roberta': roberta_sentiment
        }
    }

def create_visualizations(analytics: ReviewAnalytics):
    if not analytics.reviews_data:
        return []
    
    df = analytics.get_dataframe()
    visualizations = []
    
    # 1. Sentiment Distribution by Brand
    fig1 = px.pie(df, names='sentiment', title='Overall Sentiment Distribution',
                  color='sentiment', color_discrete_map={
                      'Positive': '#00CC96',
                      'Neutral': '#FFA500',
                      'Negative': '#FF4B4B'
                  })
    visualizations.append(("Sentiment Distribution", fig1))
    
    # 2. Brand Performance Comparison
    brand_sentiment = pd.crosstab(df['brand'], df['sentiment'], normalize='index') * 100
    fig2 = px.bar(brand_sentiment, title='Brand Performance Comparison',
                  color_discrete_sequence=['#FF4B4B', '#FFA500', '#00CC96'])
    visualizations.append(("Brand Performance", fig2))
    
    # 3. Department Sentiment Analysis
    dept_data = []
    for review in analytics.reviews_data:
        for dept, count in review['departments'].items():
            dept_data.append({
                'Department': dept,
                'Sentiment': review['sentiment'],
                'Count': count,
                'Brand': review['brand']
            })
    
    if dept_data:
        dept_df = pd.DataFrame(dept_data)
        fig3 = px.sunburst(dept_df, path=['Department', 'Sentiment', 'Brand'], 
                          values='Count',
                          title='Department Sentiment Analysis')
        visualizations.append(("Department Analysis", fig3))
    
    # 4. Confidence Score Distribution
    fig4 = px.histogram(df, x='confidence', nbins=20,
                       title='Model Confidence Distribution',
                       color='sentiment',
                       color_discrete_map={
                           'Positive': '#00CC96',
                           'Neutral': '#FFA500',
                           'Negative': '#FF4B4B'
                       })
    visualizations.append(("Confidence Distribution", fig4))
    
    # 5. Review Length Analysis
    fig5 = px.scatter(df, x='length', y='confidence', 
                     color='sentiment', size='confidence',
                     title='Review Length vs Confidence',
                     color_discrete_map={
                         'Positive': '#00CC96',
                         'Neutral': '#FFA500',
                         'Negative': '#FF4B4B'
                     })
    visualizations.append(("Review Analysis", fig5))
    
    # 6. Department Performance Heatmap
    if dept_data:
        dept_df = pd.DataFrame(dept_data)
        dept_sentiment = pd.crosstab(dept_df['Department'], 
                                   [dept_df['Brand'], dept_df['Sentiment']])
        
        fig6 = go.Figure(data=go.Heatmap(
            z=dept_sentiment.values,
            x=[f"{b}-{s}" for b, s in dept_sentiment.columns],
            y=dept_sentiment.index,
            colorscale='RdYlGn'
        ))
        fig6.update_layout(title='Department Performance by Brand')
        visualizations.append(("Department Heatmap", fig6))
    
    # 7. Sentiment Trends
    df['hour'] = df['timestamp'].dt.hour
    hourly_sentiment = pd.crosstab(df['hour'], df['sentiment'])
    fig7 = px.line(hourly_sentiment, title='Hourly Sentiment Trends',
                   color_discrete_map={
                       'Positive': '#00CC96',
                       'Neutral': '#FFA500',
                       'Negative': '#FF4B4B'
                   })
    visualizations.append(("Sentiment Trends", fig7))
    
    # 8. Department Mention Frequency by Brand
    if dept_data:
        fig8 = px.bar(dept_df, x='Department', y='Count',
                     color='Brand', title='Department Mentions by Brand',
                     barmode='group')
        visualizations.append(("Department Mentions", fig8))
    
    # 9. Brand Confidence Comparison
    fig9 = px.box(df, x='brand', y='confidence', color='sentiment',
                  title='Confidence Distribution by Brand',
                  color_discrete_map={
                      'Positive': '#00CC96',
                      'Neutral': '#FFA500',
                      'Negative': '#FF4B4B'
                  })
    visualizations.append(("Brand Confidence", fig9))
    
    # 10. Department Performance Radar
    if dept_data:
        dept_perf = dept_df.groupby('Department').apply(
            lambda x: (x['Sentiment'] == 'Positive').mean() * 100
        ).reset_index(name='Performance')
        
        fig10 = go.Figure(data=go.Scatterpolar(
            r=dept_perf['Performance'],
            theta=dept_perf['Department'],
            fill='toself'
        ))
        fig10.update_layout(title='Department Performance Radar')
        visualizations.append(("Performance Radar", fig10))
    
    return visualizations

def main():
    st.set_page_config(page_title="Brand Analysis Dashboard", page_icon="📊", layout="wide")
    
    if 'analytics' not in st.session_state:
        st.session_state.analytics = ReviewAnalytics()
    
    st.title("Brand Analysis Dashboard")
    st.markdown("""
    Analyze customer sentiments for Adidas, Nike, and Puma products using our advanced dual-model system:
    - BERT (General Sentiment Model)
    - RoBERTa (Custom-trained on Brand Reviews)
    """)
    
    tab1, tab2 = st.tabs(["Analysis", "Dashboard"])
    
    with tab1:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            brand = st.selectbox(
                "Select Brand:",
                ["Adidas", "Nike", "Puma"],
                help="Choose the brand you want to analyze"
            )
            
            brand_info = {
                'Adidas': '⭐',
                'Nike': '✔️',
                'Puma': '🐆'
            }
            st.markdown(f"### Selected: {brand} {brand_info[brand]}")
        
        text = st.text_area(
            "Enter review text:",
            height=150,
            placeholder=f"Enter your {brand} product review here..."
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if text.strip():
                with st.spinner(f"Analyzing {brand} review..."):
                    result = analyze_text(text, brand, load_models())
                    st.session_state.analytics.add_review(text, result, brand)
                
                st.markdown("### Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    emoji = {
                        'Positive': '😊',
                        'Neutral': '😐',
                        'Negative': '😞'
                    }[result['sentiment']]
                    st.markdown(f"**Sentiment**: {result['sentiment']} {emoji}")
                with col2:
                    confidence_str = "Strong" if result['confidence'] > 0.7 else "Moderate" if result['confidence'] > 0.5 else "Weak"
                    st.markdown(f"**Confidence**: {confidence_str} ({result['confidence']:.1%})")
                
                # Model predictions
                st.markdown("### Model Predictions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Combined Model**")
                    for sent, prob in result['probabilities'].items():
                        st.write(f"{sent}: {prob:.2%}")
                
                with col2:
                    st.markdown("**BERT (General Model)**")
                    for sent, prob in result['individual_predictions']['bert'].items():
                        st.write(f"{sent}: {prob:.2%}")
                
                with col3:
                    st.markdown("**RoBERTa (Brand-Trained)**")
                    for sent, prob in result['individual_predictions']['roberta'].items():
                        st.write(f"{sent}: {prob:.2%}")
                
                # Department Analysis
                departments = detect_departments(text)
                if departments:
                    st.markdown("### Detected Departments")
                    for dept, count in departments.items():
                        sentiment_color = {
                            'Positive': 'green',
                            'Neutral': 'orange',
                            'Negative': 'red'
                        }[result['sentiment']]
                        st.markdown(f"""
                        <div style='padding: 10px; border-left: 5px solid {sentiment_color}'>
                            <b>{dept}</b>: Mentioned {count} times
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        if not st.session_state.analytics.reviews_data:
            st.warning("No data available. Please analyze some reviews first.")
        else:
            st.markdown("## Analytics Dashboard")
            st.markdown("### Key Insights")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            df = st.session_state.analytics.get_dataframe()
            
            with col1:
                total_reviews = len(df)
                st.metric("Total Reviews", total_reviews)
            
            with col2:
                positive_rate = (df['sentiment'] == 'Positive').mean() * 100
                st.metric("Positive Rate", f"{positive_rate:.1f}%")
            
            with col3:
                avg_confidence = df['confidence'].mean() * 100
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col4:
                dept_coverage = len(set().union(*[set(r['departments'].keys()) 
                                                for r in st.session_state.analytics.reviews_data]))
                st.metric("Departments Covered", f"{dept_coverage}/10")
            
            # Visualizations
            visualizations = create_visualizations(st.session_state.analytics)
            
            for i in range(0, len(visualizations), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(visualizations):
                        st.markdown(f"### {visualizations[i][0]}")
                        st.plotly_chart(visualizations[i][1], use_container_width=True)
                
                with col2:
                    if i + 1 < len(visualizations):
                        st.markdown(f"### {visualizations[i+1][0]}")
                        st.plotly_chart(visualizations[i+1][1], use_container_width=True)

if __name__ == "__main__":
    main()