import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
import streamlit as st

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

    def predict(self, texts):
        """Predict sentiment for a list of texts"""
        self.model.eval()
        predictions = []
        
        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, prediction = torch.max(outputs.logits, dim=1)
                predictions.append(prediction.item())
                
        return predictions