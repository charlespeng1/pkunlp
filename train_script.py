from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel, BertTokenizer, AdamW
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import pandas as pd

class FastChineseStyleDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length=128, max_samples=10000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read files
        with open(os.path.join(data_dir, f"{split}.src"), 'r', encoding='utf-8') as f:
            classical_texts = [line.strip().replace(" ", "") for line in f.readlines()]
        with open(os.path.join(data_dir, f"{split}.tgt"), 'r', encoding='utf-8') as f:
            modern_texts = [line.strip().replace(" ", "") for line in f.readlines()]
        
        # Subsample if needed
        if max_samples and len(classical_texts) > max_samples:
            indices = random.sample(range(len(classical_texts)), max_samples)
            classical_texts = [classical_texts[i] for i in indices]
            modern_texts = [modern_texts[i] for i in indices]
            
        self.texts = classical_texts + modern_texts
        self.labels = [1] * len(classical_texts) + [0] * len(modern_texts)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class FastChineseStyleClassifier(nn.Module):
    def __init__(self, pretrained_model="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        
        # Freeze most of BERT
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Only unfreeze the last 2 layers
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs[1])

def plot_training_metrics(history):
    """Plot training metrics and save the plots"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Modern', 'Classical'],
                yticklabels=['Modern', 'Classical'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_fast(model, train_loader, val_loader, device, epochs=5):
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = nn.BCELoss()
    best_val_acc = 0
    
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs.squeeze() >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = (outputs.squeeze() >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        # Store metrics
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_train_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Plot confusion matrix for this epoch
        plot_confusion_matrix(val_labels, val_preds)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_chinese_classifier.pt')
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    # Plot training metrics
    plot_training_metrics(history)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    return history

def test_classifier(model, text, tokenizer, device):
    # Preprocess
    text = text.replace(" ", "")
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        score = output.squeeze().item()
        
    return {
        'text': text,
        'classical_score': score,
        'classification': 'Classical Chinese' if score >= 0.5 else 'Modern Chinese',
        'confidence': max(score, 1-score)
    }

def main():
    # Initialize
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FastChineseStyleDataset(".", "train", tokenizer, max_samples=10000)
    val_dataset = FastChineseStyleDataset(".", "dev", tokenizer, max_samples=1000)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train model
    model = FastChineseStyleClassifier().to(device)
    history = train_fast(model, train_loader, val_loader, device)
    
    # Test some examples
    test_cases = [
        "子曰：學而時習之，不亦說乎？",  # Classical
        "我今天去超市买了些水果",        # Modern
        "天下大勢，分久必合，合久必分",  # Classical
        "这个周末我要去看电影",          # Modern
    ]
    
    print("\nTesting examples:")
    for text in test_cases:
        result = test_classifier(model, text, tokenizer, device)
        print(f"\nText: {result['text']}")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Classical Score: {result['classical_score']:.4f}")

if __name__ == "__main__":
    main()