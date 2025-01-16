import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
import os
import torch
from datasets import load_dataset, concatenate_datasets

class ModernChineseTranslator:
    def __init__(self):
        print("Initializing ModernChineseTranslator...")
        # Initialize Helsinki-NLP translator
        self.model_name = "Helsinki-NLP/opus-mt-zh-en"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Initialize the classical Chinese translator
        self.setup_classical_translator()
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print("Models initialized successfully!")

    def setup_classical_translator(self):
        """Setup the classical Chinese translator using pre-trained model"""
        print("Setting up classical Chinese translator...")
        try:
            model_name = "raynardj/wenyanwen-ancient-translate-to-modern"
            self.classical_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.classical_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.classical_model = self.classical_model.to('cuda')
            self.has_classical_translator = True
            print("Classical Chinese translator loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load classical Chinese translator. Error: {str(e)}")
            self.has_classical_translator = False

    def translate_classical_to_modern(self, text):
        """Translate classical Chinese to modern Chinese"""
        if not self.has_classical_translator:
            print("Warning: Classical Chinese translator not available")
            return text
        
        # Prepare input
        inputs = self.classical_tokenizer(text, 
                                        return_tensors="pt", 
                                        max_length=512, 
                                        truncation=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.classical_model.generate(
                inputs.input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                bos_token_id=101,
                eos_token_id=self.classical_tokenizer.sep_token_id,
                pad_token_id=self.classical_tokenizer.pad_token_id
            )
        
        # Decode output
        translated_text = self.classical_tokenizer.decode(outputs[0], 
                                                        skip_special_tokens=True)
        return translated_text

    def translate(self, text, max_length=128):
        """Translate Chinese text to English using Helsinki-NLP model"""
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, 
                              truncation=True, padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode and return the translation
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

    def translate_batch(self, texts, batch_size=32, max_length=128):
        """Translate a batch of Chinese texts to English"""
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", max_length=max_length,
                                  truncation=True, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            batch_translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
        
        return translations

    def save(self, path):
        """Save model configuration"""
        if not os.path.exists(path):
            os.makedirs(path)
        # Save tokenizer configurations
        self.tokenizer.save_pretrained(os.path.join(path, 'helsinki_tokenizer'))
        if self.has_classical_translator:
            self.classical_tokenizer.save_pretrained(os.path.join(path, 'classical_tokenizer'))

    def load(self, path):
        """Load model configuration"""
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'helsinki_tokenizer'))
        if os.path.exists(os.path.join(path, 'classical_tokenizer')):
            self.classical_tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'classical_tokenizer'))