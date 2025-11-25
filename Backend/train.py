import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import gc
from tqdm import tqdm

from config import Config

# Memory optimization


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class OptimizedUniversityClassifier(nn.Module):
    def __init__(self, n_classes, model_name=Config.MODEL_NAME):
        super(OptimizedUniversityClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # 🎯 BETTER ARCHITECTURE - Smaller & More Efficient
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),  # Reduced size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)


class OptimizedModelTrainer:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")

        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.tag_responses = None

    def load_intents_data(self):
        """Load and preprocess data"""
        print("📖 Loading intents data...")

        with open(Config.INTENTS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        labels = []
        tag_responses = {}

        for intent in data['intents']:
            tag = intent['tag']
            patterns = intent['patterns']
            responses = intent['responses']

            tag_responses[tag] = responses

            # Add original patterns
            for pattern in patterns:
                texts.append(pattern.lower())
                labels.append(tag)

        print(
            f"✅ Loaded {len(texts)} samples across {len(set(labels))} classes")
        return texts, labels, tag_responses

    def train(self):
        """Optimized training with better learning rate and scheduling"""
        print("🎯 Starting OPTIMIZED University Chatbot Training...")

        # Load data
        texts, labels, tag_responses = self.load_intents_data()
        self.tag_responses = tag_responses

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=0.2,  # More training data
            random_state=Config.RANDOM_STATE,
            stratify=labels
        )

        print(f"📊 Training samples: {len(train_texts)}")
        print(f"📊 Validation samples: {len(val_texts)}")

        # Encode labels
        self.label_encoder = LabelEncoder()
        train_encoded = self.label_encoder.fit_transform(train_labels)
        val_encoded = self.label_encoder.transform(val_labels)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        n_classes = len(self.label_encoder.classes_)
        self.model = OptimizedUniversityClassifier(n_classes=n_classes)
        self.model = self.model.to(self.device)

        # Create datasets
        train_dataset = IntentDataset(
            train_texts, train_encoded, self.tokenizer, Config.MAX_LENGTH)
        val_dataset = IntentDataset(
            val_texts, val_encoded, self.tokenizer, Config.MAX_LENGTH)

        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        # 🎯 BETTER OPTIMIZER SETTINGS
        optimizer = AdamW(self.model.parameters(), lr=1e-4,
                          weight_decay=1e-4)  # Higher LR
        loss_fn = nn.CrossEntropyLoss()

        # Learning rate scheduler
        total_steps = len(train_loader) * Config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # Warmup phase
            num_training_steps=total_steps
        )

        # Training
        best_val_accuracy = 0
        patience = 3
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            print(f"\n📍 Epoch {epoch + 1}/{Config.NUM_EPOCHS}")

            # Training phase
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0

            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            avg_loss = total_loss / len(train_loader)

            print(
                f"📊 Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f}")
            print(f"📈 Val Loss: {avg_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_model()
                print("💾 Saved best model!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping after {epoch + 1} epochs")
                    break

            clear_memory()

        # Save tokenizer
        self.tokenizer.save_pretrained(Config.TOKENIZER_PATH)
        print(f"✅ Training completed! Best accuracy: {best_val_accuracy:.4f}")

    def save_model(self):
        """Save the trained model"""
        os.makedirs('./models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'tag_responses': self.tag_responses,
        }, Config.MODEL_SAVE_PATH)


def main():
    trainer = OptimizedModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
