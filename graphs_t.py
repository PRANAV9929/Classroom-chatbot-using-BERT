from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json
model_path = "trained_models/train_v_1_0"
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.optim import AdamW

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def preprocess_data(data):
    texts = []
    labels = []

    for item in data:
        intent = item["Intent"]
        patterns = item["Patterns"]
        for pattern in patterns:
            texts.append(pattern)
            labels.append(intent)

    unique_intents = list(set(labels))
    intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id_to_intent = {idx: intent for idx, intent in enumerate(unique_intents)}

    label_ids = [intent_to_id[label] for label in labels]
    return texts, label_ids, id_to_intent, unique_intents


def train_val_split(texts, label_ids):
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, label_ids, test_size=0.2)
    return train_texts, val_texts, train_labels, val_labels


def load_fine_tuned_model(path):
    tokenizer = BertTokenizer.from_pretrained(path + "/tokenizer")
    model = BertForSequenceClassification.from_pretrained(path + "/model")
    return tokenizer, model


def predict_intent(tokenizer, model, text, id_to_intent):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=1).item()
    return id_to_intent[predicted_id]


# Load and preprocess the data
data = load_data("data/train-v1-0.json")
texts, label_ids, id_to_intent, unique_intents = preprocess_data(data)
train_texts, val_texts, train_labels, val_labels = train_val_split(texts, label_ids)
tokenizer = BertTokenizer.from_pretrained(model_path + "/tokenizer")
model = BertForSequenceClassification.from_pretrained(model_path + "/model")

# Load the id_to_intent dictionary
with open(model_path + "/id_to_intent.json", 'r') as f:
    id_to_intent = json.load(f)
    
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


# Create a custom dataset
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

import numpy as np
#from datasets import load_metric

# metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)
preds=[]
for i in val_texts:
    inputs = tokenizer.encode(i, return_tensors="pt")
        # print("inputs:", inputs)

    outputs =model(inputs)[0]
    #print("outputs:", model(inputs))
    #torch.argmax(self.model(inputs), dim=1)
    #y_pred = tf.nn.softmax(self.model.predict(user_input))
    from torch.nn.functional import softmax

    #probs = softmax(outputs, dim=1)
    
    predicted_intent_index = torch.argmax(outputs, dim=1)
    #print("predicted_intent_index:", predicted_intent_index)


    preds.append(predicted_intent_index.item())
from sklearn.metrics import classification_report
print(classification_report(val_labels, preds))

import numpy as np
import matplotlib.pyplot as plt

 # 0 to 15 point radii

plt.scatter(val_labels, preds,   alpha=0.5)
plt.show()
plt.figure(figsize=(10,10))
plt.scatter(val_labels, preds, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(preds), max(val_labels))
p2 = min(min(preds), min(val_labels))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

