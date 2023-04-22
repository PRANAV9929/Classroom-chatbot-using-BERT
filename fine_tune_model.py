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

# Initialize the model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_intents))
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


train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=5,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch',
    
)

optimizer = AdamW(model.parameters(), lr=5e-5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, None),
)

# Train the model
trainer.train()
trainer.evaluate()

# Save the trained model and tokenizer
model_path = "trained_models/train_v_1_0/model"
tokenizer_path = "trained_models/train_v_1_0/tokenizer"
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

# Load the fine-tuned model and tokenizer
tokenizer, model = load_fine_tuned_model("trained_models/train_v_1_0")

# Save the id_to_intent dictionary
id_to_intent_path = "trained_models/train_v_1_0/id_to_intent.json"
with open(id_to_intent_path, 'w') as f:
    json.dump(id_to_intent, f)



# Make predictions
question = "Is there a class today or has it been canceled?"
predicted_intent = predict_intent(tokenizer, model, question, id_to_intent)
print(predicted_intent)




 
  
