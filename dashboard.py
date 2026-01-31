import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

train_df = pd.read_csv("train.csv")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
model.eval()

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].tolist(),
    train_df["target"].tolist(),
    test_size=0.2,
    random_state=42
)

def encode_texts(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    dataset = []
    for i in range(len(texts)):
        dataset.append({
            "input_ids": encodings["input_ids"][i],
            "attention_mask": encodings["attention_mask"][i],
            "labels": torch.tensor(labels[i])
        })
    return dataset

val_dataset = encode_texts(val_texts, val_labels)

all_preds = []
all_labels = []
with torch.no_grad():
    for item in val_dataset:
        outputs = model(
            input_ids=item["input_ids"].unsqueeze(0),
            attention_mask=item["attention_mask"].unsqueeze(0)
        )
        pred = torch.argmax(outputs.logits, dim=1).item()
        all_preds.append(pred)
        all_labels.append(item["labels"].item())


report = classification_report(all_labels, all_preds, output_dict=True)
cm = confusion_matrix(all_labels, all_preds)

st.title("Disaster Tweets Classification Dashboard")

st.header("Classification Report")
st.write(pd.DataFrame(report).transpose())

st.header("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.header("Tweets Distribution")
st.bar_chart(pd.Series(all_labels).value_counts().sort_index())

st.header("Try Your Own Tweet")
user_input = st.text_area("Enter a tweet here:")

if user_input:
    encoding = tokenizer(user_input, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        output = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        pred = torch.argmax(output.logits, dim=1).item()
    label = "Disaster" if pred == 1 else "Not Disaster"
    st.write(f"Prediction: **{label}**")
