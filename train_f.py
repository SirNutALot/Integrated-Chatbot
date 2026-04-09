import pandas as pd
import re
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from torch.utils.data import Dataset

df=pd.read_csv("sentiment_analysis.csv")
df=df[['text','sentiment']].dropna()

def preprocess(text):
    text=str(text).lower()
    return text

df['clean_text']=df['text'].apply(preprocess)

le=LabelEncoder()
df['label']=le.fit_transform(df['sentiment'])

print(le.classes_)

train_texts,val_texts,train_labels,val_labels=train_test_split(
    df['clean_text'],df['label'],test_size=0.2,random_state=42
)

tokenizer=AutoTokenizer.from_pretrained("roberta-base")

train_encodings=tokenizer(list(train_texts),truncation=True,padding=True)
val_encodings=tokenizer(list(val_texts),truncation=True,padding=True)

class SentimentDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=list(labels)
    def __getitem__(self,idx):
        item={key:torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset=SentimentDataset(train_encodings,train_labels)
val_dataset=SentimentDataset(val_encodings,val_labels)

model=AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=len(le.classes_)
)

def compute_metrics(pred):
    labels=pred.label_ids
    preds=pred.predictions.argmax(-1)
    precision,recall,f1,_=precision_recall_fscore_support(labels,preds,average='weighted')
    acc=accuracy_score(labels,preds)
    return {'accuracy':acc,'f1':f1}

training_args=TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

trainer.save_model("sentiment_model")
tokenizer.save_pretrained("sentiment_model")

label_map={i:label for i,label in enumerate(le.classes_)}
with open("sentiment_model/label_map.json","w") as f:
    json.dump(label_map,f)

print("DONE TRAINING")