import torch
import tkinter as tk
from tkinter import scrolledtext
import json
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import AutoTokenizer,AutoModelForSequenceClassification

chat_tokenizer=AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_tokenizer.padding_side="left"
chat_model=AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

sent_tokenizer=AutoTokenizer.from_pretrained("sentiment_model")
sent_model=AutoModelForSequenceClassification.from_pretrained("sentiment_model")

with open("sentiment_model/label_map.json") as f:
    label_map=json.load(f)

chat_history_ids=None

def get_sentiment(text):
    inputs=sent_tokenizer(text,return_tensors="pt",truncation=True,padding=True)
    with torch.no_grad():
        outputs=sent_model(**inputs)
    pred=torch.argmax(outputs.logits,dim=1).item()
    return label_map[str(pred)]

def send_message(event=None):
    global chat_history_ids
    user_input=entry_box.get()
    entry_box.delete(0,tk.END)
    if user_input.strip()=="":
        return
    chat_area.insert(tk.END,"You: "+user_input+"\n")
    sentiment=get_sentiment(user_input)
    new_input_ids=chat_tokenizer.encode(user_input+chat_tokenizer.eos_token,return_tensors='pt')
    bot_input_ids=torch.cat([chat_history_ids,new_input_ids],dim=-1) if chat_history_ids is not None else new_input_ids
    with torch.no_grad():
        chat_history_ids=chat_model.generate(bot_input_ids,max_length=300,pad_token_id=chat_tokenizer.eos_token_id)
    response=chat_tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],skip_special_tokens=True)

    if sentiment=="negative":
        response="We are sorry about your experience."
    elif sentiment=="positive":
        response="Glad you liked it!"
    else:
        response="Thanks for your feedback."

    chat_area.insert(tk.END,f"Bot ({sentiment}): {response}\n\n")
    chat_area.yview(tk.END)

def clear_chat():
    global chat_history_ids
    chat_history_ids=None
    chat_area.delete('1.0',tk.END)

root=tk.Tk()
root.title("AI Product Feedback Chatbot")
root.geometry("520x600")

chat_area=scrolledtext.ScrolledText(root,wrap=tk.WORD,width=60,height=20,font=("Arial",12))
chat_area.pack(padx=10,pady=10)

entry_box=tk.Entry(root,width=50,font=("Arial",12))
entry_box.pack(padx=10,pady=5)
entry_box.bind("<Return>",send_message)

tk.Button(root,text="Send",command=send_message).pack()
tk.Button(root,text="Clear Chat",command=clear_chat).pack()

root.mainloop()