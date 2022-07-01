'''
    train - this code provides a fine-tuning training / evaluation loop for the FOBIE Dataset on pretrained SciBERT
        Accelerator code adapted from HuggingFace Token Classification samples
        Procedure:
            - preprocess FOBIE dataset and create PyTorch Dataloaders
            - instantiate pretrained SciBERT model and set hyperparameters
            - train the model with Acceleration
    Author: Rishub Tamirisa (rishub.tamirisa@nasa.gov / rishubt2@illinois.edu)
'''
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AdamW, get_scheduler
import json
import preprocess as pre
from torch.utils.data import DataLoader
import torch
import training_functions
from accelerate import Accelerator
from datasets import load_metric

checkpoint = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
train_data = json.load(open("SciBERT-FOBIE/data/train_set.json"))
test_data = json.load(open("SciBERT-FOBIE/data/test_set.json"))
tokenized_datasets = pre.create_train_test_dict(
    pre.FOBIE_preprocess(checkpoint, tokenizer, train_data), 
    pre.FOBIE_preprocess(checkpoint, tokenizer, test_data)
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], collate_fn=data_collator, batch_size=8
)

label_names = ['not function', 'function']

id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)

optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

training_functions.train_loop(
                accelerator=accelerator,
                model=model, 
                optimizer=optimizer, 
                metric=load_metric("seqeval"), 
                train_dataloader=train_dataloader, 
                eval_dataloader=eval_dataloader, 
                num_train_epochs=3, 
                label_names=label_names,
                tokenizer=tokenizer,
                output_dir="SciBERT-FOBIE/model"
                )
