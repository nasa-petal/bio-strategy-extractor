from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import preprocess
import json
from datasets import load_metric
import numpy as np
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm.auto import tqdm
import training_functions

# Instatiate the tokenizer
checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the dataset
with open('../AskNature-Summarize/data/asknature-data.json') as json_file:
    data = json.load(json_file)

# Create the train/test dictionary
dataset = preprocess.create_train_test_dict(data, split=0.8)
tokenized_datasets = preprocess.apply_tokenization(dataset, tokenizer)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#Create torch dataloaders
batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], collate_fn=data_collator, batch_size=batch_size
)

# Create the optimizer, accelerator, and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
accelerator = Accelerator()
model, optimizer, train_data_loader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

num_train_epochs = 2
num_updates_per_epoch = len(train_data_loader)
num_training_steps = num_updates_per_epoch * num_train_epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#output params
model_name = "mt5-small-AskNature"
output_dir = "../AskNature-Summarize/model/"+model_name
progress_bar = tqdm(range(num_training_steps))

#train
training_functions.train_loop(
    num_train_epochs=num_train_epochs,
    model=model,
    train_dataloader=train_data_loader,
    eval_dataloader=eval_dataloader,
    accelerator=accelerator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    progress_bar=progress_bar,
    tokenizer=tokenizer,
    metric=load_metric("rouge"),
    output_dir=output_dir
)