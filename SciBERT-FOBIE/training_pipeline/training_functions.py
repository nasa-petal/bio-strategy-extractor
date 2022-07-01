'''
    training_functions - this code provides training functions that detail the training / evaluation loop as well as label
        postprocessing.
        Training Loop / label postprocess adapted from HuggingFace token classification example
        Procedure:
            - calculate training steps based on epoch and training DataLoader sizes
            - forward pass, backward pass, optimizer step, lr scheduler step, optimizer zero grad, progress bar update
            - evaluation loop:
                - gather predictions with Accelerator and compute metrics
                - print metrics
            - save model
    Author: Rishub Tamirisa (rishub.tamirisa@nasa.gov / rishubt2@illinois.edu)
    Citation: https://huggingface.co/course/chapter7/2?fw=pt#training-loop
'''
from transformers import get_scheduler
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_metric

def postprocess(predictions, labels, label_names):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def train_loop(accelerator, model, optimizer, metric, train_dataloader, eval_dataloader, num_train_epochs, label_names, tokenizer, output_dir):

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    progress_bar = tqdm(range(num_training_steps))

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered, label_names)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )

        # Save the model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
        
