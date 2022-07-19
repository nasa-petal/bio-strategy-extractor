'''
    preprocess - this code preprocess the FOBIE dataset into a format compatible with the HF transformers API
        Procedure:
            - iterate through the FOBIE json dataset and extract the sentences and spans
            - adjust spans to match the tokenization of the sentence and padding of the tokenizer
    Author: Rishub Tamirisa (rishub.tamirisa@nasa.gov / rishubt2@illinois.edu)
'''

import json
<<<<<<< HEAD
from re import L
=======
>>>>>>> main
import torch
import numpy as np
from transformers import AutoTokenizer,  AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import tokenizations
import datasets


def match_span_to_tokenizer(ground, split, span):
    """Converts a span to the tokenizer's tokenization.
    Returns:
        (list): list containing:
            [int, int]: span in tokenizer's tokenization
    """
<<<<<<< HEAD
    ground.insert(0, "[CLS]")
    ground.append("[SEP]")
=======
>>>>>>> main
    a2b, b2a = tokenizations.get_alignments(ground, split)
    adjusted = []
    for idx, val in enumerate(b2a):
        if val and val[0] in range(span[0], span[1]+1):
            adjusted.append(idx)
    if adjusted:
        return [adjusted[0]+1, adjusted[len(adjusted)-1]+1]
    else:
        return None

def FOBIE_preprocess(checkpoint, tokenizer, data):
    """Prepares the FOBIE dataset for the HF transformers API.
    Returns:
        (dict): dict containing:
            'attention_mask': (torch.tensor),
            'input_ids': (torch.tensor),
            'token_type_ids': (torch.tensor),
            'labels': (torch.tensor)
    """
    dict = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'labels': []
        }
    sentences = []
    for source_doc_id in data:
        for sentence_id in data[source_doc_id]:
            sentence: str = data[source_doc_id][sentence_id]['sentence']
            sentences.append(sentence)
<<<<<<< HEAD
=======

>>>>>>> main
    for source_doc_id in data:
        for sentence_id in data[source_doc_id]:
            sentence: str = data[source_doc_id][sentence_id]['sentence']
            input = tokenizer(sentence)
            label = np.zeros(len(input.tokens()))
<<<<<<< HEAD
            good_sentence = True
=======
>>>>>>> main
            for sentence_modifier_id in data[source_doc_id][sentence_id]['annotations']['modifiers']:
                for arg_id in data[source_doc_id][sentence_id]['annotations']['modifiers'][sentence_modifier_id]:
                    arg_attributes = data[source_doc_id][sentence_id]['annotations']['modifiers'][sentence_modifier_id][arg_id]
                    span = [int(arg_attributes['span_start']), int(arg_attributes['span_end'])]
<<<<<<< HEAD
                    if (span[0] >= len(sentence.split())):
                        good_sentence = False
                    span = match_span_to_tokenizer(sentence.split(), input.tokens(), span)
                        
                    if span:
                        label[span[0]:span[1]] = 1
            if good_sentence and np.any(label):
                label[0] = -100
                label[-1] = -100
                dict['labels'].append(np.array(label))
                dict['input_ids'].append(input['input_ids'])
                dict['token_type_ids'].append(input['token_type_ids'])
                dict['attention_mask'].append(input['attention_mask'])
=======
                    span = match_span_to_tokenizer(sentence.split(), input.tokens(), span)
                    if span:
                        label[span[0]:span[1]] = 1
           
            label[0] = -100
            label[-1] = -100
            dict['labels'].append(np.array(label))
            dict['input_ids'].append(input['input_ids'])
            dict['token_type_ids'].append(input['token_type_ids'])
            dict['attention_mask'].append(input['attention_mask'])
>>>>>>> main
    return dict

def create_train_test_dict(train, test) :
    """Creates training and testing DatasetDicts.
    Returns:
        (datasets.DatasetDict): datasets.DatasetDict containing:
            'train':datasets.Dataset.from_dict(
                {'labels', 'input_ids', 'token_type_ids', 'attention_mask'}),
            'test':datasets.Dataset.from_dict(
                {'labels', 'input_ids', 'token_type_ids', 'attention_mask'})
    """
    dict = datasets.DatasetDict(
        {
        'train':datasets.Dataset.from_dict(
            {'labels':train['labels'], 'input_ids':train['input_ids'], 'token_type_ids':train['token_type_ids'], 'attention_mask':train['attention_mask']}),
        'test':datasets.Dataset.from_dict(
            {'labels':test['labels'], 'input_ids':test['input_ids'], 'token_type_ids':test['token_type_ids'], 'attention_mask':test['attention_mask']})
        }
    )
    return dict