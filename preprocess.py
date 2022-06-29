'''
    preprocess - this code preprocess the FOBIE dataset into a format compatible with the HF transformers API
        Procedure:
            - iterate through the FOBIE json dataset and extract the sentences and spans
            - adjust spans to match the tokenization of the sentence and padding of the tokenizer
    Author: Rishub Tamirisa (rishub.tamirisa@nasa.gov / rishubt2@illinois.edu)
'''

import json
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
            'Train': {
                # 'sentences':[],
                'labels':[]
            }
        }
    sentences = []
    for source_doc_id in data:
        for sentence_id in data[source_doc_id]:
            sentence: str = data[source_doc_id][sentence_id]['sentence']
            sentences.append(sentence)

    batch = tokenizer(sentences, padding=True, truncation=False, return_tensors='pt')

    for source_doc_id in data:
        for sentence_id in data[source_doc_id]:
            sentence: str = data[source_doc_id][sentence_id]['sentence']
            tokens = tokenizer.tokenize(sentence)
            label = np.zeros(len(batch['input_ids'][0]))
            for sentence_modifier_id in data[source_doc_id][sentence_id]['annotations']['modifiers']:
                for arg_id in data[source_doc_id][sentence_id]['annotations']['modifiers'][sentence_modifier_id]:
                    arg_attributes = data[source_doc_id][sentence_id]['annotations']['modifiers'][sentence_modifier_id][arg_id]
                    span = [int(arg_attributes['span_start']), int(arg_attributes['span_end'])]
                    span = match_span_to_tokenizer(sentence.split(), tokens, span)
                    if span:
                        label[span[0]:span[1]] = 1
            # print(label)
            
            # dict['Train']['sentences'].append(tokens)
            dict['Train']['labels'].append(label)
    dict['Train']['labels'] = np.array(dict['Train']['labels'])
    batch['labels'] = torch.tensor(dict['Train']['labels']).long()
    # print(batch['labels'])
    return batch

def create_train_test_dict(train, test) :
    dict = datasets.DatasetDict(
        {
        'train':datasets.Dataset.from_dict(
            {'attention_mask':train['attention_mask'],'input_ids':train['input_ids'],'token_type_ids':train['token_type_ids'],'labels':train['labels']}),
        'test':datasets.Dataset.from_dict(
            {'attention_mask':test['attention_mask'],'input_ids':test['input_ids'],'token_type_ids':test['token_type_ids'],'labels':test['labels']})
        }
    )
    return dict

checkpoint = "allenai/scibert_scivocab_cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
train_data = json.load(open("data/dev_set.json"))
test_data = json.load(open("data/test_set.json"))

tokenized_dataset = create_train_test_dict(
    FOBIE_preprocess(checkpoint, tokenizer, train_data), 
    FOBIE_preprocess(checkpoint, tokenizer, test_data)
)


# training_args= TrainingArguments("SciBERT-FOBIE", evaluation_strategy="epoch")
# model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=2)

# trainer = Trainer (
#     model,
#     training_args,
#     train_dataset=tokenized_dataset['train'],
#     eval_dataset=tokenized_dataset['test'],
#     tokenizer=tokenizer
# )
# trainer.train()