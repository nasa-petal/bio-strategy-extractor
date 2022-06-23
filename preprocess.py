import json
import torch
import numpy as np
import pandas as pd
from transformers import AdamW, AutoTokenizer


checkpoint = "allenai/scibert_scivocab_cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# FOBIE Dataset
data = json.load(open("data/dev_set.json"))

#   1   2 3    4     5  6     7
# this is a testing of the spanning
#   1   2 3   4   5  6   7   8    9
# this is a test ing of the span ing

def match_span_to_tokenizer(ground, split, span):
    gd_idx = 0
    sp_idx = 0
    ground.append("[END]")
    split.append("[END]")
    result = [span[0], span[1]]
    shift = None
    while gd_idx < len(ground):
        if gd_idx < len(ground) - 1 and ground[gd_idx] != split[sp_idx]:
            gd_idx += 1
            while (sp_idx < len(split) and split[sp_idx] != ground[gd_idx]):
                sp_idx += 1
            shift = sp_idx - gd_idx
            if gd_idx - 1 <= span[1]:
                result[1] = span[1] + shift
            if gd_idx - 1 <= span[0]:
                result[0] = span[0] + shift
        sp_idx += 1
        gd_idx += 1
    return result


ground = ['Another', 'interesting', 'finding', 'was', 'the', 'lack', 'of', 'correlation', 'between', 'resting', 'levels', 'of', 'both', 'HSP', '90β', 'and', 'HSC', '70', 'and', 'CTmax.']
split = ['another', 'interesting', 'finding', 'was', 'the', 'lack', 'of', 'correlation', 'between', 'resting', 'levels', 'of', 'both', 'hs', '##p', '90', '##β', 'and', 'hs', '##c', '70', 'and', 'ct', '##max', '.']
span = [1,3] 
span = match_span_to_tokenizer(ground, split, span)
# print(span)       
        
dict = {
        'Train': {
            'sentences':[],
            'labels':[]
        }
       }
dict['Train']['sentences'].append('test')
print(dict)

for source_doc_id in data:
    for sentence_id in data[source_doc_id]:
        sentence: str = data[source_doc_id][sentence_id]['sentence']
        tokens = tokenizer.tokenize(sentence)
        for sentence_modifier_id in data[source_doc_id][sentence_id]['annotations']['modifiers']:
            label = np.zeros(len(tokens))
            for arg_id in data[source_doc_id][sentence_id]['annotations']['modifiers'][sentence_modifier_id]:
                arg_attributes = data[source_doc_id][sentence_id]['annotations']['modifiers'][sentence_modifier_id][arg_id]
                span = [int(arg_attributes['span_start']), int(arg_attributes['span_end'])]
                print(arg_attributes)

                # span = match_span_to_tokenizer(sentence.lower().split(), tokens, span)
                # label[span[0]:span[1]] = 1
            print(label)