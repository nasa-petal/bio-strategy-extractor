from transformers import AutoTokenizer,  AutoModelForTokenClassification
import json
from preprocess import FOBIE_preprocess


checkpoint = "allenai/scibert_scivocab_cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data = json.load(open("data/dev_set.json"))

batch = FOBIE_preprocess(checkpoint, tokenizer, data)

print(batch.keys())