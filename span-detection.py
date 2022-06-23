from transformers import AutoTokenizer, pipeline,  AutoModelForTokenClassification
checkpoint = "allenai/scibert_scivocab_cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForTokenClassification.from_pretrained(checkpoint)

