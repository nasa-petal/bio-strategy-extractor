import datasets

def create_train_test_dict(data: dict, split: float = 0.5) :
    """
    Creates a train/test dictionary from a dataset.
    """
    train = {'summary': [], "source_excerpt": []}
    test = {'summary': [], "source_excerpt": []}
    count = 0
    
    for i in range(count, len(data['summary'])):
        if i % (1 / split) == 0:
            test['summary'].append(data['summary'][i])
            test['source_excerpt'].append(data['source_excerpt'][i])
        else:
            train['summary'].append(data['summary'][i])
            train['source_excerpt'].append(data['source_excerpt'][i])

    split = datasets.DatasetDict(
        {
        'train':datasets.Dataset.from_dict(
            {'source':train['source_excerpt'], 'summary':train['summary']}),
        'test':datasets.Dataset.from_dict(
            {'source':test['source_excerpt'], 'summary':test['summary']})
        }
    )
    
    return split

def apply_tokenization(dataset, tokenizer, max_input_length=128, max_target_length=30):
    def preprocess_function(data):
        model_inputs = tokenizer(
            data["source"], max_length=max_input_length, truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                data["summary"], max_length=max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
    return dataset.map(preprocess_function, batched=True)