import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split
import configparser


os.environ['WANDB_PROJECT'] = 'project3'

config = configparser.ConfigParser()
config.read('./code/config.ini')




class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []
        self.labels = []
        
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)

def train():
    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    filename = config.get('model','name')
    batch = config.get('model','batch')

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, './data')
    OUTPUT_DIR = os.path.join(BASE_DIR, './output')

    model_name = 'klue/bert-base'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train = config.get("data","train")
    data = pd.read_csv(os.path.join(DATA_DIR, f'{train}.csv'))
    dataset_train, dataset_valid = train_test_split(data, test_size=0.3, stratify=data['target'],random_state=SEED)

    data_train = BERTDataset(dataset_train, tokenizer)
    data_valid = BERTDataset(dataset_valid, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average='macro')
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate= 2e-05,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=2,
        load_best_model_at_end=False,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED,
        report_to="wandb",
        run_name=filename,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(f'./best_model/{filename}')



def main():
    train()
    # eval()

if __name__ == '__main__':
    main()