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

from baseline_code import BERTDataset

import configparser


config = configparser.ConfigParser()
config.read('./code/config.ini')
for section in config.sections():
    print(f'Section: {section}')
    for key, value in config.items(section):
        print(f'  {key} = {value}')


def eval():
    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, './data')
    OUTPUT_DIR = os.path.join(BASE_DIR, './output')
    
    filename=config.get('model','name')
    

    model_name = 'klue/bert-base'
    model = AutoModelForSequenceClassification.from_pretrained(f'./best_model/{filename}', num_labels=7).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train = config.get("data","train")
    dataset_valid = pd.read_csv(os.path.join(DATA_DIR, f'./validation/dev.csv'))
    
    
    # dataset_valid=pd.read_csv('./addDateJson/total_remove_punc.csv')
    # dataset_valid = pd.read_csv(os.path.join(DATA_DIR, f'../dev_g2p.csv'))

    model.eval()
    preds = []
    for idx, sample in tqdm(dataset_valid.iterrows()):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    
    dataset_valid['pred_target'] = preds
    dataset_valid.to_csv(os.path.join(BASE_DIR, f'dev/dev_{filename}.csv'), index=False)
    # dataset_valid.to_csv(os.path.join(BASE_DIR, f'dev/dev_AI_limit.csv'), index=False)
    
    
    
def main():
    eval()

if __name__ == '__main__':
    main()