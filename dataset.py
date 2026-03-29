import pandas as pd
import spacy
import re
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self, freq_threshold=1,apply_cleaning=True):
        self.freq_threshold = freq_threshold
        self.apply_cleaning = apply_cleaning
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

        # python -m spacy download en_core_web_sm
        self.spacy_eng = spacy.load('en_core_web_sm')

    def __len__(self):
        return len(self.itos)
    
    def clean_text(self,text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
    
    def fix_sql(self,query):
        query = query.lower()

        # split patterns like t1customername → t1 customername
        query = re.sub(r"([a-z]+\d+)([a-z_]+)", r"\1 \2", query)

        return query


    def tokenizer(self, text):
        if self.apply_cleaning:
            text = self.clean_text(text)
            return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]
        else:
            text = self.fix_sql(text)   # this already handles the merge split
            tokens = re.findall(
                r"[a-zA-Z_]+\d*\.[a-zA-Z_]+|[a-zA-Z]+\d+|[a-zA-Z_]+|\d+|!=|==|<=|>=|[(),.*=<>]",
                text
            )
            return tokens
    
    def build_vocabulary(self, sentence_list):
        """Build vocabulary from the given list of sentences."""
        counter = Counter()
        for sent in sentence_list:
            for word in self.tokenizer(sent):
                counter[word] += 1

        idx = len(self.itos)
        for word, count in counter.items():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)

        tokens = [self.stoi["<start>"]] + [
            self.stoi.get(word, self.stoi["<unk>"]) for word in tokens
        ] + [self.stoi["<end>"]]

        return tokens
    
    def get_itos_stoi(self):
        return self.itos,self.stoi
    
    def get_max_length(self, sentence_list, percentile=95):
        lengths = []

        for sent in sentence_list:
            tokens = self.numericalize(sent)  # includes <start> and <end>
            lengths.append(len(tokens))

        return int(np.percentile(lengths, percentile))
    
    def encode(self, text):
        tokens = self.numericalize(text)
        return torch.tensor(tokens)
    
    def decode(self, token_ids):
        words = []

        for idx in token_ids:
            word = self.itos.get(idx.item(), "<unk>")

            if word in ["<pad>", "<start>", "<end>"]:
                continue

            words.append(word)

        return " ".join(words).capitalize()
    

class Build_Dataset(Dataset):
    def __init__(self, root_dir, Vocabulary):   # accept, don't build
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(self.root_dir)

        self.text_query = self.df['text_query']
        self.sql_query  = self.df['sql_command']

        self.text_vocab = Vocabulary(freq_threshold=1)
        self.text_vocab.build_vocabulary(self.text_query.tolist())

        self.sql_vocab = Vocabulary(freq_threshold=1, apply_cleaning=False)
        self.sql_vocab.build_vocabulary(self.sql_query.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text_encoded = self.text_vocab.encode(self.text_query[index])
        sql_encoded  = self.sql_vocab.encode(self.sql_query[index])
        return text_encoded, sql_encoded
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):

        text_query = [item[0] for item in batch]
        sql_query  = [item[1] for item in batch]

        text_query = pad_sequence(
            text_query,
            batch_first=True,
            padding_value=self.pad_idx
        )

        sql_query = pad_sequence(
            sql_query,
            batch_first=True,
            padding_value=self.pad_idx
        )

        return text_query, sql_query
    
def Loader(dataset, pad_idx, batch_size = 32,shuffle=True,num_worker=0):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=MyCollate(pad_idx),num_workers=num_worker)
    return loader