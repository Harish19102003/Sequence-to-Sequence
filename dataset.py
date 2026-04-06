import pandas as pd
import spacy
import re
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from config import batch_size

g=torch.Generator()
g.manual_seed(42)

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
    
def parse_schema(schema_str):
    """
    Converts CREATE TABLE statement into compact schema string.
    
    Input:
        CREATE TABLE addresses (
            address_id INTEGER,
            line_1 TEXT,
            country TEXT
        )
    
    Output:
        "addresses : address_id line_1 country"
    """
    if not isinstance(schema_str, str):
        return ""

    result = []

    # find all CREATE TABLE blocks
    # matches: CREATE TABLE tablename ( ... )
    table_blocks = re.findall(
        r"CREATE\s+TABLE\s+(\w+)\s*\(([^)]+)\)",
        schema_str,
        re.IGNORECASE
    )

    for table_name, columns_block in table_blocks:
        # extract column names — first word on each line
        col_names = []
        for line in columns_block.strip().split("\n"):
            line = line.strip().strip(",")
            if not line:
                continue
            col_name = line.split()[0]   # first word is column name
            col_names.append(col_name.lower())

        result.append(f"{table_name.lower()} : {' '.join(col_names)}")

    return " | ".join(result)

class Build_Dataset(Dataset):
    def __init__(self, root_dir, Vocabulary):
        self.root_dir = Path(root_dir)
        self.df       = pd.read_csv(self.root_dir)

        self.text_query = self.df['question']    # ← column name changed
        self.sql_query  = self.df['query']       # ← column name changed
        self.schemas    = self.df['schema']      # ← new column

        # parse CREATE TABLE → compact string for every row
        self.parsed_schemas = [
            parse_schema(s) for s in self.schemas
        ]

        # combine question + schema for vocab building
        combined = [
            f"{q} | {s}"
            for q, s in zip(self.text_query, self.parsed_schemas)
        ]

        self.text_vocab = Vocabulary(freq_threshold=1)
        self.text_vocab.build_vocabulary(combined)

        self.sql_vocab = Vocabulary(freq_threshold=1, apply_cleaning=False)
        self.sql_vocab.build_vocabulary(self.sql_query.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        question = self.text_query[index]
        sql      = self.sql_query[index]
        schema   = self.parsed_schemas[index]

        # model sees: "What is the total number... | addresses : address_id line_1..."
        combined = f"{question} | {schema}"

        text_encoded = self.text_vocab.encode(combined)
        sql_encoded  = self.sql_vocab.encode(sql)
        return text_encoded, sql_encoded
    
class MyCollate:
    def __init__(self,pad_idx):
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
    
root_dir = "data/text-to-sql/train.csv"

dataset = Build_Dataset(root_dir, Vocabulary)
text_vocab = dataset.text_vocab
sql_vocab  = dataset.sql_vocab
pad_idx = text_vocab.stoi["<pad>"]

train_dataset,val_dataset = random_split(dataset, [0.8, 0.2], generator=g)
train_dataset,test_dataset = random_split(train_dataset, [0.9, 0.1], generator=g)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCollate(pad_idx), generator=g)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx), generator=g)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=MyCollate(pad_idx), generator=g)