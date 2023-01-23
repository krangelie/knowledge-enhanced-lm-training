import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer


class KelmDataset(Dataset):
    def __init__(self, data_file, tokenizer=None, max_seq_length=0):
        super().__init__()

        with open(data_file) as json_file:
            self.json_list = list(json_file)

        #self.kelm_original_indices = []
        self.kelm_generated_sentences = []
        for item in self.json_list:
            entry = json.loads(item)
            #self.kelm_original_index += [entry["original_idx"]]
            self.kelm_generated_sentences += [entry["gen_sentence"]]

        if tokenizer:
            # TODO: change this loop: indexable list of dicts for faster access later
            self.kelm_generated_sentences = tokenizer(self.kelm_generated_sentences,
                                                      max_length=max_seq_length,
                                                      padding='max_length',
                                                      truncation=True,
                                                      return_tensors="pt")

    def __getitem__(self, idx):
        return self.kelm_generated_sentences[idx]

    def __len__(self):
        return len(self.kelm_generated_sentences)

    def __iter__(self):
        for sentence in self.kelm_generated_sentences:
            yield sentence

    