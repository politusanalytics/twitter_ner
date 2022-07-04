import json
import torch
# import pandas as pd

class TransformersData(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_seq_length=512, has_token_type_ids=False,
                 with_label=True):
        self.examples = examples
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.with_label = with_label
        self.has_token_type_ids = has_token_type_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if self.with_label:
            all_labels = [ex[1]]
        else: # tokenize_and_align_labels function returns mock labels in this case
            all_labels = []

        return tokenize_and_align_labels(self.tokenizer, [ex[0]], all_labels=all_labels,
                                         max_length=self.max_seq_length,
                                         has_token_type_ids=self.has_token_type_ids)

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(tokenizer, texts, all_labels=[], max_length=512,
                              has_token_type_ids=False):
    t = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length,
                  is_split_into_words=True,return_tensors="pt")

    labels = []
    for i, text in enumerate(texts):
        word_ids = t.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-1)
            elif word_idx != previous_word_idx:
                if all_labels:
                    label_ids.append(all_labels[i][word_idx])
                else: # mock label
                    label_ids.append(0) # for "O" label
            else:
                label_ids.append(-1)
            previous_word_idx = word_idx

        labels.append(label_ids)

    labels = torch.LongTensor(labels)

    if labels.shape[0] == 1:
        if has_token_type_ids:
            return labels.squeeze(0), t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0), t["token_type_ids"].squeeze(0)
        else:
            return labels.squeeze(0), t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0)

    if has_token_type_ids:
        return labels, t["input_ids"], t["attention_mask"], t["token_type_ids"]
    else:
        return labels, t["input_ids"], t["attention_mask"]

def get_examples_BIO(filename, with_label=True, label_to_idx={}):
    with open(filename, "r", encoding="utf-8") as f:
        samples = f.read().split("\n\n")

    examples = []
    for sample in samples:
        if not sample.strip():
            continue

        lines = sample.splitlines()
        tokens = []
        labels = []
        for (i, line) in enumerate(lines):
            if with_label:
                token, label = line.split()
                labels.append(label_to_idx[label])
            else:
                token = line.strip()

            tokens.append(token)

        if with_label:
            examples.append([tokens, labels])
        else:
            examples.append([tokens])

    return examples

def get_examples(filename, with_label=True, label_to_idx={}):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    examples = []
    for (i, line) in enumerate(lines):
        line = json.loads(line)
        # tokens = [tok for toks in line["tokens"] for tok in toks] # flatten 2d array
        tokens = line["tokens"]
        if with_label:
            if len(label_to_idx) > 0:
                # token_labels = [label_to_idx[lab] for labs in line["token_labels"] for lab in labs] # flatten 2d array
                token_labels = [label_to_idx[lab] for lab in line["token_labels"]]
            else:
                # token_labels = [lab for labs in line["token_labels"] for lab in labs] # flatten 2d array
                token_labels = line["token_labels"]
            examples.append([tokens, token_labels])
        else:
            examples.append([tokens])

    return examples
