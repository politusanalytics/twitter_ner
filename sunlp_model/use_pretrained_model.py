from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
# from data import tokenize_and_align_labels
# from commons import postprocess_labels
import numpy as np
import json
import torch

# Globals
PRETRAINED_TRANSFORMERS_MODEL = "busecarik/bert-loodos-sunlp-ner-turkish"
DEVICE = torch.device("cuda")
BATCHSIZE = 512
MAX_SEQ_LEN = 128 # more than enough for tweets

# Load model
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TRANSFORMERS_MODEL)
config = AutoConfig.from_pretrained(PRETRAINED_TRANSFORMERS_MODEL)
has_token_type_ids = config.type_vocab_size > 1
IDX_TO_LABEL = config.id2label

model = AutoModelForTokenClassification.from_pretrained(PRETRAINED_TRANSFORMERS_MODEL)
model.to(DEVICE)
model = torch.nn.DataParallel(model)
model.eval()


import pymongo
# Connect to mongodb
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["politus_twitter"]
tweet_col = db["tweets"]


# # Preprocess text (username and link placeholders)
# def preprocess(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

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

def postprocess_labels(tokens, token_labels):
    spans = []
    prev_token_label = "O"
    start_idx = 0
    for token_idx, token_label in enumerate(token_labels):
        if token_label == "O" and prev_token_label != "O":
            spans.append([prev_token_label, " ".join(tokens[start_idx:token_idx]), (start_idx, token_idx-1)])
            prev_token_label = "O"

        elif token_label.startswith("B-"):
            if prev_token_label != "O":
                spans.append([prev_token_label, " ".join(tokens[start_idx:token_idx]), (start_idx, token_idx-1)])

            start_idx = token_idx
            prev_token_label = token_label[2:]

        elif token_label.startswith("I-"):
            if prev_token_label == "O":
                start_idx = token_idx
            else:
                if prev_token_label != token_label[2:]:
                    spans.append([prev_token_label, " ".join(tokens[start_idx:token_idx]), (start_idx, token_idx-1)])
                    start_idx = token_idx

            prev_token_label = token_label[2:]

    return spans

def model_predict(batch):
    if has_token_type_ids:
        mock_label_ids, input_ids, input_mask, token_type_ids = batch
        input_ids = input_ids.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        out = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
    else:
        mock_label_ids, input_ids, input_mask = batch
        input_ids = input_ids.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        out = model(input_ids, attention_mask=input_mask)

    # ipdb.set_trace()
    scores = out.logits.detach().cpu().numpy()
    preds = np.argmax(scores, axis=-1)

    mock_label_ids = mock_label_ids.numpy()
    preds = preds[mock_label_ids != -1].tolist()

    # Number of tokens we process may change, since we cut documents to MAX_SEQ_LEN (after tokenizing)
    token_lengths = []
    for i in range(mock_label_ids.shape[0]):
        curr_mock_label_ids = mock_label_ids[i, :]
        curr_token_length = len(curr_mock_label_ids[curr_mock_label_ids != -1].tolist())
        token_lengths.append(curr_token_length)

    return preds, token_lengths

if __name__ == "__main__":
    output_file = open("random_10k_ner_preds.json", "w", encoding="utf-8")
    query = {"text": {"$nin": ["", None]}}
    # tweets_to_predict = tweet_col.find(query, ["_id", "text"])

    # For random selection
    tweets_to_predict = tweet_col.aggregate([
        {"$match": query},
        {"$sample": { "size": 10000 }}
    ])

    curr_batch = []
    for row in tweets_to_predict:
        id_str = row["_id"]
        text = row["text"] # preprocess(row["text"])
        text = text.strip().split() # TODO: nltk here?
        if len(text) > 0:
            curr_batch.append({"id_str": id_str, "text": text})

        if len(curr_batch) == BATCHSIZE:
            texts = [d["text"] for d in curr_batch]
            inputs = tokenize_and_align_labels(tokenizer, texts, all_labels=[],
                                               max_length=MAX_SEQ_LEN,
                                               has_token_type_ids=has_token_type_ids)

            preds, token_lengths = model_predict(inputs)
            assert(len(preds) == sum(token_lengths))
            pred_index = 0
            for j, curr_tok_length in enumerate(token_lengths):
                curr_preds = [IDX_TO_LABEL[pred] for pred in preds[pred_index:pred_index+curr_tok_length]]
                pred_index += curr_tok_length

                curr_d = curr_batch[j]
                spans = postprocess_labels(curr_d["text"], curr_preds)
                curr_d["preds"] = spans
                output_file.write(json.dumps(curr_d) + "\n")

            curr_batch = []

    if len(curr_batch) != 0:
        texts = [d["text"] for d in curr_batch]
        inputs = tokenize_and_align_labels(tokenizer, texts, all_labels=[],
                                           max_length=MAX_SEQ_LEN,
                                           has_token_type_ids=has_token_type_ids)

        preds, token_lengths = model_predict(inputs)
        pred_index = 0
        assert(len(preds) == sum(token_lengths))
        for j, curr_tok_length in enumerate(token_lengths):
            curr_preds = [IDX_TO_LABEL[pred] for pred in preds[pred_index:pred_index+curr_tok_length]]
            pred_index += curr_tok_length

            curr_d = curr_batch[j]
            spans = postprocess_labels(curr_d["text"], curr_preds)
            # if spans:
            curr_d["preds"] = spans
            output_file.write(json.dumps(curr_d) + "\n")

    output_file.close()
