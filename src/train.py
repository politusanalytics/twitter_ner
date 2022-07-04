import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup
from data import get_examples, TransformersData, tokenize_and_align_labels, get_examples_BIO
from torch.utils.data import DataLoader
import numpy as np
import time
from conlleval import evaluate2
import random
from tqdm import tqdm
import json
import sys
# import pandas as pd


# INPUTS
pretrained_transformers_model = sys.argv[1] # For example: "xlm-roberta-base"
seed = int(sys.argv[2])

# MUST SET THESE VALUES
repo_path = "/path/to/this/repo"
train_filename = f"{repo_path}/data/train_examples.txt"
test_filename = f"{repo_path}/data/test_examples.txt"
# test_filename = f"{repo_path}/data/file_to_predict.txt"
label_list = ['O', 'B-LOC', 'B-ORG', 'B-PER',
              'I-LOC', 'I-ORG', 'I-PER'] # IMPORTANT NOTE: Always start with "O". I use 0 as index
                                         # for "O" label in other parts of this script.
only_test = False # Only perform testing
predict = False # Predict instead of testing
only_predict_up_to_max_seq_length = True # If True, does not predict all tokens in the given document
                                         # if its tokenized length exceeds max_seq_length
max_seq_length = 512
batch_size = 16 # NOTE: May break if last batch has only one document
dev_ratio = 0.1

# SETTINGS
learning_rate = 2e-5
dev_metric = "f1"
num_epochs = 10
dev_set_splitting = "random" # random, or any filename
use_gpu = True
device_ids = [4, 5, 6, 7]
model_path = "{}_{}.pt".format(pretrained_transformers_model.replace("/", "_"), seed)

# optional, used in testing
classifier_path = ""# repo_path + "/models/best_models/classifier_sentence-transformers_paraphrase-xlm-r-multilingual-v1_44.pt"
encoder_path = ""#repo_path + "/models/best_models/encoder_sentence-transformers_paraphrase-xlm-r-multilingual-v1_44.pt"
if not classifier_path:
    classifier_path =  repo_path + "/models/classifier_" + model_path
if not encoder_path:
    encoder_path =  repo_path + "/models/encoder_" + model_path


# Don't change anything below this line
trans_config = AutoConfig.from_pretrained(pretrained_transformers_model)
has_token_type_ids = trans_config.type_vocab_size > 1

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:%d"%(device_ids[0]))
else:
    device = torch.device("cpu")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

idx_to_label = {}
label_to_idx = {}
for (i, label) in enumerate(label_list):
    idx_to_label[i] = label
    label_to_idx[label] = i

tokenizer = None
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)


class MLP(torch.nn.Module): # only 1 layer
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout else None
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.linear2(x)
        return x

def get_batches_for_examples(examples, tokenizer, with_label=True):
    all_batches = []
    org_token_labels = []
    for i in range(0, len(examples), batch_size):
        texts = [ex[0] for ex in examples[i:i+batch_size]] # no need to check for array bounds
        labels = []
        if with_label:
            labels = [ex[1] for ex in examples[i:i+batch_size]]
            org_token_labels.append(labels)

        batch = tokenize_and_align_labels(tokenizer, texts, all_labels=labels,
                                          max_length=max_seq_length,
                                          has_token_type_ids=has_token_type_ids)

        all_batches.append(batch)

    if with_label:
        return all_batches, org_token_labels

    return all_batches

def test_whole_set(tokenizer, encoder, classifier, filename, lang):
    test_examples = get_examples_BIO(filename, label_to_idx=label_to_idx)
    test_batches, test_org_token_labels = get_batches_for_examples(test_examples, tokenizer)

    print("***** %s TEST RESULTS *****" %lang)
    result, test_loss = test_model(encoder, classifier, test_batches,
                                   org_token_labels=test_org_token_labels)
    result["test_loss"] = test_loss

    for key in sorted(result.keys()):
        print("  %s = %.6f" %(key, result[key]))

    print("%s TEST SCORE: %.6f" %(lang, result[dev_metric]))

def test_model(encoder, classifier, batches, org_token_labels=[]):
    all_preds = []
    all_label_ids = []
    eval_loss = 0
    nb_eval_steps = 0
    for val_step, batch in enumerate(batches):
        if has_token_type_ids:
            label_ids, input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        else:
            label_ids, input_ids, input_mask = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask)[0]

        with torch.no_grad():
            out = classifier(embeddings)
            tmp_eval_loss = criterion(out.view(-1, len(label_list)), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()

        out = out.detach().cpu().numpy()
        preds = np.argmax(out, axis=2) # preds: [batch_size, max_seq_length]
        labels = label_ids.detach().cpu().numpy()

        if org_token_labels: # if these are given we can do length fix
            curr_batch_org_labels = org_token_labels[val_step]
            for i in range(len(curr_batch_org_labels)): # for each document
                curr_doc_preds = preds[i,:]
                curr_doc_labels = labels[i,:]
                curr_doc_org_labels = curr_batch_org_labels[i]

                # get rid of extra subwords, CLS, SEP, PAD
                curr_doc_preds = curr_doc_preds[curr_doc_labels != -1].tolist()
                curr_doc_preds.extend([0] * (len(curr_doc_org_labels) - len(curr_doc_preds))) # 0 is for "O" label
                all_preds.extend(curr_doc_preds)
                all_label_ids.extend(curr_doc_org_labels)

        else:
            all_preds.extend(preds[labels != -1].tolist())
            all_label_ids.extend(labels[labels != -1].tolist())

        nb_eval_steps += 1

    f1 = 0.0
    if all_preds:
        (precision, recall, f1), _ = evaluate2([idx_to_label[x] for x in all_label_ids], [idx_to_label[x] for x in all_preds])
        precision /= 100
        recall /= 100
        f1 /= 100

    eval_loss /= nb_eval_steps
    result = {"precision": precision,
              "recall": recall,
              "f1": f1}

    return result, eval_loss

def postprocess_labels(tokens, token_labels):
    spans = []
    prev_token_label = "O"
    start_idx = 0
    for token_idx, token_label in enumerate(token_labels):
        if token_label == "O" and prev_token_label != "O":
            spans.append([prev_token_label, " ".join(tokens[start_idx:token_idx])])
            prev_token_label = "O"

        elif token_label.startswith("B-"):
            if prev_token_label != "O":
                spans.append([prev_token_label, " ".join(tokens[start_idx:token_idx])])

            start_idx = token_idx
            prev_token_label = token_label[2:]

        elif token_label.startswith("I-"):
            if prev_token_label == "O":
                start_idx = token_idx
            else:
                if prev_token_label != token_label[2:]:
                    spans.append([prev_token_label, " ".join(tokens[start_idx:token_idx])])
                    start_idx = token_idx

            prev_token_label = token_label[2:]

    return spans

def model_predict(encoder, classifier, examples):
    all_example_preds = []
    batches = get_batches_for_examples(divided_examples, tokenizer, with_label=False)

    for batch_idx, batch in enumerate(batches):
        if has_token_type_ids:
            mock_label_ids, input_ids, input_mask, token_type_ids = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        else:
            mock_label_ids, input_ids, input_mask = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            embeddings = encoder(input_ids, attention_mask=input_mask)[0]

        out = classifier(embeddings)
        scores = out.detach().cpu().numpy()
        preds = np.argmax(scores, axis=-1)

        mock_label_ids = mock_label_ids.numpy()
        preds = preds[mock_label_ids != -1].tolist()

        # Number of tokens we process may change, since we cut documents to MAX_SEQ_LEN (after tokenizing)
        pred_index = 0
        for i in range(mock_label_ids.shape[0]):
            curr_mock_label_ids = mock_label_ids[i, :]
            curr_predicted_token_length = len(curr_mock_label_ids[curr_mock_label_ids != -1].tolist())

            curr_preds = [idx_to_label[pred]
                          for pred in preds[pred_index:pred_index+curr_predicted_tok_length]]
            pred_index += curr_predicted_tok_length

            curr_real_token_length = len(examples[batch_idx*batchsize + i][0])
            curr_preds += ["O"] * (curr_real_token_length-curr_predicted_token_length)
            all_example_preds.append(curr_preds)

    return all_example_preds

def model_predict_with_chunking(encoder, classifier, examples):
    # Divide examples to process all of the tokens in an example
    all_example_lengths = []
    all_missing_token_indexes = []
    divided_examples = []
    for example in examples:
        toks = example[0]
        tok_index = 0
        # curr_missing_token_indexes = [] # some tokens are completely discarded by the tokenizer

        t = tokenizer.encode_plus(toks, add_special_tokens=False, is_split_into_words=True,
                                  return_tensors='pt') # no max length

        word_ids = t.word_ids()
        # TODO: Do all encoder add CLS and SEP tokens? If not this is wrong!
        previous_word_idx = -1
        for i in range(0, len(word_ids), max_seq_length-2):
            curr_word_ids = word_ids[i:i+(max_seq_length-2)]
            token_counter = 0
            for word_idx in curr_word_ids:
                if word_idx == previous_word_idx+1:
                    token_counter += 1
                # elif word_idx > previous_word_idx:
                #     ipdb.set_trace()
                #     token_counter += word_idx - previous_word_idx
                #     curr_missing_token_indexes.extend([idx for idx in range(previous_word_idx+1, word_idx)])

                previous_word_idx = word_idx

            curr_toks = toks[tok_index:tok_index+token_counter]
            tok_index += token_counter
            divided_examples.append([curr_toks])

        all_example_lengths.append(len(toks))
        # all_example_lengths.append(len(toks) - len(curr_missing_token_indexes))
        # all_missing_token_indexes.append(curr_missing_token_indexes)

    # Get batches of these divided examples
    divided_batches = get_batches_for_examples(divided_examples, tokenizer, with_label=False)

    # Run the batches
    all_preds = []
    for i,batch in enumerate(divided_batches):
        if has_token_type_ids:
            mock_label_ids, input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch)
            embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
        else:
            mock_label_ids, input_ids, input_mask = tuple(t.to(device) for t in batch)
            if len(input_ids.shape) == 1:
                mock_label_ids = mock_label_ids.unsqueeze(0)
                input_ids = input_ids.unsqueeze(0)
                input_mask = input_mask.unsqueeze(0)
            embeddings = encoder(input_ids, attention_mask=input_mask)[0]

        with torch.no_grad():
            out = classifier(embeddings)

        out = out.detach().cpu().numpy()
        preds = np.argmax(out, axis=-1) # [batch_size, max_seq_length]
        mock_label_ids = mock_label_ids.detach().cpu().numpy()
        all_preds.extend([idx_to_label[pred] for pred in preds[mock_label_ids != -1].tolist()])

    # Assign each example its labels
    curr_index = 0
    all_example_preds = []
    for example_index, curr_example_length in enumerate(all_example_lengths):
        curr_example_preds = all_preds[curr_index:curr_index+curr_example_length]
        # # Fill missing indexes with "O" label
        # curr_missing_token_indexes = all_missing_token_indexes[example_index]
        # for missing_tok_index in curr_missing_token_indexes:
        #     curr_example_preds.insert(missing_tok_index, "O")

        all_example_preds.append(curr_example_preds)
        curr_index += curr_example_length

        assert(len(curr_example_preds) == len(examples[example_index][0]))

    return all_example_preds


def build_model(train_examples, dev_examples, pretrained_model, n_epochs=10, curr_model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    encoder = AutoModel.from_pretrained(pretrained_model)
    classifier = MLP(encoder.config.hidden_size, encoder.config.hidden_size * 4, len(label_list))

    train_dataset = TransformersData(train_examples, tokenizer, max_seq_length=max_seq_length, has_token_type_ids=has_token_type_ids)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    dev_batches, dev_org_token_labels = get_batches_for_examples(dev_examples, tokenizer)

    classifier.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
    encoder.to(device)

    optimizer = torch.optim.AdamW(list(classifier.parameters()) + list(encoder.parameters()), lr=learning_rate)
    num_train_steps = int(len(train_examples) / batch_size * num_epochs)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps = 0,
    #                                             num_training_steps = num_train_steps)

    best_score = -1e6
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0
        encoder.train()
        classifier.train()

        print("Starting Epoch %d"%(epoch+1))
        global_step = 0
        train_loss = 0.0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            if has_token_type_ids:
                label_ids, input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch)
                embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
            else:
                label_ids, input_ids, input_mask = tuple(t.to(device) for t in batch)
                embeddings = encoder(input_ids, attention_mask=input_mask)[0]

            out = classifier(embeddings)
            loss = criterion(out.view(-1, len(label_list)), label_ids.view(-1))

            loss.backward()
            global_step += 1
            nb_tr_steps += 1

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            encoder.zero_grad()
            classifier.zero_grad()

            train_loss += loss.item()

        train_loss /= nb_tr_steps
        elapsed = time.time() - start_time

        # Validation
        encoder.eval()
        classifier.eval()
        print("***** Epoch " + str(epoch + 1) + " *****")
        result, val_loss = test_model(encoder, classifier, dev_batches,
                                      org_token_labels=dev_org_token_labels)
        result["train_loss"] = train_loss
        result["dev_loss"] = val_loss
        result["elapsed"] = elapsed
        for key in sorted(result.keys()):
            print("  %s = %.6f" %(key, result[key]))

        print("Val score: %.6f" %result[dev_metric])

        if result[dev_metric] > best_score:
            print("Saving model!")
            torch.save(classifier.state_dict(), repo_path + "/models/classifier_" + curr_model_path)
            encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # To handle multi gpu
            torch.save(encoder_to_save.state_dict(), repo_path + "/models/encoder_" + curr_model_path)
            best_score = result[dev_metric]

        print("------------------------------------------------------------------------")

    return encoder, classifier

if __name__ == '__main__':
    train_examples = get_examples_BIO(train_filename, label_to_idx=label_to_idx)
    random.shuffle(train_examples)
    if dev_set_splitting == "random":
        dev_split = int(len(train_examples) * dev_ratio)
        dev_examples = train_examples[:dev_split]
        train_examples = train_examples[dev_split:]
    else: # it's a custom filename
        dev_examples = get_examples_BIO(dev_set_splitting, label_to_idx=label_to_idx)

    if not only_test:
        encoder, classifier = build_model(train_examples, dev_examples, pretrained_transformers_model, n_epochs=num_epochs, curr_model_path=model_path)
        classifier.load_state_dict(torch.load(repo_path + "/models/classifier_" + model_path))
        encoder.module.load_state_dict(torch.load(repo_path + "/models/encoder_" + model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_transformers_model)
        encoder = AutoModel.from_pretrained(pretrained_transformers_model)
        classifier = MLP(encoder.config.hidden_size, encoder.config.hidden_size * 4, len(label_list))

        classifier.to(device)
        encoder.to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))

        if torch.cuda.device_count() > 1 and device.type == "cuda" and len(device_ids) > 1:
            encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    encoder.eval()
    classifier.eval()

    if predict:
        # TODO: predict BIO file as well
        test_examples = get_examples(test_filename, with_label=False, label_to_idx=label_to_idx)

        if only_predict_up_to_max_seq_length:
            all_example_preds = model_predict(encoder, classifier, test_examples)
        else:
            all_example_preds = model_predict_with_chunking(encoder, classifier, test_examples)

        with open(test_filename, "r", encoding="utf-8") as f:
            test = [json.loads(line) for line in f.read().splitlines()]

        with open(repo_path + "/out.json", "w", encoding="utf-8") as g:
            for i, t in enumerate(test):
                curr_preds = all_example_preds[i]
                curr_spans = postprocess_tokens_and_labels(t["tokens"], curr_preds)
                t["prediction"] = curr_spans
                g.write(json.dumps(t) + "\n")

    else:
        test_whole_set(tokenizer, encoder, classifier, test_filename, "TR")
