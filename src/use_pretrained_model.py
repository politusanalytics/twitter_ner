from transformers import AutoModel, AutoTokenizer, AutoConfig
from data import tokenize_and_align_labels
from commons import postprocess_labels, MLP
import numpy as np
import sys
import json
import torch
import gzip

# Globals
IDX_TO_LABEL = ['O', 'B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER']
MODEL_NAME = "cardiffnlp_twitter-xlm-roberta-base_combined_42.pt"
PRETRAINED_TRANSFORMERS_MODEL = "cardiffnlp/twitter-xlm-roberta-base"
DEVICE = torch.device("cuda")
BATCHSIZE = 512
MAX_SEQ_LEN = 128 # more than enough for tweets

# Load model
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TRANSFORMERS_MODEL)
config = AutoConfig.from_pretrained(PRETRAINED_TRANSFORMERS_MODEL)
has_token_type_ids = config.type_vocab_size > 1

encoder = AutoModel.from_pretrained(PRETRAINED_TRANSFORMERS_MODEL)
encoder.to(DEVICE)
encoder.load_state_dict(torch.load("/home/omutlu/twitter_ner/models/encoder_" + MODEL_NAME, map_location=DEVICE))
classifier = MLP(encoder.config.hidden_size, encoder.config.hidden_size * 4, len(IDX_TO_LABEL))
classifier.to(DEVICE)
classifier.load_state_dict(torch.load("/home/omutlu/twitter_ner/models/classifier_" + MODEL_NAME, map_location=DEVICE))

encoder = torch.nn.DataParallel(encoder)
encoder.eval()
classifier.eval()

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def model_predict(batch):
    if has_token_type_ids:
        mock_label_ids, input_ids, input_mask, token_type_ids = batch
        input_ids = input_ids.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[0]
    else:
        mock_label_ids, input_ids, input_mask = batch
        input_ids = input_ids.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        embeddings = encoder(input_ids, attention_mask=input_mask)[0]

    out = classifier(embeddings)
    scores = out.detach().cpu().numpy()
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
    # Inputs
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    output_file = open(output_filename, "w", encoding="utf-8")

    with gzip.open(input_filename, "rt", encoding="utf-8") as input_file:

        curr_batch = []
        for i, line in enumerate(input_file):
            line = json.loads(line)
            twt_id_str = list(line.keys())[0]
            twt_txt = preprocess(line[twt_id_str])
            twt_txt = twt_txt.strip().split() # TODO: nltk here?
            if len(twt_txt) > 0:
                curr_batch.append({"twt_id_str": twt_id_str, "twt_txt": twt_txt})

            if len(curr_batch) == BATCHSIZE:
                texts = [d["twt_txt"] for d in curr_batch]
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
                    curr_tokens = curr_d.pop("twt_txt")

                    spans = postprocess_labels(curr_tokens, curr_preds)
                    # if spans:
                    curr_d["entities"] = spans
                    output_file.write(json.dumps(curr_d) + "\n")

                curr_batch = []

    if len(curr_batch) != 0:
        texts = [d["twt_txt"] for d in curr_batch]
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
            curr_tokens = curr_d.pop("twt_txt")

            spans = postprocess_labels(curr_tokens, curr_preds)
            # if spans:
            curr_d["entities"] = spans
            output_file.write(json.dumps(curr_d) + "\n")

    output_file.close()
