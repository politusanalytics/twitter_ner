"""
Training part of the code is mostly from
https://github.com/alisafaya/mukayese/blob/main/ner/bert-crf/train_flair.py
"""

import torch, flair
from tqdm import tqdm
import sys

from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus

from torch.optim.lr_scheduler import OneCycleLR

from data import get_examples
from conlleval import evaluate2

# define columns
columns = {0: 'text', 1: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = sys.argv[1]
hf_model = sys.argv[2]
# hf_model = 'dbmdz/bert-base-turkish-cased'
# hf_model = 'cardiffnlp/twitter-xlm-roberta-base'
train = False
test = True
predict = False

test_filename = data_folder + "/wikiann_test.txt"
flair.device = torch.device('cuda')
output_folder = f"flert-{data_folder.replace('/', '.' ) }-crf-{hf_model.replace('/', '.' ) }"


if train:
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    embeddings = TransformerWordEmbeddings(
        model=hf_model,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=True,
        use_context=False,
        respect_document_boundaries=False,
    )

    tag_type="ner"
    tag_dictionary = corpus.make_tag_dictionary(tag_type)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        use_rnn=False,
        reproject_embeddings=False,
    )

    trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

    trainer.train(
        output_folder,
        learning_rate=5.0e-5,
        mini_batch_size=16,
        mini_batch_chunk_size=1,
        max_epochs=10,
        scheduler=OneCycleLR,
        embeddings_storage_mode='none',
        weight_decay=0.,
        train_with_dev=False
    )

elif test:
    tagger = SequenceTagger.load(output_folder + "/best-model.pt")

    def test_model(tagger, test_filename, label_to_idx, idx_to_label):
        test_examples = get_examples(test_filename, label_to_idx=label_to_idx)
        all_labels = []
        all_predictions = []
        for example in test_examples:
            tokens = example[0]
            labels = example[1]
            all_labels.extend([idx_to_label[x] for x in labels])

            flair_sentence = Sentence(tokens)
            tagger.predict(flair_sentence)

            preds = ["O"] * len(tokens)
            for span in flair_sentence.get_spans("ner"):
                curr_span_tok_idxs = [int(pos)-1 for pos in span.position_string.split("-")]
                curr_pred = span.tag
                for i, tok_idx in enumerate(curr_span_tok_idxs):
                    if i == 0:
                        preds[tok_idx] = "B-" + curr_pred
                    else:
                        preds[tok_idx] = "I-" + curr_pred

            all_predictions.extend(preds)

        (precision, recall, f1), _ = evaluate2(all_labels, all_predictions)
        return precision, recall, f1

    label_to_idx = {k.decode("utf-8"):v for k, v in tagger.tag_dictionary.item2idx.items()}
    idx_to_label = [x.decode("utf-8") for x in tagger.tag_dictionary.idx2item]
    precision, recall, f1 = test_model(tagger, test_filename, label_to_idx, idx_to_label)
    print(precision, recall, f1)

elif predict:
    tagger = SequenceTagger.load(output_folder + "/final-model.pt")

    def model_predict(tagger, test_filename, idx_to_label):
        test_examples = get_examples(test_filename, with_label=False)
        all_predictions = []
        for i, example in enumerate(test_examples):
            tokens = example[0]

            flair_sentence = Sentence(tokens)
            tagger.predict(flair_sentence)

            preds = ["O"] * len(tokens)
            for span in flair_sentence.get_spans("ner"):
                curr_span_tok_idxs = [int(pos)-1 for pos in span.position_string.split("-")]
                curr_pred = span.tag
                for i, tok_idx in enumerate(curr_span_tok_idxs):
                    if i == 0:
                        preds[tok_idx] = "B-" + curr_pred
                    else:
                        preds[tok_idx] = "I-" + curr_pred

            # TODO: write preds to file

        return
