import torch

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
