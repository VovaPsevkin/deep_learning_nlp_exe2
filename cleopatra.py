import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_layer, num_of_labels, context_size):
        super(LanguageModeler, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.fc = nn.Linear(context_size * embedding_dim, hidden_layer)
        self.out = nn.Linear(hidden_layer, num_of_labels)

    def forward(self, inputs):
        # (1) input layer
        inputs = inputs

        # (2) embedding layer
        embeds = self.embeddings(inputs)
        embeds = embeds.view((1, -1))
        # embeds = torch.flatten(embeds,start_dim=1)
        # (3) hidden layer
        hidden_layer = self.fc(embeds)
        activation = F.tanh(hidden_layer)

        # (4) output layer --> it's probability of label (ner or pos).
        out = self.out(activation)
        # log_probs = F.softmax(out, dim=1)
        out = F.softmax(out, dim = 1)

        return out


