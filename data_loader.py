import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import re
import numpy as np
from cleopatra import LanguageModeler

def read_data(path: str) -> List[str]:
    """Read data from csv file, split into two groups (word, ner)
        Args:
        -----
          path: str, address of file.

        Returns:
        --------
          dict: dictionary of words as keys and part of speech as value

    """
    l_words = []
    l_ner = []

    with open(path) as file:
        for i, line in enumerate(file):
            try:
                pairs = line.split('\t')
                word, ner = pairs[0], re.sub(r'\W', '', pairs[1])
                l_words.append(word), l_ner.append(ner)
            except IndexError:
                pass
            # vocab = set(test_sentence)
            # word_to_ix = {word: i for i, word in enumerate(vocab)}
    return (l_words, l_ner)



def main():
    train_data, labels = read_data(path=r"C:\Users\kofma\Desktop\datascience\deep\rachel_vova\ex2\ner\train")
    # we should tokenize the input, but we will ignore that for now
    # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

    CONTEXT_SIZE = 3
    EMBEDDING_DIM = 10

    arr_labels = np.array(labels)
    condlist = [arr_labels == 'LOC', arr_labels == 'PER', arr_labels == 'O', arr_labels == 'ORG', arr_labels == 'MISC']
    choicelist = [0, 1, 2, 3, 4]
    labels = np.select(condlist, choicelist)
    print(labels)
    # maybe add double __START__ in the begining of the corpus and __END__ in the end,
    #convert all numbers to __NUMBERS__, maybe in the test we doesn't have vocab words must
    #to add to vocab __UNKNOW__
    trigrams = [([train_data[i], train_data[i + 1], train_data[i + 2]],labels[i + 1])
                for i in range(len(train_data) - 2)]
    print(trigrams[:3])

    vocab = set(train_data)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    losses = []
    loss_function = nn.NLLLoss()
    model = LanguageModeler(len(vocab), EMBEDDING_DIM, hidden_layer=64, num_of_labels=len(set(labels)), context_size = CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!

if __name__ == "__main__":
    main()