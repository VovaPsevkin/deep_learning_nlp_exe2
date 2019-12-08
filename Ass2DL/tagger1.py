import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
from random import shuffle


# PARAMETERS FOR BOTH TAGGING TASKS
# Embedding dimension
EMBEDDING_DIM = 50
pos_params = {
    "hidden": 120,
    "batch_size": 10000,
    "lr": 0.01,
    "wd": 1e-5,
    "epochs": 20
}

ner_params = {
    "hidden": 40,
    "batch_size": 1000,
    "lr": 0.01,
    "wd": 1e-5,
    "epochs": 20
}
# Choose tagging task from command line
if sys.argv[1] == 'pos':
    params = pos_params
elif sys.argv[1] == 'ner':
    params = ner_params

# PREPROCESSING DATA


def get_vocab_and_tagset(train_file):
    """
    read all words and their tags in the corpus and get vocabulary and tagset
    return: vocabulary and tagset sets, and word-index, tag-index dictionaries
    """
    # Initialize with padding
    words = ['*start*', '*end*']
    tags = []
    # Read corpus
    with open(train_file, 'r') as file:
        for line in file:
            # Remove end of line
            line = line.strip()
            # Not space
            if line:
                # Get word and its tag
                word = line.split()[0]
                tag = line.split()[1]
                # Add to vocabulary and tagset
                words.append(word)
                tags.append(tag)
    vocab = set(words)
    # Unknown title for words not in corpus
    vocab.add('*UNK*')
    tagset = set(tags)
    # Get relevant dictionaries
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    tag_to_ix = {tag: i for i, tag in enumerate(tagset)}
    ix_to_tag = {i: tag for i, tag in enumerate(tagset)}
    return vocab, tagset, word_to_ix, tag_to_ix, ix_to_tag


def load_data(file_name):
    """
    Read (word, tag) tokens from file
    """
    tokens = []
    # Open the file
    with open(file_name, 'r') as file:
        for line in file:
            # Remove '\n'
            line = line.strip()
            if line:
                # Get word and tag and add token
                word = line.split()[0]
                tag = line.split()[1]
                tokens.append((word, tag))
            # Space between sentences
            else:
                tokens.append("")
    return tokens


def get_contexts(tokens, word_to_ix, tag_to_ix):
    """
    Get windows of words with middle word tag
    return: list of contexts in ([ppw, pw, w, nw, nnw], tag) format (converted to indexes)
    """
    contexts = []
    # Read the tokens
    for i in range(len(tokens)):
        # Two words before
        try:
            ppw = tokens[i-2][0]
        except IndexError:
            ppw = '*start*'
        # Previuos word
        try:
            pw = tokens[i-1][0]
        except IndexError:
            pw = '*start*'
        # There is '.' in the end of previous sentence
        if pw == '*start*':
            ppw = '*start*'
        # Next word
        try:
            nw = tokens[i+1][0]
        except IndexError:
            nw = '*end*'
        # Two words after
        try:
            nnw = tokens[i+2][0]
        except IndexError:
            nnw = '*end*'
        # There is another word at the beginning of next sentence
        if nw == '*end*':
            nnw = '*end*'
        # Get word and tag
        try:
            word = tokens[i][0]
            tag = tokens[i][1]
        # Space between sentences
        except IndexError:
            continue
        # Convert context to indexes using relevant dictionaries
        tag = tag_to_ix[tag]
        context = ([ppw, pw, word, nw, nnw], tag)
        for i in range(len(context[0])):
            try:
                context[0][i] = word_to_ix[context[0][i]]
            # Unknown word
            except KeyError:
                context[0][i] = word_to_ix['*UNK*']
        # Append the current context
        contexts.append(context)
    return contexts


def get_batches(contexts):
    """
    We use batching to run fast on GPU device. get batches from the contexts
    """
    # Shuffle the data
    shuffle(contexts)
    # Initializations
    batches = []
    num_of_batches = 0
    index = 0
    # Make batches list
    while index < len(contexts) - params['batch_size'] + 1:
        # New batch
        batches.append([[], []])
        for i in range(params['batch_size']):
            # Get context and its tag
            batches[num_of_batches][0].append(contexts[index][0])
            batches[num_of_batches][1].append(contexts[index][1])
            index += 1
        num_of_batches += 1
    # We miss the last contexts, but we assume multiple iterations and shuffling
    # (because we use fixed batch length)
    return batches

# THE MODEL


class WindowBaseTagger(nn.Module):
    """
    The model
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        """
        Initialize the model matrices
        """
        super(WindowBaseTagger, self).__init__()
        # Embedding matrix
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # First matrix of MLP
        self.linear1 = nn.Linear(5*embedding_dim, hidden_dim)
        # Second matrix of MLP
        self.linear2 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs):
        """
        Get model predictions from input
        """
        # Embedding and concat
        embeds = self.embedding(inputs)
        #print('Before:',embeds.shape)
        #embeds = embeds.view((-1, 5*embeds.shape[-1]))
        embeds = torch.flatten(embeds, start_dim=1)
        #print('After', embeds.shape)
        # Hidden later and tanh
        out = torch.tanh(self.linear1(embeds))
        # Output layer
        out = self.linear2(out)
        # Softmax
        probs = F.softmax(out, dim=1)
        return probs


# class WindowBaseTagger(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, hidden_layer, num_of_labels):
#         super(WindowBaseTagger, self).__init__()
#
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#
#         self.fc = nn.Linear(embedding_dim, hidden_layer)
#         self.out = nn.Linear(hidden_layer, num_of_labels)
#
#     def forward(self, inputs):
#         # (1) input layer
#         inputs = inputs
#
#         # (2) embedding layer
#         embeds = self.embeddings(inputs)
#         print('Before:',embeds.shape)
#         #t.reshape(1, 9)
#         embeds = embeds.reshape((embeds.shape[0], embeds.shape[1] * embeds.shape[2]))
#         #embeds = embeds.view((1, -1))
#         print('After', embeds.shape)
#         # (3) hidden layer
#         hidden_layer = self.fc(embeds)
#         activation = F.tanh(hidden_layer)
#
#         # (4) output layer --> it's probability of label (ner or pos).
#         out = self.out(activation)
#         #out = F.softmax(out, dim=1)
#         out = F.softmax(out)
#
#         return out

# TRAINING PROCESS


def train_epoch(contexts, model, loss_function, optimizer, device):
    """
    Train the model with one iteration on the data
    """
    total_loss = 0
    # Get data batches
    batches = get_batches(contexts)
    for batch in batches:
        # Get relevant tensors and move to GPU
        batch_contexts = batch[0]
        batch_tags = batch[1]

        batch_contexts = torch.LongTensor(batch_contexts)
        batch_tags = torch.LongTensor(batch_tags)

        batch_contexts = batch_contexts.to(device)
        batch_tags = batch_tags.to(device)

        # Get prediction from the model
        model.zero_grad()
        probs = model(batch_contexts)
        # Compute loss
        loss = loss_function(probs, batch_tags)
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Return average loss
    return total_loss / len(batches)


def train(device, contexts, vocab, tagset, dev_file, word_to_ix, tag_to_ix):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.CrossEntropyLoss()
    # Set model with relevant parameters
    model = WindowBaseTagger(len(vocab), EMBEDDING_DIM, params['hidden'], len(tagset))
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    for epoch in range(params['epochs']):
        # Train model and get loss
        loss = train_epoch(contexts, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute accuracy
        accuracy = evaluation(model, device, dev_file, word_to_ix, tag_to_ix, loss_function)
        # Print epoch and accuracy
        print("epoch: " + str(1 + epoch), "accuracy: " + str(accuracy))
    # Print train losses
    print(losses)
    return model

# EVALUATION


def ner_accuracy_metric(prediction, target, tag_to_ix):
    """
    Accuracy metric for the NER data. Ignore correctly predicted 'O' tags
    """
    # Get 'O' tag index from dictionary
    o_index = tag_to_ix['O']
    # Get relevant indexes (when the tag is not 'O' that was predicted correctly)
    indexes = [i for i in range(params['batch_size']) if not (prediction[i] == o_index and target[i] == o_index)]
    # Get relevant prediction and true tags
    check_pred = torch.tensor([prediction[index] for index in indexes])
    check_target = torch.tensor([target[index] for index in indexes])
    # Compute accuracy
    accuracy = (check_pred == check_target).sum().item() / len(indexes)
    return accuracy


def evaluation(model, device, dev_file, word_to_ix, tag_to_ix, loss_function):
    """
    Model evaluation on dev data - compute accuracy and loss
    """
    model.eval()
    # Get dev data and preprocess it
    dev_tokens = load_data(dev_file)
    dev_contexts = get_contexts(dev_tokens, word_to_ix, tag_to_ix)
    dev_batches = get_batches(dev_contexts)
    accuracy = 0
    total_loss = 0
    # Evaluate batch
    for batch in dev_batches:
        # Get relevant tensors and move to GPU
        batch_contexts = batch[0]
        batch_tags = batch[1]
        batch_contexts = torch.LongTensor(batch_contexts)
        batch_tags = torch.LongTensor(batch_tags)
        batch_contexts = batch_contexts.to(device)
        batch_tags = batch_tags.to(device)
        # Get prediction from the model
        model.zero_grad
        probs = model(batch_contexts)
        # Compute loss
        loss = loss_function(probs, batch_tags)
        total_loss += loss.item()
        # Get specific predicted tag
        prediction = torch.argmax(probs, dim=1)
        # Two accuracy metrics for two tagging tasks
        if sys.argv[1] == 'pos':
            acc = (prediction == batch_tags).sum().item() / params['batch_size']
        if sys.argv[1] == 'ner':
            acc = ner_accuracy_metric(prediction, batch_tags, tag_to_ix)
        # Get accuracy
        accuracy += acc
    # Average accuracy and loss
    accuracy /= len(dev_batches)
    total_loss /= len(dev_batches)
    # Write loss in file (for graphics later)
    with open('tag1_'+sys.argv[1]+'_loss', 'a+') as file:
        file.write(str(total_loss) + '\n')
    return accuracy

# TEST PREDICTING


def get_test_contexts(words, word_to_ix):
    """
    Get windows of words with middle word
    return: list of contexts in ([ppw, pw, word, nw, nnw], word) format (converted to indexes)
    """
    contexts = []
    # read all words
    for i in range(len(words)):
        # Space between sentences
        if not words[i]:
            contexts.append('')
            continue
        # Two words before
        if i >= 2 and words[i - 2]:
            ppw = words[i - 2]
        else:
            ppw = '*start*'
        # Previous word
        if i >= 1 and words[i - 1]:
            pw = words[i - 1]
        else:
            pw = '*start*'
        # There is '.' in the end of previous sentence
        if pw == '*start*':
            ppw = '*start*'
        # Next word
        if i < len(words) - 1 and words[i + 1]:
            nw = words[i + 1]
        else:
            nw = '*end*'
        # Two words after
        if i < len(words) - 2 and words[i + 2]:
            nnw = words[i + 2]
        else:
            nnw = '*end*'
        # There is another word at the beginning of next sentence
        if nw == '*end*':
            nnw = '*end*'
        # Get current word and context
        word = words[i]
        context = ([ppw, pw, word, nw, nnw], word)
        # Convert to indexes using relevant dictionaries
        for j in range(len(context[0])):
            try:
                context[0][j] = word_to_ix[context[0][j]]
            # Unknown word
            except KeyError:
                context[0][j] = word_to_ix['*UNK*']
        # Append to context list
        contexts.append(context)
    return contexts


def predict_test(model, test_file, word_to_ix, ix_to_tag, device):
    """
    Predict tags for the test file
    """
    words = []
    # Read words in test file
    with open(test_file, 'r') as file:
        for line in file:
            line = line.strip()
            words.append(line)
    # Preprocess data
    test_contexts = get_test_contexts(words, word_to_ix)
    # Write in prediction file
    with open('tag1_test_prediction_' + sys.argv[1], 'w+') as file:
        for context in test_contexts:
            # Space between sentences
            if not context:
                file.write('\n')
                continue
            word = context[1]
            # Predict the tag with the model
            model.zero_grad
            context = torch.tensor(context[0])
            context = context.to(device)
            probs = model(context)
            prediction = torch.argmax(probs, dim=1)
            tag = ix_to_tag[prediction.item()]
            # Different spaces in the files
            if sys.argv[1] == 'pos':
                space = " "
            elif sys.argv[1] == 'ner':
                space = '\t'
            # Write prediction in file
            file.write(word + space + tag + '\n')


def main(argv):
    # Timer
    st = time.time()
    # Set CUDA GPU device
    device = torch.device(argv[2] if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(device)
    # train/dev/test file names
    train_file = 'data/'+argv[1]+'/' + "train"
    dev_file = 'data/'+argv[1]+'/' + "dev"
    test_file = 'data/'+argv[1]+'/' + "test"
    # Preprocess data
    vocab, tagset, word_to_ix, tag_to_ix, ix_to_tag = get_vocab_and_tagset(train_file)
    tokens = load_data(train_file)
    contexts = get_contexts(tokens, word_to_ix, tag_to_ix)
    # Train and evaluate model
    model = train(device, contexts, vocab, tagset, dev_file, word_to_ix, tag_to_ix)
    # Predict the test file
    predict_test(model, test_file, word_to_ix, ix_to_tag, device)
    # Print time
    print(time.time() - st)


if __name__ == '__main__':
    main(sys.argv)