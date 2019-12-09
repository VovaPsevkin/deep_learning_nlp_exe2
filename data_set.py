import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from cleopatra import LanguageModeler
from data_loader import CustomDataset


dataset = CustomDataset(
    data_root=r"/home/vova/PycharmProjects/Deep_Exe2/Ass2DL/data/ner/train")
dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)


obj=CustomDataset(data_root=r"/home/vova/PycharmProjects/Deep_Exe2/Ass2DL/data/ner/train")
vocab = obj.vocab

CONTEXT_SIZE = 5
EMBEDDING_DIM = 10
losses = []
loss_function = nn.CrossEntropyLoss()
model = LanguageModeler(len(vocab), EMBEDDING_DIM, hidden_layer=64, num_of_labels=5, context_size = CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)



for epoch in range(30):
    total_loss = 0
    for i, data in enumerate(dataloader):
        context, target = data

#         # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
#         # into integer indices and wrap them in tensors)
#         context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
#         batch_tags = torch.LongTensor([target])
        loss = loss_function(log_probs, target)
        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!

