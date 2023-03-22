import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import time

# Read in data
words = open("names.txt").read().splitlines()
print("Number of words", len(words))
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Hyper Parameters (hand tuned)
block_size = 4      # Context length, how many characters do we consider?
dims = 12           # Dimensionality of embedding space
n_hidden = 396      # Size of hidden layer
iters = 500000      # How many steps of training
batch_size = 32     # Mini batch size

def build_dataset(words):
    """ Build Dataset (Inputs and Outputs) from a list of words. """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]   # Crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y


# Construct train, dev, test split
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtst, Ytst = build_dataset(words[n2:])

# nn optimised parameters
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, dims), generator=g)                    # Embedding
W1 = torch.randn((block_size*dims, n_hidden), generator=g)  # Weights of hidden layer
b1 = torch.randn(n_hidden, generator=g)                     # Bias of hidden layer
W2 = torch.randn((n_hidden, 27), generator=g)               # Weights of output layer
b2 = torch.randn(27, generator=g)                           # Bias of output layer
parameters = [C, W1, b1, W2, b2]

print("Number of params:", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

lri = []
lossi = []
stepi = []

start = time.time()
for i in range(iters):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    # Forward pass
    emb = C[Xtr[ix]]  # Embedding, for all the contexts into the C space
    h = torch.tanh(emb.view(-1, block_size*dims) @ W1 + b1)     # Activations for hidden layer
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])   # Softmax, NLL
    loss += 0.01 * (((W1 ** 2).mean() + (W2 ** 2).mean()) / 2)

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    if i < 100000:
        lr = 0.1
    elif i < 200000:
        lr = 0.07
    elif i < 300000:
        lr = 0.04
    else:
        lr = 0.01

    # update
    for p in parameters:
        p.data += -lr * p.grad

    # gather stats
    lri.append(lr)
    lossi.append(loss.log10().item())
    stepi.append(i)


print(f"Training complete: time taken {time.time() - start}s")
emb = C[Xtr]
h = torch.tanh(emb.view(-1, block_size * dims) @ W1 + b1)  # Activations for hidden layer
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)  # Softmax, NLL
print("Final Train loss:", loss.item())

emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * dims) @ W1 + b1)  # Activations for hidden layer
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)  # Softmax, NLL
print("Final Dev loss:", loss.item())

plt.plot(stepi, lossi)
plt.show()

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    print(''.join(itos[i] for i in out))

print(','.join(str(x) for x in [
    block_size,
    dims,
    n_hidden,
    iters,
    batch_size,
]))

# emb = C[Xtst]
# h = torch.tanh(emb.view(-1, block_size * dims) @ W1 + b1)  # Activations for hidden layer
# logits = h @ W2 + b2
# loss = F.cross_entropy(logits, Ytst)  # Softmax, NLL
# print("Final Test loss:", loss.item())
