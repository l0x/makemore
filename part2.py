import torch
import torch.nn.functional as F

words = open("names.txt").read().splitlines()
letters = sorted(list(set(''.join(words))))
stoi = {s: i for i, s in enumerate(letters, start=1)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# create training set of bigrams(x, y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# Deterministic random init 27 neurons, each with 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

print(f"Num Elements: {num}")

for k in range(500):

    ################
    # Forward pass #
    ################

    xenc = F.one_hot(xs, num_classes=27).float()        # one hot encode inputs
    logits = xenc @ W                                   # Get activations, predict log counts

    # Make probabiltiy distribution, softmax
    counts = logits.exp()                               # exponentiate to get positive numbers
    probs = counts / counts.sum(1, keepdim=True)        # Normalise by row sum
    loss = -probs[torch.arange(num), ys].log().mean()   # calc negative log loss
    loss += 0.01*(W**2).mean()                          # regularisation
    print(loss.item())

    #################
    # Backward Pass #
    #################

    W.grad = None       # Rest gradients
    loss.backward()     # Set the gradients

    # Update the weights
    W.data += -50 * W.grad



