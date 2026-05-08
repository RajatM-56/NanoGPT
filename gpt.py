from hyperparameters import *
from BigramLanguageModel import BigramLanguageModel

# A lookup table to map a string to its corresponding index for encoding
stringToInteger = {character: integer for integer, character in enumerate(uniqueCharacters)}
integerToString = {integer: character for integer, character in enumerate(uniqueCharacters)}

# Simple encoding and decoding using the index of the letter in the character set
encode = lambda string: [stringToInteger[character] for character in string]
decode = lambda sequence: ''.join(integerToString[integer] for integer in sequence)

# Representing the input as tensors
inputTensor = torch.tensor(encode(input), dtype=torch.long)

# 9:1 training and validation split
n = int(0.9 * len(inputTensor))
trainingData = inputTensor[:n]
validationData = inputTensor[n:]

def getBatch(split):
    data = trainingData if split == "train" else validationData
    ix = torch.randint(len(data) - blockSize, size=(batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i + 1: i + blockSize + 1] for i in ix])

    return x, y

@torch.no_grad()
def estimateLoss():
    out={}
    model.eval()

    for split in ["train", "validation"]:
        losses = torch.zeros(evalIters)
        for k in range(evalIters):
            X, Y = getBatch(split)
            X = X.to(device)
            Y = Y.to(device)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = getBatch("train")
model = BigramLanguageModel()
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

for iter in range(maxIters):

    # every once in a while evaluate the loss on train and val sets
    if iter % evalInterval == 0:
        losses = estimateLoss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

    # sample a batch of data
    xb, yb = getBatch('train')
    xb, yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, maxNewTokens=500)[0].tolist()))