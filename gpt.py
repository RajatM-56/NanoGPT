import torch
import torch.nn as nn
import torch.nn.functional as F

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, charsetSize):
        super().__init__()
        self.tokenEmbeddingTable = nn.Embedding(charsetSize, charsetSize)

    def forward(self, idx, targets):
        logits = self.tokenEmbeddingTable(idx)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

with open("input.txt", 'r', encoding="utf-8") as inputFile:
    input = inputFile.read()

# Extracting unique characters from the input text
uniqueCharacters = sorted(list(set(input)))
charsetSize = len(uniqueCharacters)

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

torch.manual_seed(1337)
blockSize = 8
batchSize = 4

def getBatch(split):
    data = trainingData if split == "train" else validationData
    ix = torch.randint(len(data) - blockSize, size=(batchSize,))
    x = torch.stack([data[i:i+blockSize] for i in ix])
    y = torch.stack([data[i + 1: i + blockSize + 1] for i in ix])

    return x, y

xb, yb = getBatch("train")
m = BigramLanguageModel(charsetSize)
logits, loss = m(xb, yb)

print(logits.shape)
