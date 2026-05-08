import torch


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
maxIters = 5000
evalInterval = 300
learningRate = 3e-4
evalIters = 200
torch.manual_seed(1337)
blockSize = 256
batchSize = 64
nEmbd = 384
numberOfHeads = 6
numberOfLayers = 6
dropout = 0.2

with open("input.txt", 'r', encoding="utf-8") as inputFile:
    input = inputFile.read()

# Extracting unique characters from the input text
uniqueCharacters = sorted(list(set(input)))
charsetSize = len(uniqueCharacters)