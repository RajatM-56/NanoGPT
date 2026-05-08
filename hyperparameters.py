import torch


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)

maxIters = 5000        # keep — gives good convergence
evalInterval = 500     # ↑ from 300 — less eval overhead, smoother run
learningRate = 3e-4    # keep
evalIters = 200        # keep
blockSize = 256        # keep
batchSize = 32         # ↓ from 64 — the only meaningful change
nEmbd = 384            # keep
numberOfHeads = 6      # keep
numberOfLayers = 6     # keep
dropout = 0.2          # keep

with open("input.txt", 'r', encoding="utf-8") as inputFile:
    input = inputFile.read()

# Extracting unique characters from the input text
uniqueCharacters = sorted(list(set(input)))
charsetSize = len(uniqueCharacters)