import torch

with open("input.txt", 'r', encoding="utf-8") as inputFile:
    input = inputFile.read()

# Extracting unique characters from the input text
uniqueCharacters = sorted(list(set(input)))

# A lookup table to map a string to its corresponding index for encoding
stringToInteger = {character: integer for integer, character in enumerate(uniqueCharacters)}
integerToString = {integer: character for integer, character in enumerate(uniqueCharacters)}

# Simple encoding and decoding using the index of the letter in the character set
encode = lambda string: [stringToInteger[character] for character in string]
decode = lambda sequence: ''.join(integerToString[integer] for integer in sequence)

# Representing the input as tensors
inputTensor = torch.tensor(encode(input), dtype=torch.int16)

# 9:1 training and validation split
n = 0.9 * len(inputTensor)
trainingData = inputTensor[:n]
validationData = inputTensor[n:]

blockSize = 8
