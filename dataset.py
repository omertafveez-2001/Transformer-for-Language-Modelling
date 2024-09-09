import re
from nltk.corpus import stopwords
import pathlib
import tiktoken
import nltk
import numpy as np
import torch
import requests


def get_data(BLOCK_SIZE: int, BATCH_SIZE: int, encoder):

    encoder = encoder

    PATH = pathlib.Path().resolve()
    nltk.download('stopwords')
    print("Fetching Data...")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)

    with open('input.txt', 'w') as f:
        f.write(response.text)
    
    with open('input.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    print("File downloaded successfully")
    scripts = ''.join(lines)

    # preprocessing text
    print("Processing Text...")
    # 1. Remove special characters
    scripts = re.sub(r'�', '', scripts)

    # 2. Remove numbers
    scripts = re.sub(r'\d+', '', scripts)

    # 3. Remove URL
    scripts = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', scripts)

    ids = encoder.encode(scripts)

    indices = np.random.randint(0, len(ids)-BLOCK_SIZE - 1, BATCH_SIZE)
    x = [ids[i:i + BLOCK_SIZE]for i in indices]
    y = [ids[i+1:i+BLOCK_SIZE+1]for i in indices]
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return x, y
