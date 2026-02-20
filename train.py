import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
import numpy as np

# ========================
# Configuration
# ========================

CSV_FILE = "vibrio_proteins_dataset.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 32
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
SEQ_LEN = 100
EPOCHS = 30
LR = 0.003

# ========================
# Load Dataset
# ========================

df = pd.read_csv(CSV_FILE)
sequences = df["sequence"].tolist()

# Add special tokens
START_TOKEN = "<"
END_TOKEN = ">"
PAD_TOKEN = "_"

# Build vocabulary
all_chars = set("".join(sequences))
vocab = [PAD_TOKEN, START_TOKEN, END_TOKEN] + sorted(list(all_chars))

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(vocab)

print("Vocab size:", vocab_size)

# Encode sequences
encoded_sequences = []

for seq in sequences:
	seq = START_TOKEN + seq + END_TOKEN
	encoded = [stoi[ch] for ch in seq]
	encoded_sequences.append(encoded)

# ========================
# Create Training Data
# ========================

def get_batch():
	x_batch = []
	y_batch = []

	for _ in range(BATCH_SIZE):
		seq = random.choice(encoded_sequences)

		if len(seq) < SEQ_LEN + 1:
			continue

		start = random.randint(0, len(seq) - SEQ_LEN - 1)
		chunk = seq[start:start + SEQ_LEN + 1]

		x = chunk[:-1]
		y = chunk[1:]

		x_batch.append(x)
		y_batch.append(y)

	x_batch = torch.tensor(x_batch).to(DEVICE)
	y_batch = torch.tensor(y_batch).to(DEVICE)

	return x_batch, y_batch

# ========================
# LSTM Model
# ========================

class ProteinLSTM(nn.Module):
	def __init__(self):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, EMBED_DIM)
		self.lstm = nn.LSTM(
			EMBED_DIM,
			HIDDEN_DIM,
			NUM_LAYERS,
			batch_first=True
		)
		self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

	def forward(self, x, hidden=None):
		x = self.embed(x)
		out, hidden = self.lstm(x, hidden)
		out = self.fc(out)
		return out, hidden

model = ProteinLSTM().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ========================
# Training Loop
# ========================

for epoch in range(EPOCHS):

	model.train()
	total_loss = 0

	for step in range(200):

		x, y = get_batch()

		optimizer.zero_grad()

		output, _ = model(x)

		loss = criterion(
			output.view(-1, vocab_size),
			y.view(-1)
		)

		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	print(f"Epoch {epoch+1} | Loss: {total_loss/200:.4f}")

# ========================
# Generation Function
# ========================

def generate(max_length=500, temperature=1.0):

	model.eval()

	input_char = torch.tensor([[stoi[START_TOKEN]]]).to(DEVICE)
	hidden = None
	generated = ""

	for _ in range(max_length):

		output, hidden = model(input_char, hidden)

		logits = output[:, -1, :] / temperature
		probs = torch.softmax(logits, dim=-1)

		char_index = torch.multinomial(probs, 1).item()
		char = itos[char_index]

		if char == END_TOKEN:
			break

		generated += char
		input_char = torch.tensor([[char_index]]).to(DEVICE)

	return generated

# ========================
# Generate Samples
# ========================

print("\nGenerated Proteins:\n")

for _ in range(5):
	print(generate())
	print("-" * 60)
