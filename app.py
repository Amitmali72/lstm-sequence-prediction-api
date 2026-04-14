from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI()

# Input schema
class InputText(BaseModel):
    text: str

# Load model
checkpoint = torch.load("shakespeare_lstm.pt", map_location="cpu")

word_to_idx = checkpoint['word_to_idx']
idx_to_word = checkpoint['idx_to_word']

class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 512, batch_first=True)
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel(len(word_to_idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(input: InputText):
    words = input.text.lower().split()[-5:]
    seq = [word_to_idx.get(w, 0) for w in words]
    seq = torch.tensor(seq).unsqueeze(0)

    with torch.no_grad():
        output = model(seq)
        pred = torch.argmax(output).item()

    return {"next_word": idx_to_word[pred]}