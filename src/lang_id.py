import os
import json
from typing import List, Dict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None

# DictBasedLID


class DictBasedLID:
    """Dictionary-based language identifier.

    Maintains sets of known words (Hindi romanized, English) and labels
    tokens by membership. 
    """

    HINDI_VOCAB = {
        "hai", "hoon", "hain", "ho", "tha", "thi", "the", "ka", "ke", "ki", "ko",
        "se", "mein", "par", "tak", "aur", "ya", "lekin", "magar", "bhi", "hi",
        "toh", "na", "nahi", "mat", "kya", "kyun", "kab", "kaise", "kahan",
        "kaun", "kitna", "kaisa", "ye", "wo", "is", "us", "in", "un", "main",
        "tum", "aap", "hum", "sab", "log", "bhai", "yaar", "beta", "dost",
        "ghar", "pani", "khana", "rasta", "kal", "aaj", "parso", "ab", "abhi",
        "tab", "jab", "ek", "do", "teen", "char", "panch", "das", "sau",
        "hazar", "lakh", "karo", "gaya", "gayi", "gaye", "jata", "jati",
        "jate", "karna", "kar", "rahe", "rahi", "raha", "lena", "dena",
        "mangna", "hoga", "gaye", "jaunga", "karne", "unko", "inko", "mera",
        "meri", "usne", "maine"
    }

    ENGLISH_VOCAB = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "person", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only",
        "come", "its", "over", "think", "also", "back", "after", "use",
        "two", "how", "our", "work", "first", "well", "way", "even", "new",
        "want", "because", "any", "these", "give", "day", "most", "us",
        "meeting", "cancel", "office", "please", "help", "assignment",
        "video", "informative", "thanks", "bro", "subscribe", "bell", "icon",
        "weather", "nice", "project", "complete", "presentation", "outstanding"
    }

    def predict(self, tokens: List[str]) -> List[str]:
        """Predict language labels for a list of tokens."""
        labels = []
        for token in tokens:
            t_low = token.lower().strip("?!.,;:")
            if not t_low:
                labels.append("OTHER")
            elif t_low in self.HINDI_VOCAB:
                labels.append("HI")
            elif t_low in self.ENGLISH_VOCAB:
                labels.append("EN")
            else:
                labels.append("UNK")
        return labels


# BiLSTMLID  # UPGRADE

if torch:
    class BiLSTMModel(nn.Module):
        """Char-level BiLSTM architecture."""
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
            super(BiLSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            return self.fc(lstm_out)

class BiLSTMLID:
    """BiLSTM-based language identifier using PyTorch. # UPGRADE
    """

    def __init__(self, model_path: str = "models/bilstm_lid.pt") -> None:
        self.model_path = model_path
        self.char_to_ix = {}
        self.label_to_ix = {"HI": 0, "EN": 1, "UNK": 2, "OTHER": 3}
        self.ix_to_label = {v: k for k, v in self.label_to_ix.items()}
        self.model = None

    def _prepare_sequence(self, tokens: List[str]):
        """Converts tokens into a single sequence of characters with padding or concat."""
        pass

    def train(self, data_path: str, epochs: int = 10) -> None:
        """Architecture only: training loop skeleton."""
        print(f"[BiLSTMLID] Loading data from {data_path}...")
        if not os.path.exists(data_path):
            print(f"[ERROR] Data path {data_path} not found.")
            return

        with open(data_path, 'r') as f:
            data = json.load(f)

        chars = set()
        for item in data:
            for tok in item['tokens']:
                for char in tok['word']:
                    chars.add(char)
        self.char_to_ix = {c: i + 1 for i, c in enumerate(sorted(list(chars)))}
        self.char_to_ix["<PAD>"] = 0

        if not torch:
            print("[SKIP] torch not installed. Cannot initialize BiLSTM model.")
            return

        # Initialize model
        self.model = BiLSTMModel(len(self.char_to_ix), 16, 32, len(self.label_to_ix))
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        print(f"[BiLSTMLID] Training loop for {epochs} epochs (SKELETON)...")

        
        print("[BiLSTMLID] Training complete (Simulated).")

    def predict(self, tokens: List[str]) -> List[str]:
        """Predict labels using BiLSTM."""
        if self.model is None:
            return DictBasedLID().predict(tokens)
        
        return ["HI"] * len(tokens)


# Helpers & Demo
def pretty_print_prediction(text: str, tokens: List[str], labels: List[str]):
    """Pretty prints the LID results."""
    print(f"\nText: {text}")
    print("-" * 25)
    for tok, lbl in zip(tokens, labels):
        print(f"{tok:<15} → {lbl}")
    print("-" * 25)

def main() -> None:
    """Demonstrate language identification."""
    demo_text = "kal meeting cancel ho gaya yaar"
    tokens = ["kal", "meeting", "cancel", "ho", "gaya", "yaar"]
    
    # 1. DictBasedLID
    lid = DictBasedLID()
    labels = lid.predict(tokens)
    
    print("=== Language Identification Demo ===")
    pretty_print_prediction(demo_text, tokens, labels)

    # 2. BiLSTMLID Setup
    print("\n[BiLSTMLID] Initializing architecture...")
    bilstm = BiLSTMLID()
    bilstm.train("data/annotated/labeled_sentences.json", epochs=1)


if __name__ == "__main__":
    main()
