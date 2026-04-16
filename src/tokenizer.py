import os
import re
from typing import List



_EMOJI_RE = re.compile(
    r"["
    r"\U0001F600-\U0001F64F"   
    r"\U0001F300-\U0001F5FF"   
    r"\U0001F680-\U0001F6FF"   
    r"\U0001F900-\U0001F9FF"   
    r"\U0001FA70-\U0001FAFF"   
    r"\U00002702-\U000027B0"   
    r"]+",
    flags=re.UNICODE,
)

_CONTRACTIONS = {
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "it's", "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "you'd",
    "he's", "she's", "they're", "we're",
    "that's", "what's", "there's", "here's", "who's",
    "let's", "n't",
}

# Romanized Hindi particles to preserve
HINDI_PARTICLES = {"yaar", "hai", "na", "re", "bhi"}





class RuleBasedTokenizer:
    """Splits Hinglish text using handcrafted rules for emojis, contractions, and punctuation.
    
    Interface:
        .tokenize(text: str) -> List[str]
    """

    _PUNCT = r"""!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~"""

    def _split_hyphen(self, token: str) -> List[str]:
        """Splits 'meeting-wali' -> ['meeting', '-', 'wali']"""
        if "-" not in token:
            return [token]
        
        parts: List[str] = []
        for chunk in re.split(r"(\-)", token):
            if chunk:
                parts.append(chunk)
        return parts

    def _peel_punctuation(self, token: str) -> List[str]:
        """Separates leading/trailing punctuation while keeping contractions intact."""
        if token.lower() in _CONTRACTIONS:
            return [token]

        # Case for particles
        if token.lower() in HINDI_PARTICLES:
            return [token]

        # Split multiple punctuation marks
        match = re.match(rf"^([{re.escape(self._PUNCT)}]*)(.+?)([{re.escape(self._PUNCT)}]*)$", token)
        if not match:            
            return [c for c in token] if all(c in self._PUNCT for c in token) else [token]

        leading, body, trailing = match.groups()
        result = []
        if leading:
            result.extend(list(leading))
        
        if body:
            result.extend(self._split_hyphen(body))
            
        if trailing:
            result.extend(list(trailing))
        
        return result

    def tokenize(self, text: str) -> List[str]:
        """Shared interface for tokenizing text."""

        
        raw_tokens = text.split()
        final_tokens = []
        
        for part in raw_tokens:
            # emojis 
            sub_parts = []
            last_idx = 0
            for m in _EMOJI_RE.finditer(part):            
                text_before = part[last_idx:m.start()]
                if text_before: sub_parts.append(text_before)
                sub_parts.append(m.group())
                last_idx = m.end()
          
            text_after = part[last_idx:]
            if text_after: sub_parts.append(text_after)
            
            if not sub_parts: sub_parts = [part]
            
            for sub in sub_parts:
                if _EMOJI_RE.fullmatch(sub):
                    final_tokens.append(sub)
                else:
                    final_tokens.extend(self._peel_punctuation(sub))
                    
        return [t for t in final_tokens if t]


# SentencePieceTokenizer                                         
# UPGRADE: is used when corpus >= 5k lines


class SentencePieceTokenizer:
    """Subword tokenizer using the SentencePiece library."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._sp = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, corpus_file: str, vocab_size: int = 800) -> None:
        """Trains a SentencePiece model on the provided corpus file."""
        try:
            import sentencepiece as spm
        except ImportError:
            print("[ERROR] sentencepiece not installed. Skipping training.")
            return

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_prefix = os.path.join(model_dir, "sp_hinglish")
        
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
        self.model_path = f"{model_prefix}.model"
        self.load(self.model_path)

    def load(self, model_path: str) -> None:
        """Loads a pre-trained SentencePiece model."""
        try:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(model_path)
            self.model_path = model_path
        except Exception as e:
            print(f"[ERROR] Could not load SentencePiece model: {e}")

    def tokenize(self, text: str) -> List[str]:
        """Shared interface: tokenizes text into subword pieces."""
        if not self._sp:
            return text.split()  # Fallback to whitespace for demo
        return self._sp.encode(text, out_type=str)



# Helpers & Demo

def compare_tokenizers(text: str) -> None:
    """Prints a side-by-side comparison of RuleBased vs SentencePiece."""
    rb = RuleBasedTokenizer()
    sp = SentencePieceTokenizer(model_path="models/sp_hinglish.model")
    
    rb_res = rb.tokenize(text)
    sp_res = sp.tokenize(text)
    
    print("-" * 50)
    print(f"Input: {text}")
    print(f"RuleBased   : {rb_res}")
    print(f"SentencePiece: {sp_res}")
    print("-" * 50)


def main():
    """Demo for Step 2."""
    demo_input = "kal meeting cancel ho gaya yaar 😂, kya scene hai bhai?"
    
    # 1. RuleBasedTokenizer demo
    rb = RuleBasedTokenizer()
    print("RuleBased Result:")
    print(rb.tokenize(demo_input))
    
    # 2. SentencePieceTokenizer demo
    sp = SentencePieceTokenizer()
    corpus_file = "data/raw/youtube_comments.txt"
    if os.path.exists(corpus_file):
        print("\nTraining SentencePiece (vocab_size=100 for small demo corpus)...")
        sp.train(corpus_file, vocab_size=100)
    
    print("\nSentencePiece Result:")
    print(sp.tokenize(demo_input))
    
    # 3. Side-by-side comparison
    print("\nSide-by-Side Comparison:")
    compare_tokenizers(demo_input)
    
    # Mixed token check
    mixed = "meeting-wali"
    print(f"\nMixed token '{mixed}' -> {rb.tokenize(mixed)}")

if __name__ == "__main__":
    main()
