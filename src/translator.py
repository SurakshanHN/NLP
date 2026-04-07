import time
from typing import Dict, List, Optional

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

# Internal imports
from src.tokenizer import RuleBasedTokenizer
from src.lang_id import DictBasedLID
from src.normalizer import HinglishNormalizer


# ---------------------------------------------------------------------------
# HinglishTranslator
# ---------------------------------------------------------------------------

class HinglishTranslator:
    """Translates Hinglish text into English.

    Offers both raw and preprocessed translation for quality comparison.
    """

    def __init__(
        self, 
        lang_id: Optional[DictBasedLID] = None, 
        normalizer: Optional[HinglishNormalizer] = None
    ) -> None:
        """
        Initialize the translator.
        
        Args:
            lang_id: Optional instance of a language identification module.
            normalizer: Optional instance of a spelling normalization module.
        """
        self.lang_id = lang_id or DictBasedLID()
        self.normalizer = normalizer or HinglishNormalizer()
        self.tokenizer = RuleBasedTokenizer()
        
        if GoogleTranslator:
            self._translator = GoogleTranslator(source='auto', target='en')
        else:
            self._translator = None

    def preprocess(self, text: str) -> str:
        """Tokenize, identify, and normalize Hinglish text.
        
        Args:
            text: Raw input string.
            
        Returns:
            Cleaned and normalized string.
        """
        # 1. Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # 2. Identify and Normalize
        # We only normalize tokens that aren't clearly English
        labels = self.lang_id.predict(tokens)
        
        final_tokens = []
        for tok, lbl in zip(tokens, labels):
            if lbl in ["HI", "UNK"]:
                # Normalize Hindi or unknown tokens
                norm_tok, _ = self.normalizer.phonetic_normalize(tok)
                final_tokens.append(norm_tok)
            else:
                final_tokens.append(tok)
                
        return " ".join(final_tokens)

    def translate_raw(self, text: str) -> str:
        """Direct API call with minimal intervention."""
        if not self._translator:
            return f"[MOCK] Translation of: {text}"
            
        try:
            return self._translator.translate(text)
        except Exception as e:
            return f"[ERROR] API Call Failed: {e}"

    def translate_clean(self, text: str) -> str:
        """Preprocess text first, then translate."""
        clean_text = self.preprocess(text)
        return self.translate_raw(clean_text)

    def compare(self, text: str) -> Dict[str, str]:
        """Provides a comparison between raw and preprocessed translation."""
        preprocessed = self.preprocess(text)
        
        # Simulate a slight delay to avoid rate limits if running multiple
        raw_trans = self.translate_raw(text)
        time.sleep(0.5)
        clean_trans = self.translate_raw(preprocessed)
        
        return {
            "input": text,
            "preprocessed_text": preprocessed,
            "raw_translation": raw_trans,
            "clean_translation": clean_trans
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate Hinglish translation quality improvements."""
    test_sentences = [
        "kal meeting cancel ho gaya yaar",
        "bhai ye movie bahut amazing thi",
        "mai ghar ja raha hu abhi"
    ]
    
    tr = HinglishTranslator()
    
    print("=== Hinglish to English Translator Comparison ===")
    print(f"{'Input Sentence':<35} | {'Raw Translation':<30} | {'Clean Translation'}")
    print("-" * 100)
    
    for sentence in test_sentences:
        res = tr.compare(sentence)
        print(f"{res['input']:<35} | {res['raw_translation']:<30} | {res['clean_translation']}")
        print(f"  (Preprocessed: {res['preprocessed_text']})\n")

if __name__ == "__main__":
    main()
