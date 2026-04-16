import os
from typing import Dict, List, Tuple

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

# HinglishNormalizer
class HinglishNormalizer:
    """Normalizes romanized Hindi spelling variants to canonical forms.
    
    Includes a hand-curated map and a phonetic fallback for spelling variations.
    """

    
    VARIANT_MAP: Dict[str, str] = {
        # Pronouns
        "mai": "main", "main": "main", "mein": "main", "mi": "main",
        "tu": "aap", "tum": "aap", "aap": "aap", "tuh": "aap",
        "tera": "tera", "teri": "tera", "tere": "tera", "tor": "tera",
        "mera": "mera", "meri": "mera", "mere": "mera",
        "wo": "woh", "woh": "woh", "voh": "woh", "wa": "woh",
        "hum": "hum", "hame": "hum", "humein": "hum",
        "unke": "unka", "unki": "unka", "unka": "unka",
        "kiska": "kiska", "kiski": "kiska", "kiske": "kiska",

        # Verbs & Endings
        "hai": "hai", "hain": "hai", "he": "hai", "hay": "hai",
        "tha": "tha", "thi": "tha", "the": "tha",
        "karo": "karo", "kro": "karo", "kar": "karo", "karna": "karo",
        "hoon": "hoon", "hun": "hoon", "hu": "hoon",
        "hoga": "hoga", "hoge": "hoga", "hogi": "hoga",
        "jata": "jata", "jati": "jata", "jate": "jata", "ja": "jata", "jaa": "jata",
        "raha": "raha", "rahe": "raha", "rahi": "raha",

        # Negation
        "nahi": "nahi", "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "na": "nahi",

        # Common Misspellings/Particles
        "bahut": "bahut", "bohot": "bahut", "bahot": "bahut", "bhot": "bahut",
        "accha": "accha", "acha": "accha", "achha": "accha",
        "yaar": "yaar", "yr": "yaar", "yaara": "yaar",
        "kyun": "kyun", "kyu": "kyun", "kio": "kyun",
        "kyon": "kyun",
        "samajh": "samajh", "samaj": "samajh", "smjh": "samajh",
        "phir": "phir", "fir": "phir",
    }

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def phonetic_normalize(self, token: str) -> Tuple[str, bool]:
        """Finds nearest canonical form using rapidfuzz similarity.
        
        Args:
            token: Input token to normalize.
            
        Returns:
            Tuple of (normalized_token, was_changed).
        """
        token_low = token.lower().strip("?!.,;:")
        if not token_low:
            return token, False

        
        if token_low in self.VARIANT_MAP:
            canonical = self.VARIANT_MAP[token_low]
            return canonical, canonical != token_low

        if not fuzz:
            return token, False

        best_match = None
        best_score = 0
        
        
        for variant in self.VARIANT_MAP.keys():
            score = fuzz.ratio(token_low, variant)
            if score > best_score:
                best_score = score
                best_match = variant
        
        
        if best_match and best_score >= (self.threshold * 100):
            canonical = self.VARIANT_MAP[best_match]
            return canonical, canonical != token_low

        return token, False

    def normalize_sequence(self, tokens: List[str]) -> List[str]:
        """Normalizes a sequence of tokens and prints a diff of changes."""
        normalized_tokens = []
        changes = []

        for token in tokens:
            norm, changed = self.phonetic_normalize(token)
            normalized_tokens.append(norm)
            if changed:
                changes.append(f"  {token:<10} -> {norm}")

        if changes:
            print("\n[Normalization Diff]")
            for change in changes:
                print(change)
        
        return normalized_tokens

    def normalize(self, tokens: List[str]) -> List[str]:
        """Alias for normalize_sequence without the extra print (for API)."""
        return [self.phonetic_normalize(t)[0] for t in tokens]



# Demo

def main() -> None:
    """Demonstrate Hinglish normalization."""
    demo_tokens = ["mai", "ghar", "ja", "raha", "hu", "yaar"]
    
    print("=== HinglishNormalizer Demo ===")
    print(f"Original: {demo_tokens}")
    
    norm = HinglishNormalizer(threshold=0.85)
    normalized = norm.normalize_sequence(demo_tokens)
    
    print(f"\nFinal Normalized: {normalized}")

    print("\n--- Edge Case Test ---")
    edge_cases = ["nhi", "bohot", "acha", "hain", "kro"]
    print(f"Original: {edge_cases}")
    edge_normalized = norm.normalize_sequence(edge_cases)
    print(f"Final Normalized: {edge_normalized}")


if __name__ == "__main__":
    main()
