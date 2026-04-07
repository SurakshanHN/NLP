# Annotation Guide — Hinglish Code-Switched Text

## Purpose
This guide describes how to annotate Hinglish (Hindi + English) text at the
**token level** for language identification.

---

## Label Set

| Label | Meaning | Examples |
|-------|---------|----------|
| `HI`  | Hindi (romanized) | kal, mujhe, bahut, karo, hai |
| `EN`  | English | meeting, cancel, amazing, please |
| `MIX` | Mixed/ambiguous (borrowed words used natively in both) | use sparingly |
| `OTHER` | Punctuation, emojis, numbers, named entities | ?, 😂, 42 |

## Rules

1. **Tokenize on whitespace first**, then split punctuation that is attached
   to words (e.g., `office?` → `office` + `?`).
2. **Label each token** with exactly one tag from the label set above.
3. **Borrowed words**: Words like "phone", "school", "bus" that are fully
   assimilated into Hindi should still be labeled `EN` for consistency.
   Add a `"note": "borrowed"` field if you want to flag them.
4. **Ambiguous tokens**: If a word exists in both Hindi and English with the
   same meaning (e.g., "guru"), label based on the **sentential context**.
5. **Named entities**: People, places, brands → `OTHER` unless clearly
   one language.
6. **Contractions**: `don't` stays as one token labeled `EN`.

## JSON Schema

```json
{
  "id": <int>,
  "text": "<original sentence>",
  "tokens": [
    {
      "word": "<token>",
      "lang": "HI" | "EN" | "MIX" | "OTHER"
    }
  ]
}
```

## Quality Checklist

- [ ] Every token in `text` is represented in `tokens`
- [ ] No token is left unlabeled
- [ ] Punctuation is split from words
- [ ] Spelling is preserved as-is (do NOT normalize during annotation)
- [ ] Inter-annotator agreement target: **κ ≥ 0.80**

## Sources

- YouTube comments on Hindi vlogs / tech channels
- Twitter/X Hinglish threads
- Synthetic sentences crafted to cover specific switch patterns:
  - Intra-sentential: `maine usko bola ki don't worry`
  - Inter-sentential: `Ye galat hai. This is wrong.`
  - Tag-switching: `achha hai na, right?`
