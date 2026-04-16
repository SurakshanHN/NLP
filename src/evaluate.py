from typing import Dict, List
import sacrebleu
from sklearn.metrics import f1_score, accuracy_score, classification_report

def token_f1(pred: List[str], gold: List[str]) -> Dict[str, float]:
    """Compute per-class and macro F1 for language identification.
    
    Args:
        pred: Predicted language labels.
        gold: Gold-standard language labels.
        
    Returns:
        Dict mapping each class and 'macro' to its F1 score.
    """
    labels = sorted(list(set(gold + pred)))
    
    # Classification report for the terminal
    print("\n[LID Classification Report]")
    print(classification_report(gold, pred, labels=labels, zero_division=0))
    
    # Calculate F1 scores
    f1_per_class = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(gold, pred, average='macro', zero_division=0)
    accuracy = accuracy_score(gold, pred)
    
    metrics = {labels[i]: f1_per_class[i] for i in range(len(labels))}
    metrics['macro'] = macro_f1
    metrics['accuracy'] = accuracy
    
    return metrics

def bleu_score(hypotheses: List[str], references: List[str]) -> float:
    """Compute corpus-level BLEU score.
    
    Args:
        hypotheses: List of system translations.
        references: List of reference translations.
        
    Returns:
        BLEU score as a float.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return float(bleu.score)

def code_switch_index(token_labels: List[str]) -> float:
    """Compute the Code-Switch Index (CSI).
    
    CSI = (number of switches) / (total tokens - 1).
    A switch occurs when adjacent labels differ.
    """
    if len(token_labels) <= 1:
        return 0.0
    
    switches = 0
    for i in range(len(token_labels) - 1):
        if token_labels[i] != token_labels[i+1]:
            switches += 1
            
    return switches / (len(token_labels) - 1)

def evaluation_report(
    pred_labels: List[str], 
    gold_labels: List[str], 
    hypotheses: List[str], 
    references: List[str]
):
    """Prints a formatted summary evaluation report."""
    print("\n" + "="*50)
    print("           HINGLISH PIPELINE EVALUATION")
    print("="*50)
    
    # 1. LID Metrics
    lid_metrics = token_f1(pred_labels, gold_labels)
    
    # 2. Translation Metrics
    bleu = bleu_score(hypotheses, references)
    
    # 3. Complexity Metrics
    csi = code_switch_index(gold_labels)
    
    print("\n[Summary Metrics]")
    print(f"LID Macro F1  : {lid_metrics['macro']:.4f}")
    print(f"LID Accuracy  : {lid_metrics['accuracy']:.4f}")
    print(f"Translation BLEU: {bleu:.4f}")
    print(f"Code-Switch Index: {csi:.4f}")
    print("="*50)

def main() -> None:
    """Demonstrate evaluation metrics with synthetic data."""
    # LID Data
    gold_lid = ["HI", "EN", "HI", "HI", "EN", "UNK", "HI"]
    pred_lid = ["HI", "EN", "EN", "HI", "EN", "HI", "HI"]
    
    # Translation Data
    references = [
        "the meeting was cancelled",
        "i am going home now",
        "this movie was very amazing"
    ]
    hypotheses = [
        "the meeting got cancelled",
        "i am going to my house",
        "this movie is really amazing"
    ]
    
    # Run full report
    evaluation_report(pred_lid, gold_lid, hypotheses, references)

if __name__ == "__main__":
    main()
