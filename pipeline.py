import argparse
import json
import sys
from typing import Any, Dict, List


try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    CONSOLE = Console()
except ImportError:
    CONSOLE = None

# Internal imports
from src.tokenizer import RuleBasedTokenizer
from src.lang_id import DictBasedLID
from src.normalizer import HinglishNormalizer
from src.translator import HinglishTranslator


# Pipeline Engine

class HinglishPipeline:
    """Orchestrates the Hinglish NLP components into a unified workflow."""
    
    def __init__(self):
        self.tokenizer = RuleBasedTokenizer()
        self.lid = DictBasedLID()
        self.normalizer = HinglishNormalizer()
        self.translator = HinglishTranslator(lang_id=self.lid, normalizer=self.normalizer)

    def run(self, text: str, mode: str, compare: bool = False) -> Dict[str, Any]:
        """Run the pipeline in a specific mode."""
        results = {"input": text, "mode": mode}
        
        # 1. Tokenize (Base for all modes)
        tokens = self.tokenizer.tokenize(text)
        results["tokens"] = tokens
        if mode == "tokenize":
            return results
            
        # 2. Language ID
        labels = self.lid.predict(tokens)
        results["labels"] = labels
        if mode == "langid":
            return results
            
        # 3. Normalize
        normalized_tokens = [
            self.normalizer.phonetic_normalize(t)[0] if l in ["HI", "UNK"] else t 
            for t, l in zip(tokens, labels)
        ]
        results["normalized"] = normalized_tokens
        results["clean_text"] = " ".join(normalized_tokens)
        if mode == "normalize":
            return results
            
        # 4. Translate
        if mode in ["translate", "full"]:
            if compare:
                comp_res = self.translator.compare(text)
                results["translation_raw"] = comp_res["raw_translation"]
                results["translation_clean"] = comp_res["clean_translation"]
            else:
                results["translation"] = self.translator.translate_clean(text)
                
        return results


# Output Formatting

def print_pretty(result: Dict[str, Any], mode: str):
    """Prints results using the rich library for a premium CLI experience."""
    if not CONSOLE:
        # Fallback to plain text if rich is missing
        print(f"\n--- {mode.upper()} RESULT ---")
        print(json.dumps(result, indent=2))
        return

    # Create a nice layout
    CONSOLE.print(Panel(f"[bold blue]Hinglish Pipeline[/] - [italic]{mode}[/]", box=box.DOUBLE))
    
    # 1. Input Section
    CONSOLE.print(f"[bold cyan]Input:[/] {result['input']}")

    # 2. Tokenization & LID Results (Table)
    if "tokens" in result:
        table = Table(title="Token Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Token", style="dim")
        if "labels" in result:
            table.add_column("Lang ID", justify="center")
        if "normalized" in result:
            table.add_column("Normalized", style="green")
            
        for i in range(len(result["tokens"])):
            row = [result["tokens"][i]]
            if "labels" in result:
                label = result["labels"][i]
                color = "yellow" if label == "HI" else "blue" if label == "EN" else "red"
                row.append(f"[{color}]{label}[/]")
            if "normalized" in result:
                row.append(result["normalized"][i])
            table.add_row(*row)
        
        CONSOLE.print(table)

    # 3. Translation Section
    if "translation" in result:
        CONSOLE.print(Panel(f"[bold green]Translation:[/] {result['translation']}", border_style="green"))
    
    if "translation_raw" in result:
        CONSOLE.print("\n[bold]Translation Comparison:[/]")
        CONSOLE.print(f"[red]Raw  :[/] {result['translation_raw']}")
        CONSOLE.print(f"[green]Clean:[/] {result['translation_clean']}")


def main():
    parser = argparse.ArgumentParser(description="Hinglish NLP CLI Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input Hinglish text")
    parser.add_argument("--mode", "-m", choices=["tokenize", "langid", "normalize", "translate", "full"], 
                        default="full", help="Processing mode")
    parser.add_argument("--output", "-o", choices=["pretty", "json"], default="pretty", help="Output format")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare raw and clean translation")
    
    args = parser.parse_args()
    
    # Run
    pipeline = HinglishPipeline()
    result = pipeline.run(args.input, args.mode, compare=args.compare)
    
    # Output
    if args.output == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_pretty(result, args.mode)


if __name__ == "__main__":
    main()
