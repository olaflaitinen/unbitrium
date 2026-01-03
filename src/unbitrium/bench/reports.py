"""
Report Generation.
"""

from typing import Dict, Any, List

def generate_report(results: List[Dict[str, Any]], metrics: Dict[str, float], output_dir: str) -> None:
    """
    Generates a Markdown report from experiment results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write("# Experiment Report\n\n")
        f.write(f"## Metrics\n\n")
        for k, v in metrics.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Results\n\n")
        # Table of round history
        f.write("| Round | Accuracy | Loss |\n")
        f.write("| ----- | -------- | ---- |\n")
        # ... row generation ...
