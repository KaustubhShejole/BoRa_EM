import re
import pandas as pd

def convert_text_to_csv(input_filename, output_filename="baselines.csv"):
    data = []
    
    # Regular expressions to capture mean and optional standard deviation (± std)
    acc_pattern = re.compile(r"Accuracy:\s*([0-9.]+)(?:\s*±\s*([0-9.]+))?")
    wacc_pattern = re.compile(r"Weighted Accuracy:\s*([0-9.]+)(?:\s*±\s*([0-9.]+))?")
    tau_pattern = re.compile(r"Kendall's Tau:\s*([0-9.]+)(?:\s*±\s*([0-9.]+))?")

    with open(input_filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or lines that don't contain the expected keywords
            if not line or "Accuracy:" not in line:
                continue
            
            # Split line into baseline name and metrics portion
            parts = line.split(":", 1)
            baseline_name = parts[0].strip()
            metrics_str = parts[1].strip()
            
            # Extract Accuracy
            acc_match = acc_pattern.search(metrics_str)
            acc_mean = float(acc_match.group(1)) if acc_match else None
            acc_std = float(acc_match.group(2)) if acc_match and acc_match.group(2) else 0.0
            
            # Extract Weighted Accuracy
            wacc_match = wacc_pattern.search(metrics_str)
            wacc_mean = float(wacc_match.group(1)) if wacc_match else None
            wacc_std = float(wacc_match.group(2)) if wacc_match and wacc_match.group(2) else 0.0
            
            # Extract Kendall's Tau
            tau_match = tau_pattern.search(metrics_str)
            tau_mean = float(tau_match.group(1)) if tau_match else None
            tau_std = float(tau_match.group(2)) if tau_match and tau_match.group(2) else 0.0
            
            data.append({
                "Method": baseline_name,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "wacc_mean": wacc_mean,
                "wacc_std": wacc_std,
                "tau_mean": tau_mean,
                "tau_std": tau_std
            })
            
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Set 'baseline' as index so the requested column headers line up exactly
    df.set_index("Method", inplace=True)
    
    # Save to CSV
    df.to_csv(output_filename)
    print(f"Successfully converted and saved data to {output_filename}")

# Example usage:
# Assuming your text file is named 'input.txt'
convert_text_to_csv("baselines.txt", "baselines_passage.csv")
