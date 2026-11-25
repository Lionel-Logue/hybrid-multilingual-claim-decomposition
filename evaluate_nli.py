# Requirements: pip install transformers torch numpy tqdm
import json
import math
import argparse
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_entailment_index(config):
    """Dynamically finds the 'entailment' label index for different models."""
    for label, idx in config.label2id.items():
        if "entailment" in label.lower() and "not" not in label.lower():
            return idx
    return 0 

def check_entailment(premise, hypothesis, tokenizer, model, device, entailment_idx, threshold=0.5):
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        entailment_prob = probs[entailment_idx].item()
        
    return entailment_prob > threshold

def build_clusters(n, entailment_matrix):
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and (entailment_matrix[i][j] or entailment_matrix[j][i]):
                adj[i].add(j)
                adj[j].add(i)
                
    visited = set()
    clusters = []
    for i in range(n):
        if i not in visited:
            q = [i]
            comp = set()
            while q:
                node = q.pop()
                if node not in visited:
                    visited.add(node)
                    comp.add(node)
                    q.extend(list(adj[node] - visited))
            clusters.append(comp)
    return clusters

def evaluate_sample(row, tokenizer, model, device, entailment_idx):
    if row.get("_parse_error", False) or not row.get("atomic_claims"):
        return None # Skip invalid generations

    original_claim = row["original_claim"]
    atomic_claims = [ac["original"] for ac in row["atomic_claims"]]
    n = len(atomic_claims)

    if n == 0:
        return None

    # 1. COMPLETENESS
    combined_atomic = " ".join(atomic_claims)
    completeness = check_entailment(
        premise=combined_atomic, 
        hypothesis=original_claim, 
        tokenizer=tokenizer, model=model, device=device, entailment_idx=entailment_idx
    )

    # 2. CORRECTNESS
    correctness_results = [
        check_entailment(
            premise=original_claim, hypothesis=ac, 
            tokenizer=tokenizer, model=model, device=device, entailment_idx=entailment_idx
        )
        for ac in atomic_claims
    ]
    correctness_score = sum(correctness_results) / n

    # 3. SEMANTIC ENTROPY
    entropy_score = 0.0
    if n > 1:
        entailment_matrix = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    entailment_matrix[i][j] = check_entailment(
                        premise=atomic_claims[i], hypothesis=atomic_claims[j],
                        tokenizer=tokenizer, model=model, device=device, entailment_idx=entailment_idx
                    )
                    
        clusters = build_clusters(n, entailment_matrix)
        for c in clusters:
            p_c = len(c) / n
            entropy_score -= p_c * math.log2(p_c)
            
    # Normalized entropy
    if n > 1:
        max_entropy = math.log2(n)
        normalized_entropy = entropy_score / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0

    return {
        "claim_id": row.get("claim_id", ""),
        "completeness": 1.0 if completeness else 0.0,
        "correctness": correctness_score,
        "decompscore": sum(correctness_results),
        "semantic_entropy": entropy_score,
        "normalized_semantic_entropy": round(normalized_entropy, 6),
        "num_atomic_claims": n
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Claim Decompositions via Multilingual NLI")
    parser.add_argument("--input_file", type=str, required=True, help="Path to generated JSONL file")
    parser.add_argument(
        "--model", 
        type=str, 
        default="mdeberta",
        help="Choose the NLI evaluator model"
    )
    args = parser.parse_args()

    # Model Selection Mapping
    model_repo = {
        "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    }[args.model]

    print(f"[*] Initializing {model_repo} on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo).to(device)
    model.eval() # Set model to evaluation mode

    entailment_idx = get_entailment_index(model.config)
    print(f"[*] Identified 'entailment' label index as: {entailment_idx}")

    # Load data
    if not os.path.exists(args.input_file):
        print(f"[!] Error: File '{args.input_file}' not found.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"[*] Evaluating {len(data)} samples from {args.input_file}...")
    
    results = []
    for row in tqdm(data, desc="Running NLI Encoder"):
        res = evaluate_sample(row, tokenizer, model, device, entailment_idx)
        if res is not None:
            results.append(res)

    # Save the individual per-claim scores to a JSONL file
    if results:
        output_file = args.input_file.replace(".jsonl", f"_evaluated_{args.model}.jsonl")
        if output_file == args.input_file:
            output_file += f"_evaluated_{args.model}.jsonl"
            
        with open(output_file, "w", encoding="utf-8") as out_f:
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n[*] Saved granular per-claim scores to: {output_file}")

    # Aggregate Metrics
    if not results:
        print("!!! No valid results to aggregate.")
        return

    avg_completeness = sum(r["completeness"] for r in results) / len(results)
    avg_correctness = sum(r["correctness"] for r in results) / len(results)
    avg_decompscore = sum(r["decompscore"] for r in results) / len(results)
    avg_entropy = sum(r["semantic_entropy"] for r in results) / len(results)
    avg_norm_entropy = sum(r["normalized_semantic_entropy"] for r in results) / len(results)
    avg_claims = sum(r["num_atomic_claims"] for r in results) / len(results)

    print("\n" + "="*50)
    print(f"FINAL EVALUATION METRICS ({args.model.upper()})")
    print("="*50)
    print(f"Total Valid Claims Assessed      : {len(results)}")
    print(f"Average Completeness             : {avg_completeness:.4f}")
    print(f"Average Correctness              : {avg_correctness:.4f}")
    print(f"Average DECOMPSCORE              : {avg_decompscore:.4f}")
    print(f"Average Semantic Entropy         : {avg_entropy:.4f}")
    print(f"Average Normalized Sem. Entropy  : {avg_norm_entropy:.4f}")
    print(f"Average Subclaims Generated      : {avg_claims:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()