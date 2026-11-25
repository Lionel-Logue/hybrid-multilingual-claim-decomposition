# Requirements: openai, tqdm

import json
import re
import os
import sys
import time
import signal
import asyncio
import argparse
import urllib.request
import subprocess
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def is_server_ready(base_url: str) -> bool:
    #Check if the llama server is responsive.
    health_url = f"{base_url}/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False


def start_llama_server(args) -> subprocess.Popen:
    #Launch the llama-server subprocess if not running.
    base_url = f"http://127.0.0.1:{args.port}"
    if is_server_ready(base_url):
        print(f"[*] llama server already detected at {base_url}. Reusing existing instance.")
        return None

    cmd = [
        "llama", "serve",
        "-hf", args.model_hf,
        "-ngl", str(args.ngl),
        "-c", str(args.ctx_size),
        "-np", str(args.parallel),
        "-cb",
        "-cram", "0",
        "--port", str(args.port)
    ]

    if not args.enable_reasoning:
        cmd.extend(["-rea", "off"])

    print(f"[*] Launching server command:\n    {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        cmd[0] = "llama-server"
        cmd.pop(1)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("[*] Waiting for llama server to download model and initialize slots...")
    for _ in range(1200):
        if is_server_ready(base_url):
            print(f"[+] Server is ready on port {args.port}!")
            return proc
        time.sleep(1)

    proc.terminate()
    raise RuntimeError("Failed to start llama server within the timeout period.")


_server_process = None

def cleanup_server():
    #Ensure server process is terminated.
    global _server_process
    if _server_process is not None:
        print("\n[*] Stopping llama-server subprocess...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None

async def async_chat(client: AsyncOpenAI, messages: list, temperature: float = 0.1,
                     max_tokens: int = 2048, semaphore: asyncio.Semaphore = None) -> str:
    async with (semaphore if semaphore is not None else asyncio.Lock()):
        response = await client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

JUDGE_SYSTEM_PROMPT = """\
You are an expert linguist and fact-checking evaluator. Your task is to evaluate the quality \
of a claim decomposition - i.e., how well a compound claim was broken down into atomic \
(independently verifiable) sub-claims.

You will receive:
- The **original claim** (in its original language)
- Its **English translation**
- The **list of generated atomic claims** (each with original language text and English translation)

Evaluate the decomposition on these four dimensions using the rubric below. \
For each dimension, provide a brief justification and then a score from 1 to 5.

### RUBRIC

**1. Atomicity** - Is each sub-claim truly atomic (one verifiable fact per claim)?
  - 5: Every sub-claim contains exactly one independently verifiable fact.
  - 4: Almost all sub-claims are atomic; one might combine two very closely related facts.
  - 3: Some sub-claims bundle two or more distinct facts that should be separated.
  - 2: Many sub-claims are compound (contain multiple verifiable facts).
  - 1: No meaningful decomposition was performed; sub-claims are as complex as the original.

**2. Self-Containedness** - Can each sub-claim be understood and verified in isolation?
  - 5: Every sub-claim can be understood and verified without seeing the original or other sub-claims. \
All pronouns, anaphora, and implicit references are fully resolved.
  - 4: Almost all sub-claims are self-contained; one has a minor unresolved reference.
  - 3: Some sub-claims require context from the original or from other sub-claims to be understood.
  - 2: Many sub-claims are ambiguous or rely on implicit context.
  - 1: Sub-claims are mostly fragments that cannot stand alone.

**3. Minimal Redundancy** - Are the sub-claims non-overlapping and non-repetitive?
  - 5: Every sub-claim conveys a distinct, unique fact; no information is repeated across claims.
  - 4: Negligible overlap; one pair of claims shares a minor detail.
  - 3: Some noticeable overlap between claims; a fact or qualifier appears in multiple sub-claims.
  - 2: Significant redundancy; multiple claims essentially say the same thing in different words.
  - 1: Extreme redundancy; most sub-claims are paraphrases of each other.

**4. Linguistic Quality** - Are the original-language sub-claims grammatically correct and natural?
  - 5: All original-language sub-claims are grammatically correct, fluent, and read naturally.
  - 4: Almost all are well-formed; one has a minor grammatical or phrasing issue.
  - 3: Some sub-claims have noticeable grammatical errors or unnatural phrasing.
  - 2: Many sub-claims are poorly formed, with broken grammar or awkward constructions.
  - 1: Most sub-claims are garbled or unintelligible in the original language.

### OUTPUT FORMAT
Respond ONLY with a valid JSON object (no markdown fences, no commentary outside the JSON):
{
  "atomicity_justification": "<brief reasoning>",
  "atomicity": <1-5>,
  "self_containedness_justification": "<brief reasoning>",
  "self_containedness": <1-5>,
  "redundancy_justification": "<brief reasoning>",
  "minimal_redundancy": <1-5>,
  "linguistic_quality_justification": "<brief reasoning>",
  "linguistic_quality": <1-5>
}"""


def format_judge_user_message(row: dict) -> str:
    original_claim = row.get("original_claim", "")
    translation = row.get("original_claim_translation", "")
    language = row.get("language", "")
    atomic_claims = row.get("atomic_claims", [])

    claims_text = ""
    for i, ac in enumerate(atomic_claims, 1):
        orig = ac.get("original", "")
        eng = ac.get("english", "")
        claims_text += f"  {i}. Original: {orig}\n     English: {eng}\n"

    return (
        f"**Language:** {language}\n"
        f"**Original Claim:** {original_claim}\n"
        f"**English Translation:** {translation}\n\n"
        f"**Generated Atomic Claims ({len(atomic_claims)} total):**\n{claims_text}"
    )


def parse_judge_response(raw: str) -> dict:
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return empty scores
    return {
        "atomicity": None, "self_containedness": None,
        "minimal_redundancy": None, "linguistic_quality": None,
        "_parse_error": True,
    }


class IncrementalSaver:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lock = asyncio.Lock()
        with open(self.filepath, "w", encoding="utf-8") as f:
            pass

    async def append(self, result: dict):
        async with self.lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


async def evaluate_with_llm_judge(data: list, client: AsyncOpenAI, semaphore: asyncio.Semaphore,
                                   out_path: str):
    #Run LLM-as-a-judge evaluation on all decomposition results.
    saver = IncrementalSaver(out_path)

    async def process_item(row):
        if row.get("_parse_error", False) or not row.get("atomic_claims"):
            return None

        user_msg = format_judge_user_message(row)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        raw = await async_chat(
            client, messages,
            temperature=0.1,
            max_tokens=1024,
            semaphore=semaphore,
        )

        scores = parse_judge_response(raw)
        result = {
            "claim_id": row.get("claim_id", ""),
            "language": row.get("language", ""),
            "num_atomic_claims": len(row.get("atomic_claims", [])),
            "atomicity": scores.get("atomicity"),
            "self_containedness": scores.get("self_containedness"),
            "minimal_redundancy": scores.get("minimal_redundancy"),
            "linguistic_quality": scores.get("linguistic_quality"),
            "atomicity_justification": scores.get("atomicity_justification", ""),
            "self_containedness_justification": scores.get("self_containedness_justification", ""),
            "redundancy_justification": scores.get("redundancy_justification", ""),
            "linguistic_quality_justification": scores.get("linguistic_quality_justification", ""),
        }
        if scores.get("_parse_error"):
            result["_parse_error"] = True

        await saver.append(result)
        return result

    tasks = [process_item(row) for row in data]

    results = []
    for future in tqdm_asyncio.as_completed(tasks, desc="LLM Judge Evaluation"):
        res = await future
        if res is not None:
            results.append(res)

    return results


def print_aggregate_results(results: list, label: str):
    valid = [r for r in results if not r.get("_parse_error")]
    if not valid:
        print(f"[!] No valid LLM judge results for {label}.")
        return

    n = len(valid)
    dims = ["atomicity", "self_containedness", "minimal_redundancy", "linguistic_quality"]
    avgs = {}
    for dim in dims:
        scores = [r[dim] for r in valid if r.get(dim) is not None]
        avgs[dim] = sum(scores) / len(scores) if scores else 0.0

    avg_claims = sum(r["num_atomic_claims"] for r in valid) / n

    print(f"\n{'='*55}")
    print(f"LLM-AS-JUDGE RESULTS: {label}")
    print(f"{'='*55}")
    print(f"Total Valid Samples          : {n}")
    print(f"Avg Atomicity        (1-5)   : {avgs['atomicity']:.3f}")
    print(f"Avg Self-Containedness (1-5) : {avgs['self_containedness']:.3f}")
    print(f"Avg Minimal Redundancy (1-5) : {avgs['minimal_redundancy']:.3f}")
    print(f"Avg Linguistic Quality (1-5) : {avgs['linguistic_quality']:.3f}")
    print(f"Avg Subclaims Generated      : {avg_claims:.2f}")
    print(f"{'='*55}")

async def main():
    global _server_process
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation for Claim Decomposition")

    # Input/Output
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to a generated JSONL results file from run_decomposition.py")

    # llama.cpp server arguments (mirror main_v3.py)
    parser.add_argument("--model-hf", type=str, default="unsloth/Qwen3.5-9B-GGUF:Q8_0",
                        help="Hugging Face repo or local path for GGUF model")
    parser.add_argument("-c", "--ctx-size", type=int, default=16384,
                        help="Context window size (default: 16384)")
    parser.add_argument("-np", "--parallel", type=int, default=3,
                        help="Number of slots / parallel requests (default: 3)")
    parser.add_argument("--ngl", type=int, default=99,
                        help="Number of GPU layers to offload")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the llama-server (default: 8000)")
    parser.add_argument("--enable-reasoning", action="store_true",
                        help="Enable reasoning mode for the model")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    try:
        _server_process = start_llama_server(args)

        client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{args.port}/v1",
            api_key="no-key-required",
        )
        semaphore = asyncio.Semaphore(args.parallel)

        # Load input data
        if not os.path.exists(args.input_file):
            print(f"[!] Error: File '{args.input_file}' not found.")
            return

        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        print(f"[*] Loaded {len(data)} samples from {args.input_file}")

        # Build output path
        base_name = os.path.basename(args.input_file)
        out_name = base_name.replace(".jsonl", "_llm_judge.jsonl")
        if out_name == base_name:
            out_name = base_name + "_llm_judge.jsonl"
        out_path = os.path.join(os.path.dirname(args.input_file), out_name)

        print(f"[*] Output: {out_path}")
        print(f"[*] Starting LLM-as-a-Judge evaluation...\n")

        start = time.time()
        results = await evaluate_with_llm_judge(data, client, semaphore, out_path)
        elapsed = time.time() - start

        label = base_name.replace("_results.jsonl", "").replace("_", " ").title()
        print_aggregate_results(results, label)
        print(f"\n[+] Evaluation finished in {elapsed:.2f}s.")
        print(f"[+] Per-claim scores saved to: {out_path}")

    finally:
        cleanup_server()


if __name__ == "__main__":
    asyncio.run(main())
