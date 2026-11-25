# Requirements: openai, sentence-transformers, numpy, tqdm

import json
import re
import csv
import os
import sys
import time
import signal
import asyncio
import argparse
import urllib.request
import subprocess
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from drnd_examples import STATIC_EXAMPLES, DYNAMIC_POOL

# ── Dataset Configuration ──
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_embedder = None
_pool_embeddings = None
_server_process = None


def load_dataset_csv(split="decomposable"):
    path = os.path.join(DATASET_DIR, f"MultiClaim_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def is_server_ready(base_url: str) -> bool:
    #Check if the llama server is responsive.
    health_url = f"{base_url}/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        # Fallback check on models endpoint
        try:
            req = urllib.request.Request(f"{base_url}/v1/models", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False


def start_llama_server(args) -> subprocess.Popen:
    #Launch the llama-server / llama serve subprocess if not running.
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
    
    # Try running 'llama serve', fallback to 'llama-server' binary if needed
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        cmd[0] = "llama-server"
        cmd.pop(1)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("[*] Waiting for llama server to download model and initialize slots (can take up to 10 mins)...")
    for _ in range(1200):
        if is_server_ready(base_url):
            print(f"[+] Server is ready and accepting requests on port {args.port}!")
            return proc
        time.sleep(1)

    proc.terminate()
    raise RuntimeError("Failed to start llama server within the timeout period.")


def cleanup_server():
    global _server_process
    if _server_process is not None:
        print("\n[*] Stopping llama-server subprocess...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None

def _strip_thinking(text: str) -> str:
    #Remove thinking tokens / tags from model output.
    if not text:
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'^Thinking Process:.*?(?=\n\{|\n-|\nFact|\nTranslation:|\nTerminate)', '',
                  text, flags=re.DOTALL).strip()
    if text and text[0] != '{' and '{' in text:
        idx = text.index('{')
        text = text[idx:]
    return text


def _parse_json(raw_text: str, claim_id: str = "", language: str = "", claim: str = "", claim_translation: str = ""):
    #Extract JSON object and forcefully apply the schema including ClaimID.
    result = {
        "claim_id": claim_id,
        "language": language,
        "original_claim": claim,
        "original_claim_translation": claim_translation,
        "atomic_claims": []
    }
    
    if not raw_text.strip():
        result["atomic_claims"] = [{"original": claim, "english": claim_translation}] if claim else []
        result["_parse_error"] = True
        return result
    
    try:
        parsed = json.loads(raw_text)
        result["atomic_claims"] = parsed.get("atomic_claims", [])
        if "thought_process" in parsed:
            result["thought_process"] = parsed["thought_process"]
        return result
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            result["atomic_claims"] = parsed.get("atomic_claims", [])
            if "thought_process" in parsed:
                result["thought_process"] = parsed["thought_process"]
            return result
        except json.JSONDecodeError:
            pass
            
    result["atomic_claims"] = [{"original": claim, "english": claim_translation}] if claim else []
    result["_parse_error"] = True
    return result


def _load_embedder():
    #Load sentence-transformers for logic-grounded dynamic selection.
    global _embedder, _pool_embeddings
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        pool_sentences = [ex["sentence"] for ex in DYNAMIC_POOL]
        _pool_embeddings = _embedder.encode(pool_sentences, normalize_embeddings=True)
    return _embedder, _pool_embeddings

async def async_chat(client: AsyncOpenAI, messages: list, temperature: float = 0.6, max_tokens: int = 2048, semaphore: asyncio.Semaphore = None) -> str:
    async with (semaphore if semaphore is not None else asyncio.Lock()):
        response = await client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return _strip_thinking(content)


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

# METHOD 1 - Zero-Shot
async def zero_shot_decomposition(dataset, client, semaphore, out_path, start_id=0, end_id=None):
    subset = dataset[start_id : end_id if end_id is not None else len(dataset)]
    saver = IncrementalSaver(out_path)

    system_prompt = '''You are an expert fact-checker and linguist.
    Your task is to decompose a claim into atomic claims (propositions - the smallest independently verifiable statements). If the claim is already atomic, return it as-is in a single-element list.
    Each atomic claim must be in the ORIGINAL language of the input claim. Additionally, provide an English translation for each atomic claim and the input claim itself.
    When decomposing, look for:
    - Coordinating conjunctions (and, but, or), commas joining independent clauses, or comparisons.
    - Nested or embedded claims: reported speech ('X said that …'), subordinate clauses ('that …', 'who …'), or presuppositions ('ignored evidence that …' presupposes the evidence exists). Each nesting level may contain its own verifiable fact - unpack them from the outermost layer inward.
    - If none of the above apply, the claim is already atomic.
    Respond ONLY with a valid JSON object in the following format (no markdown, no commentary):
    {
        "language": "<claim language>",
        "original_claim": "<the full original claim>",
        "original_claim_translation": "<English translation of the original claim>",
        "atomic_claims": [
            {"original": "<atomic claim in original language>", "english": "<English translation of the atomic claim>"}
        ]
    }'''

    async def process_item(row):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Language: {row['Language']}\nClaim: {row['Claim']}"},
        ]
        raw = await async_chat(client, messages, temperature=0.6, max_tokens=2048, semaphore=semaphore)
        parsed = _parse_json(
            raw_text=raw, 
            claim_id=row.get("ClaimID", ""),
            language=row["Language"], 
            claim=row["Claim"],
            claim_translation=row.get("Translation", "")
        )
        await saver.append(parsed)
        return parsed

    tasks = [process_item(row) for row in subset]

    for future in tqdm_asyncio.as_completed(tasks, desc="Processing Claims"):
        await future

    return {"status": "success", "file": out_path}

# METHOD 2 - Chain-of-Thought (CoT)
async def cot_decomposition(dataset, client, semaphore, out_path, start_id=0, end_id=None):
    subset = dataset[start_id : end_id if end_id is not None else len(dataset)]
    saver = IncrementalSaver(out_path)

    system_prompt = '''You are an expert fact-checker and linguist.
    Your task is to decompose a claim into atomic claims (propositions - the smallest independently verifiable statements).If the claim is already atomic, return it as-is.
    Each atomic claim must be in the ORIGINAL language of the input claim. Additionally, provide an English translation for each atomic claim and the input claim itself.
    Before producing your answer, reason through the following steps:
    1. Translate the claim to English to fully understand it.
    2. Identify the core subject(s), predicate(s), and any qualifiers (time, place, degree).
    3. Check for coordinating conjunctions (and, but, or), commas joining independent clauses, or comparisons that signal multiple facts.
    4. Check for NESTED or EMBEDDED claims - look for reported speech (e.g. 'X said that …'), subordinate clauses ('that …', 'who …'), or presuppositions (e.g. 'ignored evidence that …' presupposes the evidence exists). Each nesting level may contain its own verifiable fact.
    5. Split the claim so each resulting proposition contains exactly one verifiable fact. Unpack nested structures from the outermost layer inward.
    6. For each atomic claim, ensure it is self-contained - a reader should understand it without seeing the original compound claim. Resolve pronouns and implicit references.
    7. If no split is possible (no conjunctions, no nesting, no comparisons), the claim is already atomic - return it as a single-element list. Do not force a decomposition.
    Write your reasoning inside a "thought_process" field, then output the final JSON.
    Respond ONLY with a valid JSON object in the following format (no markdown, no extra commentary):
    {
      "language": "<claim language>",
      "original_claim": "<the full original claim>",
      "original_claim_translation": "<English translation of the original claim>",
      "thought_process": "<your step-by-step reasoning>",
      "atomic_claims": [
        {"original": "<atomic claim in original language>", "english": "<English translation>"}
      ]
    }
    --- begin EXAMPLE 1 (multiple nested claims) ---
    Language: uk
    Claim: "Міністр заявив, що ООН проігнорувала докази того, що Китай таємно постачає зброю Іспанії."
    RESPONSE:
    {
      "language": "uk",
      "original_claim": "Міністр заявив, що ООН проігнорувала докази того, що Китай таємно постачає зброю Іспанії.",
      "original_claim_translation": "The minister stated that the UN ignored evidence that China secretly supplies weapons to Spain.",
      "thought_process": "Step 1: The claim translates to: The minister stated that the UN ignored evidence that China secretly supplies weapons to Spain. Step 2: The outermost structure is reported speech - 'The minister stated that …'. This is itself a verifiable fact (did the minister say this?). Step 3: Inside the reported speech is the assertion 'the UN ignored evidence that …'. This is a second verifiable fact, independent of whether the minister said it. Step 4: The phrase 'evidence that China secretly supplies weapons to Spain' presupposes a third fact - that China secretly supplies weapons to Spain. This is nested inside 'ignored evidence that …' and is independently verifiable. Step 5: Splitting from outermost to innermost yields three atomic claims: (1) the minister made the statement, (2) the UN ignored evidence of Chinese arms supplies to Spain, (3) China secretly supplies weapons to Spain. Step 6: Each proposition is self-contained - subjects are fully named, no pronouns remain.",
      "atomic_claims": [
        {"original": "Міністр заявив, що ООН проігнорувала докази того, що Китай таємно постачає зброю Іспанії.", "english": "The minister stated that the UN ignored evidence that China secretly supplies weapons to Spain."},
        {"original": "ООН проігнорувала докази того, що Китай таємно постачає зброю Іспанії.", "english": "The UN ignored evidence that China secretly supplies weapons to Spain."},
        {"original": "Китай таємно постачає зброю Іспанії.", "english": "China secretly supplies weapons to Spain."}
      ]
    }
    --- end EXAMPLE 1 ---
    --- begin EXAMPLE 2 (no nesting, conjunction split) ---
    Language: de
    Claim: "Deutschland hat die meisten Einwohner in der EU und die stärkste Wirtschaft."
    RESPONSE:
    {
      "language": "de",
      "original_claim": "Deutschland hat die meisten Einwohner in der EU und die stärkste Wirtschaft.",
      "original_claim_translation": "Germany has the most inhabitants in the EU and the strongest economy.",
      "thought_process": "Step 1: The claim translates to: Germany has the most inhabitants in the EU and the strongest economy. Step 2: One subject (Germany) with two predicates joined by 'und' (and): 'has the most inhabitants in the EU' and 'has the strongest economy'. Step 3: The conjunction 'und' joins two independent factual assertions. No nested/embedded claims are present - no reported speech, no subordinate 'that' clauses. Step 4: Split at the conjunction. Proposition 1: Germany has the most inhabitants in the EU. Proposition 2: Germany has the strongest economy in the EU. Step 5: Both propositions are self-contained. The qualifier in the EU is carried to both for clarity.",
      "atomic_claims": [
        {"original": "Deutschland hat die meisten Einwohner in der EU.", "english": "Germany has the most inhabitants in the EU."},
        {"original": "Deutschland hat die stärkste Wirtschaft in der EU.", "english": "Germany has the strongest economy in the EU."}
      ]
    }
    --- end EXAMPLE 2 ---
    --- begin EXAMPLE 3 (already atomic) ---
    Language: pl
    Claim: "Warszawa jest stolicą Polski."
    RESPONSE:
    {
      "language": "pl",
      "original_claim": "Warszawa jest stolicą Polski.",
      "original_claim_translation": "Warsaw is the capital of Poland.",
      "thought_process": "Step 1: The claim translates to: Warsaw is the capital of Poland. Step 2: One subject (Warsaw), one predicate (is the capital), one qualifier (of Poland). Step 3: No conjunctions, no commas joining clauses, no comparisons. Step 4: No nested or embedded claims - no reported speech, no subordinate clauses, no presuppositions. Step 5: The claim contains exactly one verifiable fact. No split is possible - it is already atomic.",
      "atomic_claims": [
        {"original": "Warszawa jest stolicą Polski.", "english": "Warsaw is the capital of Poland."}
      ]
    }
    --- end EXAMPLE 3 ---
    Now, apply the same reasoning process to the following real claim:'''

    async def process_item(row):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Language: {row['Language']}\nClaim: {row['Claim']}"},
        ]
        raw = await async_chat(client, messages, temperature=0.6, max_tokens=4096, semaphore=semaphore)
        parsed = _parse_json(
            raw_text=raw, 
            claim_id=row.get("ClaimID", ""),
            language=row["Language"], 
            claim=row["Claim"],
            claim_translation=row.get("Translation", "")
        )
        await saver.append(parsed)
        return parsed

    tasks = [process_item(row) for row in subset]

    for future in tqdm_asyncio.as_completed(tasks, desc="Processing Claims"):
        await future

    return {"status": "success", "file": out_path}

# METHOD 3 - Logic-Grounded
async def logic_grounded_decomposition(dataset, client, semaphore, out_path, start_id=0, end_id=None):
    embedder, pool_embeddings = _load_embedder()
    subset = dataset[start_id : end_id if end_id is not None else len(dataset)]
    saver = IncrementalSaver(out_path)

    system_prompt = '''You are a linguist performing claim decomposition following Russell's theory of logical atomism and neo-Davidsonian semantics.
    Decompose each sentence into the absolute smallest independently verifiable units, where each unit designates a single property of an entity or a single relation between entities.
    Do NOT include 'existence' as a property (e.g. do not state that a mentioned person exists).
    Each fact must be in the ORIGINAL language of the sentence. After each fact, provide the English translation in parentheses prefixed with 'EN: '.
    First output a line 'Translation: <English translation of the full sentence>', then list the facts.'''

    instruction = "Please decompose the following sentence into individual atomic claims:"

    def _format_example(ex):
        lines = [f"Sentence [{ex['language']}]: {ex['sentence']}"]
        if ex["language"] != "en":
            translations = [f["english"] for f in ex["facts"]]
            lines.append(f"Translation: {' '.join(translations)}")
        lines.append("Facts:")
        for f in ex["facts"]:
            if ex["language"] == "en":
                lines.append(f"- {f['original']}")
            else:
                lines.append(f"- {f['original']} (EN: {f['english']})")
        return "\n".join(lines)

    static_block = "\n\n".join(_format_example(ex) for ex in STATIC_EXAMPLES)

    all_claims = [r["Claim"] for r in subset]
    claim_embeddings = embedder.encode(all_claims, normalize_embeddings=True)
    similarities = claim_embeddings @ pool_embeddings.T

    async def process_item(i, row):
        best_pool_idx = int(np.argmax(similarities[i]))
        dynamic_block = _format_example(DYNAMIC_POOL[best_pool_idx])
        user_msg = (
            f"{instruction}\n\n"
            f"{static_block}\n\n"
            f"{dynamic_block}\n\n"
            f"Sentence [{row['Language']}]: {row['Claim']}\nFacts:\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        raw = await async_chat(client, messages, temperature=0.7, max_tokens=2048, semaphore=semaphore)

        raw_clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw_clean = re.sub(r'^Thinking Process:.*?(?=\nTranslation:|\n-|\nFact)', '', raw_clean, flags=re.DOTALL).strip()

        original_claim_translation = ""
        atomic_claims = []
        for line in raw_clean.split("\n"):
            line = line.strip()
            if line.lower().startswith("translation:"):
                original_claim_translation = line.split(":", 1)[1].strip()
            elif line.startswith("- "):
                fact_text = line[2:].strip()
                en_match = re.search(r'\(EN:\s*(.+?)\)\s*$', fact_text)
                if en_match:
                    english = en_match.group(1).strip()
                    original = fact_text[:en_match.start()].strip()
                else:
                    original = fact_text
                    english = ""
                atomic_claims.append({"original": original, "english": english})

        if not atomic_claims:
            atomic_claims = [{"original": row["Claim"], "english": row.get("Translation", "")}]
            original_claim_translation = row.get("Translation", "")

        result = {
            "claim_id": row.get("ClaimID", ""),
            "language": row["Language"],
            "original_claim": row["Claim"],
            "original_claim_translation": original_claim_translation,
            "atomic_claims": atomic_claims,
        }
        await saver.append(result)
        return result

    tasks = [process_item(i, row) for i, row in enumerate(subset)]

    for future in tqdm_asyncio.as_completed(tasks, desc="Processing Claims"):
        await future

    return {"status": "success", "file": out_path}

# METHOD 4 - Hybrid (CoT Reasoning + Logic-Grounded Few-Shot)
async def hybrid_decomposition(dataset, client, semaphore, out_path, start_id=0, end_id=None):
    embedder, pool_embeddings = _load_embedder()
    subset = dataset[start_id : end_id if end_id is not None else len(dataset)]
    saver = IncrementalSaver(out_path)

    system_prompt = '''You are an expert fact-checker and linguist performing claim decomposition.
    Your task is to decompose a claim into atomic claims (propositions - the smallest independently verifiable statements). If the claim is already atomic, return it as-is.
    Each atomic claim must be in the ORIGINAL language of the input claim. Additionally, provide an English translation for each atomic claim and the input claim itself.
    Before producing your answer, reason through the following steps:
    1. Translate the claim to English to fully understand it.
    2. Identify the core subject(s), predicate(s), and any qualifiers (time, place, degree).
    3. Check for coordinating conjunctions (and, but, or), commas joining independent clauses, or comparisons that signal multiple facts.
    4. Check for NESTED or EMBEDDED claims - look for reported speech (e.g. 'X said that …'), subordinate clauses ('that …', 'who …'), or presuppositions (e.g. 'ignored evidence that …' presupposes the evidence exists). Each nesting level may contain its own verifiable fact.
    5. Split the claim so each resulting proposition contains exactly ONE verifiable fact. Unpack nested structures from the outermost layer inward.
    6. For each atomic claim, ensure it is SELF-CONTAINED - a reader should understand it without seeing the original compound claim. Resolve all pronouns, anaphora, and implicit references.
    7. AVOID REDUNDANCY - each atomic claim must convey a distinct, unique fact. Do not repeat the same information in multiple claims.
    8. Do NOT include trivial 'existence' claims (e.g. do not state that a mentioned person exists).
    9. If no split is possible (no conjunctions, no nesting, no comparisons), the claim is already atomic - return it as a single-element list. Do not force a decomposition.
    Write your reasoning inside a "thought_process" field, then output the final JSON.
    Respond ONLY with a valid JSON object in the following format (no markdown, no extra commentary):
    {
      "language": "<claim language>",
      "original_claim": "<the full original claim>",
      "original_claim_translation": "<English translation of the original claim>",
      "thought_process": "<your step-by-step reasoning>",
      "atomic_claims": [
        {"original": "<atomic claim in original language>", "english": "<English translation>"}
      ]
    }'''

    def _format_example_as_json(ex):
        #Format a DR-ND pool example as JSON to match the expected output format.
        claims_json = ", ".join(
            f'{{"original": "{f["original"]}", "english": "{f["english"]}"}}'
            for f in ex["facts"]
        )
        translation = " ".join(f["english"] for f in ex["facts"])
        return (
            f"--- begin EXAMPLE ---\n"
            f"Language: {ex['language']}\n"
            f"Claim: \"{ex['sentence']}\"\n\n"
            f"RESPONSE:\n"
            f"{{\n"
            f'  "language": "{ex["language"]}",\n'
            f'  "original_claim": "{ex["sentence"]}",\n'
            f'  "original_claim_translation": "{translation}",\n'
            f'  "thought_process": "Decomposed following logical atomism: each unit designates '
            f'a single property or relation.",\n'
            f'  "atomic_claims": [{claims_json}]\n'
            f"}}\n"
            f"--- end EXAMPLE ---"
        )

    all_claims = [r["Claim"] for r in subset]
    claim_embeddings = embedder.encode(all_claims, normalize_embeddings=True)
    similarities = claim_embeddings @ pool_embeddings.T
    static_examples_block = "\n\n".join(_format_example_as_json(ex) for ex in STATIC_EXAMPLES)

    async def process_item(i, row):
        best_pool_idx = int(np.argmax(similarities[i]))
        dynamic_block = _format_example_as_json(DYNAMIC_POOL[best_pool_idx])

        user_msg = (
            f"Here are some examples of correct decomposition:\n\n"
            f"{static_examples_block}\n\n"
            f"{dynamic_block}\n\n"
            f"Now, apply the same reasoning process to the following real claim:\n\n"
            f"Language: {row['Language']}\nClaim: {row['Claim']}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        raw = await async_chat(client, messages, temperature=0.6, max_tokens=4096, semaphore=semaphore)
        parsed = _parse_json(
            raw_text=raw,
            claim_id=row.get("ClaimID", ""),
            language=row["Language"],
            claim=row["Claim"],
            claim_translation=row.get("Translation", "")
        )
        await saver.append(parsed)
        return parsed

    tasks = [process_item(i, row) for i, row in enumerate(subset)]

    for future in tqdm_asyncio.as_completed(tasks, desc="Processing Claims"):
        await future

    return {"status": "success", "file": out_path}

async def main():
    global _server_process
    parser = argparse.ArgumentParser(description="Claim Decomposition Pipeline via llama-server (OpenAI API)")
    
    # Method & Data
    parser.add_argument("--method", type=str, required=True,
                        choices=["zero_shot", "cot", "logic_grounded", "hybrid"],
                        help="Method to use for decomposition")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples to process (default: 5)")
    parser.add_argument("--dataset_split", type=str, default="decomposable",
                        choices=["dev", "test", "decomposable"],
                        help="Dataset split to use")

    # llama.cpp server arguments
    parser.add_argument("--model-hf", type=str, default="unsloth/Qwen3.5-9B-GGUF:Q8_0",
                        help="Hugging Face repo or local path for GGUF")
    parser.add_argument("-c", "--ctx-size", type=int, default=16384,
                        help="Context window size (default: 16384)")
    parser.add_argument("-np", "--parallel", type=int, default=3,
                        help="Number of slots / parallel requests (default: 3)")
    parser.add_argument("--ngl", type=int, default=99,
                        help="Number of GPU layers to offload")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the llama-server (default: 8000)")
    parser.add_argument("--enable-reasoning", action="store_true",
                        help="Set to enable reasoning (defaults to -rea off)")

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

        test_data = load_dataset_csv(args.dataset_split)
        N = args.samples
        method_name = args.method
        
        out_path = os.path.join(RESULTS_DIR, f"{method_name}_results.jsonl")

        print(f"\n{'='*60}")
        print(f"Running: {method_name} on {N} samples (Concurrency: {args.parallel})")
        print(f"Output: {out_path}")
        print(f"{'='*60}\n")

        start = time.time()
        
        if method_name == "zero_shot":
            results = await zero_shot_decomposition(test_data, client, semaphore, out_path, 0, N)
        elif method_name == "cot":
            results = await cot_decomposition(test_data, client, semaphore, out_path, 0, N)
        elif method_name == "logic_grounded":
            results = await logic_grounded_decomposition(test_data, client, semaphore, out_path, 0, N)
        elif method_name == "hybrid":
            results = await hybrid_decomposition(
                test_data, client, semaphore, out_path, 0, N
            )

        elapsed = time.time() - start
        print(f"\n[+] Processing finished in {elapsed:.2f}s.")


    finally:
        cleanup_server()

if __name__ == "__main__":
    asyncio.run(main())
