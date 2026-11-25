import argparse
import asyncio
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
for candidate in (SCRIPT_DIR, PARENT_DIR, os.getcwd()):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from drnd_examples import DYNAMIC_POOL, STATIC_EXAMPLES


def _resolve_dataset_dir() -> str:
    for base in (os.getcwd(), SCRIPT_DIR, PARENT_DIR):
        path = os.path.join(base, "data")
        if os.path.isdir(path):
            return path
    return os.path.join(os.getcwd(), "data")


def _resolve_results_dir() -> str:
    if os.path.basename(SCRIPT_DIR) == "additional":
        out = os.path.join(PARENT_DIR, "results", "ablations")
    else:
        out = os.path.join(os.getcwd(), "results", "ablations")
    os.makedirs(out, exist_ok=True)
    return out


_embedder = None
_pool_embeddings = None
_server_process = None

ABLATION_CHOICES = ["no_dynamic", "no_constraints", "no_cot"]


def load_dataset_csv(split: str = "stratified_mini") -> list:
    dataset_dir = _resolve_dataset_dir()
    path = os.path.join(dataset_dir, f"MultiClaim_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def is_server_ready(base_url: str) -> bool:
    for endpoint in ("/health", "/v1/models"):
        try:
            req = urllib.request.Request(
                f"{base_url}{endpoint}", headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False


def start_llama_server(args) -> subprocess.Popen:
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
        "--port", str(args.port),
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
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(
        r"^Thinking Process:.*?(?=\n\{|\n-|\nFact|\nTranslation:|\nTerminate)",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    if text and text[0] != "{" and "{" in text:
        text = text[text.index("{") :]
    return text


def _parse_json(
    raw_text: str,
    claim_id: str = "",
    language: str = "",
    claim: str = "",
    claim_translation: str = "",
) -> dict:
    result = {
        "claim_id": claim_id,
        "language": language,
        "original_claim": claim,
        "original_claim_translation": claim_translation,
        "atomic_claims": [],
    }

    if not raw_text.strip():
        result["atomic_claims"] = (
            [{"original": claim, "english": claim_translation}] if claim else []
        )
        result["_parse_error"] = True
        return result

    for candidate in (raw_text, (re.search(r"\{.*\}", raw_text, re.DOTALL) or None)):
        text_chunk = candidate.group() if isinstance(candidate, re.Match) else candidate
        if not text_chunk:
            continue
        try:
            parsed = json.loads(text_chunk)
            result["atomic_claims"] = parsed.get("atomic_claims", [])
            if "thought_process" in parsed:
                result["thought_process"] = parsed["thought_process"]
            return result
        except json.JSONDecodeError:
            continue

    result["atomic_claims"] = (
        [{"original": claim, "english": claim_translation}] if claim else []
    )
    result["_parse_error"] = True
    return result


def _load_embedder():
    global _embedder, _pool_embeddings
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        pool_sentences = [ex["sentence"] for ex in DYNAMIC_POOL]
        _pool_embeddings = _embedder.encode(pool_sentences, normalize_embeddings=True)
    return _embedder, _pool_embeddings


async def async_chat(
    client: AsyncOpenAI,
    messages: list,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    semaphore: asyncio.Semaphore = None,
) -> str:
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
        with open(self.filepath, "w", encoding="utf-8"):
            pass

    async def append(self, result: dict):
        async with self.lock:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


def _build_hybrid_system_prompt(ablation: str) -> str:
    header = '''You are an expert fact-checker and linguist performing claim decomposition.
    Your task is to decompose a claim into atomic claims (propositions - the smallest independently verifiable statements). If the claim is already atomic, return it as-is.
    Each atomic claim must be in the ORIGINAL language of the input claim. Additionally, provide an English translation for each atomic claim and the input claim itself.'''

    thought_field = '\n      "thought_process": "<your step-by-step reasoning>",' if ablation != "no_cot" else ""
    schema = '''Respond ONLY with a valid JSON object in the following format (no markdown, no extra commentary):
    {
      "language": "<claim language>",
      "original_claim": "<the full original claim>",
      "original_claim_translation": "<English translation of the original claim>",__THOUGHT_FIELD__
      "atomic_claims": [
        {"original": "<atomic claim in original language>", "english": "<English translation>"}
      ]
    }'''.replace("__THOUGHT_FIELD__", thought_field)

    if ablation == "no_cot":
        body = '''Follow these decomposition rules:
        - Split at coordinating conjunctions (and, but, or), commas joining independent clauses, or comparisons that signal multiple facts.
        - Unpack NESTED or EMBEDDED claims from the outermost layer inward - look for reported speech (e.g. 'X said that …'), subordinate clauses ('that …', 'who …'), or presuppositions (e.g. 'ignored evidence that …' presupposes the evidence exists).
        - Ensure each resulting proposition contains exactly ONE verifiable fact and is SELF-CONTAINED (resolve all pronouns, anaphora, and implicit references).
        - AVOID REDUNDANCY - each atomic claim must convey a distinct, unique fact. Do not repeat the same information in multiple claims.
        - Do NOT include trivial 'existence' claims (e.g. do not state that a mentioned person exists).
        - If no split is possible, return the claim as a single-element list. Do not force a decomposition.'''
        return "\n".join([header, body, schema])

    # CoT reasoning steps (shared by no_dynamic and no_constraints)
    cot_steps = '''Before producing your answer, reason through the following steps:
    1. Translate the claim to English to fully understand it.
    2. Identify the core subject(s), predicate(s), and any qualifiers (time, place, degree).
    3. Check for coordinating conjunctions (and, but, or), commas joining independent clauses, or comparisons that signal multiple facts.
    4. Check for NESTED or EMBEDDED claims - look for reported speech (e.g. 'X said that …'), subordinate clauses ('that …', 'who …'), or presuppositions (e.g. 'ignored evidence that …' presupposes the evidence exists). Each nesting level may contain its own verifiable fact.
    5. Split the claim so each resulting proposition contains exactly ONE verifiable fact. Unpack nested structures from the outermost layer inward.
    6. For each atomic claim, ensure it is SELF-CONTAINED - a reader should understand it without seeing the original compound claim. Resolve all pronouns, anaphora, and implicit references.'''

    if ablation == "no_constraints":
        constraints_and_closing = '''7. If no split is possible (no conjunctions, no nesting, no comparisons), the claim is already atomic - return it as a single-element list. Do not force a decomposition.'''
    else:
        constraints_and_closing = '''7. AVOID REDUNDANCY - each atomic claim must convey a distinct, unique fact. Do not repeat the same information in multiple claims.
    8. Do NOT include trivial 'existence' claims (e.g. do not state that a mentioned person exists).
    9. If no split is possible (no conjunctions, no nesting, no comparisons), the claim is already atomic - return it as a single-element list. Do not force a decomposition.'''

    cot_closing = 'Write your reasoning inside a "thought_process" field, then output the final JSON.'
    return "\n".join([header, cot_steps, constraints_and_closing, cot_closing, schema])


def _format_example_as_json(ex: dict, include_thought_process: bool) -> str:
    """Format a DR-ND pool example as JSON matching the active ablation schema."""
    translation = " ".join(f["english"] for f in ex["facts"])
    payload = {
        "language": ex["language"],
        "original_claim": ex["sentence"],
        "original_claim_translation": translation,
    }
    if include_thought_process:
        payload["thought_process"] = (
            "Decomposed following logical atomism: each unit designates "
            "a single property or relation."
        )
    payload["atomic_claims"] = [
        {"original": f["original"], "english": f["english"]} for f in ex["facts"]
    ]
    return (
        "--- begin EXAMPLE ---\n"
        f"Language: {ex['language']}\n"
        f'Claim: "{ex["sentence"]}"\n\n'
        "RESPONSE:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "--- end EXAMPLE ---"
    )


async def run_hybrid_ablation(
    ablation: str,
    dataset: list,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    out_path: str,
    start_id: int = 0,
    end_id: int = None,
) -> dict:
    subset = dataset[start_id : end_id if end_id is not None else len(dataset)]
    saver = IncrementalSaver(out_path)

    include_cot = ablation != "no_cot"
    use_dynamic = ablation != "no_dynamic"

    system_prompt = _build_hybrid_system_prompt(ablation)
    static_examples_block = "\n\n".join(
        _format_example_as_json(ex, include_thought_process=include_cot)
        for ex in STATIC_EXAMPLES
    )

    similarities = None
    if use_dynamic:
        embedder, pool_embeddings = _load_embedder()
        all_claims = [r["Claim"] for r in subset]
        claim_embeddings = embedder.encode(all_claims, normalize_embeddings=True)
        similarities = claim_embeddings @ pool_embeddings.T

    prompt_verb = (
        "apply the same reasoning process to"
        if include_cot
        else "decompose"
    )

    async def process_item(i: int, row: dict):
        examples_sections = [static_examples_block]
        if use_dynamic:
            best_pool_idx = int(np.argmax(similarities[i]))
            examples_sections.append(
                _format_example_as_json(
                    DYNAMIC_POOL[best_pool_idx], include_thought_process=include_cot
                )
            )
        examples_block = "\n\n".join(examples_sections)

        user_msg = (
            "Here are some examples of correct decomposition:\n\n"
            f"{examples_block}\n\n"
            f"Now, {prompt_verb} the following real claim:\n\n"
            f"Language: {row['Language']}\nClaim: {row['Claim']}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        raw = await async_chat(
            client, messages, temperature=0.6, max_tokens=4096, semaphore=semaphore
        )
        parsed = _parse_json(
            raw_text=raw,
            claim_id=row.get("ClaimID", ""),
            language=row["Language"],
            claim=row["Claim"],
            claim_translation=row.get("Translation", ""),
        )
        await saver.append(parsed)
        return parsed

    tasks = [process_item(i, row) for i, row in enumerate(subset)]
    for future in tqdm_asyncio.as_completed(tasks, desc=f"hybrid_{ablation}"):
        await future

    return {"status": "success", "file": out_path}


async def main():
    global _server_process
    parser = argparse.ArgumentParser(
        description="Run Hybrid Claim Decomposition Ablations via llama-server"
    )

    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", *ABLATION_CHOICES],
        help="Which Hybrid ablation to run (default: all three sequentially)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to process (default: 100)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="stratified_mini",
        choices=["dev", "test", "stratified", "stratified_mini", "decomposable"],
        help="Dataset split to use (default: stratified_mini)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for output JSONL files (default: results/ablations)",
    )

    # llama.cpp server arguments (match main_v3.py)
    parser.add_argument(
        "--model-hf",
        type=str,
        default="unsloth/Qwen3.5-9B-GGUF:Q8_0",
        help="Hugging Face repo or local path for GGUF",
    )
    parser.add_argument(
        "-c", "--ctx-size", type=int, default=16384, help="Context window size"
    )
    parser.add_argument(
        "-np", "--parallel", type=int, default=3, help="Number of parallel slots"
    )
    parser.add_argument(
        "--ngl", type=int, default=99, help="Number of GPU layers to offload"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for llama-server"
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Set to enable reasoning (defaults to -rea off)",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    results_dir = args.output_dir or _resolve_results_dir()
    os.makedirs(results_dir, exist_ok=True)

    dataset = load_dataset_csv(args.dataset_split)
    to_run = ABLATION_CHOICES if args.ablation == "all" else [args.ablation]

    try:
        _server_process = start_llama_server(args)
        client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{args.port}/v1",
            api_key="no-key-required",
        )
        semaphore = asyncio.Semaphore(args.parallel)

        for ablation in to_run:
            out_path = os.path.join(results_dir, f"hybrid_{ablation}_results.jsonl")
            print(f"\n{'='*60}")
            print(
                f"Running ablation: hybrid_{ablation} on {args.samples} samples "
                f"(split: {args.dataset_split}, concurrency: {args.parallel})"
            )
            print(f"Output: {out_path}")
            print(f"{'='*60}\n")

            start = time.time()
            await run_hybrid_ablation(
                ablation=ablation,
                dataset=dataset,
                client=client,
                semaphore=semaphore,
                out_path=out_path,
                start_id=0,
                end_id=args.samples,
            )
            elapsed = time.time() - start
            print(f"\n[+] Finished hybrid_{ablation} in {elapsed:.2f}s -> {out_path}")
    finally:
        cleanup_server()


if __name__ == "__main__":
    asyncio.run(main())
