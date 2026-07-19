"""Stage-B base model: a pinned small open-weight instruct model.

Design: the benchmark asks about a fixed set of (subject, "capital") facts, so
model I/O is done ONCE per fact at load time and cached — greedy answer, a
logit-based confidence (mean token probability of the answer), and a
consistency confidence (agreement of k temperature samples with the greedy
answer). Conversations then run as pure lookups, exactly like Stage A, which
keeps the grid deterministic given a cache and makes reruns cheap.

The adapter never sees world truth (anti-circularity): answers are snapped to
the nearest KNOWN capital by string containment over the public capital list —
a normalization step, not a correctness oracle.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

from .basemodel import Answer
from .worldb import CAPITALS

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # PROTOCOL: FROZEN for the Stage-B grid
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
K_SAMPLES_B = 8          # PROTOCOL: revised from Stage-A 15 (compute), frozen
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_P = 0.95
MAX_NEW_TOKENS = 12

# checked against NORMALIZED text (punctuation stripped, so "don't" -> "don t")
_DECLINE_MARKERS = ("unknown", "don t know", "do not know", "not sure", "no idea",
                    "cannot answer", "can t answer")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the\s+", "", text)


_NORM_CAPITALS = {normalize(c): c for c in CAPITALS}


def snap(text: str) -> str | None:
    """Map raw model output to a canonical capital, a decline (None), or the
    raw normalized string (an unknown/hallucinated place name)."""
    n = normalize(text)
    if not n or any(m in n for m in _DECLINE_MARKERS):
        return None
    for norm_cap, cap in _NORM_CAPITALS.items():
        if norm_cap == n or f" {norm_cap} " in f" {n} " or n.startswith(norm_cap):
            return cap
    return n  # not a known capital: keep it (scored false vs truth downstream)


class LLMBaseModel:
    """query()/extract() over a per-fact cache; interface-compatible with the
    Stage-A stub via the pipeline's `extract` hook."""

    def __init__(self, subjects: list[str], seed: int, cache_path: Path | None = None,
                 device: str | None = None):
        self.subjects = subjects
        self.seed = seed
        self.cache: dict[str, dict] = {}
        self.meta: dict = {}
        if cache_path is not None and cache_path.exists():
            data = json.loads(cache_path.read_text())
            self.cache, self.meta = data["cache"], data["meta"]
            if self.meta.get("model_id") != MODEL_ID:
                raise ValueError(
                    f"cache model mismatch: {self.meta.get('model_id')!r} != {MODEL_ID!r}"
                )
            if self.meta.get("revision") != MODEL_REVISION:
                raise ValueError(
                    "cache revision mismatch: "
                    f"{self.meta.get('revision')!r} != {MODEL_REVISION!r}"
                )
            return
        self._build_cache(device)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"cache": self.cache, "meta": self.meta}, indent=1))

    # --- inference (load time only) ------------------------------------------
    def _build_cache(self, device: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32
        )
        model.to(device).eval()
        revision = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
        if revision != MODEL_REVISION:
            raise RuntimeError(f"loaded revision {revision!r}, expected {MODEL_REVISION!r}")
        torch.manual_seed(self.seed)

        latencies = []
        for subject in self.subjects:
            msgs = [
                {"role": "system", "content": "You answer geography questions with only the "
                 "name, nothing else. If you do not know, answer exactly: unknown."},
                {"role": "user", "content": f"What is the capital of {subject}?"},
            ]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, attention_mask=attn,
                                     max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     output_scores=True, return_dict_in_generate=True,
                                     pad_token_id=tok.eos_token_id)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            new_tokens = out.sequences[0, ids.shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            value = snap(text)
            # logit confidence: mean probability of the generated answer tokens
            probs = []
            for tok_id, score in zip(new_tokens.tolist(), out.scores):
                p = torch.softmax(score[0], dim=-1)[tok_id].item()
                probs.append(p)
                if tok_id == tok.eos_token_id:
                    break
            logit_conf = float(np.mean(probs)) if probs else 0.0
            # consistency: agreement of k temperature samples with the greedy value
            agree = 0
            for _ in range(K_SAMPLES_B):
                with torch.no_grad():
                    s_out = model.generate(ids, attention_mask=attn,
                                           max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                           temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
                                           pad_token_id=tok.eos_token_id)
                s_text = tok.decode(s_out[0, ids.shape[1]:], skip_special_tokens=True)
                agree += (snap(s_text) == value)
            self.cache[subject] = {
                "value": value, "raw": text.strip(),
                "logit": logit_conf,
                "consistency": agree / K_SAMPLES_B if value is not None else 0.0,
            }
        self.meta = {
            "model_id": MODEL_ID, "revision": revision, "device": device,
            "seed": self.seed, "k_samples": K_SAMPLES_B,
            "temperature": SAMPLE_TEMPERATURE, "top_p": SAMPLE_TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "query_latency_ms_median": float(np.median(latencies)),
            "query_latency_ms_p95": float(np.percentile(latencies, 95)),
            "n_facts": len(self.subjects),
        }

    # --- pipeline interface ----------------------------------------------------
    def query(self, subject: str, attribute: str) -> Answer | None:
        if attribute != "capital":
            return None  # continent is never parametric (mirrors Stage-A region)
        entry = self.cache.get(subject)
        if entry is None or entry["value"] is None:
            return None
        return Answer(entry["value"], entry["logit"])

    def extract(self, subject: str, attribute: str, answer: Answer, rng) -> dict[str, float]:
        entry = self.cache[subject]
        return {"logit": entry["logit"], "consistency": entry["consistency"]}
