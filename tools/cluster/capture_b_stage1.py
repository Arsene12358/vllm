"""Option B Stage 1 verification — bounded decode positions.

MODE=smoke (default): a small prompt that triggers eviction (> start+recent),
bounded ON, short generation. Goal: no crash (the decode-position pin is the
risk), the "[streaming-kv][bounded] decode M-RoPE positions pinned at front="
log appears, and output is coherent.

MODE=long: a ~64k-token prompt with a sink needle + question, generate enough
that the *total* (prompt+generated) crosses 65536 mid-generation. Compare
bounded ON (decode pos pinned at S+R → coherent throughout) vs OFF (decode pos
grows past 65536 → degrades). Mirrors job 1814 but in the generation dimension.
"""
import gc
import json
import os
import re
import sys
import time
from collections import Counter

MODE = os.environ.get("MODE", "smoke")
MODEL = "/model"
START = int(os.environ.get("STREAM_START_SIZE", "128"))
RECENT = int(os.environ.get("STREAM_RECENT_SIZE", "2048"))
TP = int(os.environ.get("TP", "2"))
PASSCODE = "SWORDFISH-7741"

FILLER = [
    "The committee reviewed the quarterly figures before lunch. ",
    "A light rain fell over the harbor as the ferries departed. ",
    "She filed the report and moved on to the next assignment. ",
    "The engineers debated the merits of the new cooling design. ",
]


def templ(user):
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n")


def make_filler(n_tokens, tok):
    sample = "".join(FILLER[i % len(FILLER)] for i in range(200))
    per_char = len(tok(sample).input_ids) / len(sample)
    chars = int(n_tokens / per_char * 0.98)
    return (sample * (chars // len(sample) + 1))[:chars]


def repetition_ratio(text):
    w = re.findall(r"\w+", text.lower())
    if len(w) < 8:
        return 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    c = Counter(grams)
    return round(sum(v - 1 for v in c.values()) / len(grams), 3)


def run(bounded, prompt, max_tokens):
    import vllm_omni.patch  # noqa: F401
    from vllm import LLM, SamplingParams
    tag = "bounded_ON" if bounded else "bounded_OFF"
    print(f"\n[{time.strftime('%H:%M:%S')}] === {tag} max_tokens={max_tokens} ===",
          flush=True)
    mml = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    llm = LLM(model=MODEL, tensor_parallel_size=TP, trust_remote_code=True,
              dtype="bfloat16", seed=42, enforce_eager=True,
              gpu_memory_utilization=0.85, max_model_len=mml,
              streaming_kv_start_size=START, streaming_kv_recent_size=RECENT,
              streaming_kv_bounded_positions=bounded)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
    t0 = time.time()
    out = llm.generate([{"prompt": prompt}], sp)
    dt = time.time() - t0
    ptoks = len(out[0].prompt_token_ids)
    text = out[0].outputs[0].text
    comp = len(out[0].outputs[0].token_ids)
    has_pass = PASSCODE in text
    rep = repetition_ratio(text)
    print(f"  prompt_tokens={ptoks} total={ptoks+comp} completion={comp} "
          f"{dt:.1f}s passcode={has_pass} rep={rep}", flush=True)
    print(f"  text[:300]: {text[:300]!r}", flush=True)
    print(f"  text[-200:]: {text[-200:]!r}", flush=True)
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"tag": tag, "prompt_tokens": ptoks, "total": ptoks + comp,
            "completion": comp, "passcode": has_pass, "rep": rep, "text": text}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"[{time.strftime('%H:%M:%S')}] B Stage1 MODE={MODE} start={START} "
          f"recent={RECENT} front={START+RECENT}", flush=True)

    if MODE == "smoke":
        # ~2500-token prompt (> start+recent → eviction), short gen.
        sink = f"Remember the secret passcode: {PASSCODE}. "
        user = sink + make_filler(2500, tok) + " What is the secret passcode?"
        prompt = templ(user)
        ntok = len(tok(prompt).input_ids)
        print(f"  smoke prompt tokens={ntok} (front={START+RECENT})", flush=True)
        r = run(bounded=True, prompt=prompt, max_tokens=64)
        ok = r["completion"] > 5 and r["rep"] < 0.3
        print(f"\n[B-STAGE1-SMOKE] {'PASS' if ok else 'CHECK'} — no crash, "
              f"completion={r['completion']} rep={r['rep']} passcode={r['passcode']}",
              flush=True)
        return 0

    # MODE == long
    target = int(os.environ.get("LONG_PROMPT_TOKENS", "64000"))
    sink = f"Remember the secret passcode: {PASSCODE}. "
    tail = " Now keep describing the scene in detail; also, what is the secret passcode I gave at the very beginning?"
    user = sink + make_filler(target, tok) + tail
    prompt = templ(user)
    ntok = len(tok(prompt).input_ids)
    maxt = int(os.environ.get("MAX_TOKENS", "2048"))
    print(f"  long prompt tokens={ntok}; total at end ~= {ntok+maxt} "
          f"(crosses 65536: {ntok+maxt > 65536})", flush=True)
    on = run(bounded=True, prompt=prompt, max_tokens=maxt)
    off = run(bounded=False, prompt=prompt, max_tokens=maxt)
    print("\n=== B Stage1 LONG summary ===", flush=True)
    for r in (off, on):
        print(f"  {r['tag']:11s} total={r['total']} comp={r['completion']} "
              f"passcode={r['passcode']} rep={r['rep']}", flush=True)
    decisive = (on["rep"] < 0.3 and on["passcode"]) and \
               ((not off["passcode"]) or off["rep"] >= 0.3)
    print(f"[B-STAGE1-LONG] {'PASS' if decisive else 'INCONCLUSIVE'} — bounded "
          f"coherent+recall past 65536 where unbounded degrades.", flush=True)
    json.dump({"on": on, "off": off},
              open("/workspace/golden/b_stage1_long.json", "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
