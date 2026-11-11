# CSE 519 pilot: How Good Are LLMs at 20 Questions?  (python twenty_questions_pilot.py [--mock])
import json, math, os, random, re, sys, threading, time
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Run the 20 Questions LLM pilot.")
parser.add_argument("--mock", action="store_true", help="use deterministic mock responses")
parser.add_argument("--pilot", action="store_true", help="run repeated local Llama self-play only")
parser.add_argument("--games", type=int, default=80, help="number of games in the local pilot")
parser.add_argument("--pilot-file", default="pilot_games.jsonl", help="separate pilot results file")
parser.add_argument("--repeats", type=int, default=1, help="repeats per ordered model pair and target")
parser.add_argument("--results-file", default="hf_ollama_games.jsonl", help="model-matrix results file")
parser.add_argument("--smoke-models", action="store_true", help="test one local generation from each model")
parser.add_argument("--model-set", choices=("original", "additional"), default="original",
                    help="run the original four models or three additional HF models plus Ollama")
ARGS = parser.parse_args()
MOCK = ARGS.mock

OLLAMA_MODEL = {
    "name": "Llama (Ollama)", "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "key_env": "OLLAMA_API_KEY", "backend": "ollama", "gap": 0,
}
ORIGINAL_MODELS = [
    {"name": "Qwen3 0.6B (HF local)", "model": "Qwen/Qwen3-0.6B",
     "revision": "c1899de289a04d12100db370d81485cdf75e47ca", "backend": "transformers"},
    {"name": "Qwen2.5 1.5B (HF local)", "model": "Qwen/Qwen2.5-1.5B-Instruct",
     "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "backend": "transformers"},
    {"name": "SmolLM2 1.7B (HF local)", "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
     "revision": "31b70e2e869a7173562077fd711b654946d38674", "backend": "transformers"},
    OLLAMA_MODEL,
]
ADDITIONAL_MODELS = [
    {"name": "Phi-3.5 Mini (HF local)", "model": "microsoft/Phi-3.5-mini-instruct",
     "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77", "backend": "transformers"},
    {"name": "Granite 3.3 2B (HF local)", "model": "ibm-granite/granite-3.3-2b-instruct",
     "revision": "707f574c62054322f6b5b04b6d075f0a8f05e0f0", "backend": "transformers",
     "template_kwargs": {"thinking": False}},
    {"name": "Qwen2.5 3B (HF local)", "model": "Qwen/Qwen2.5-3B-Instruct",
     "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "backend": "transformers"},
]
MODELS = ORIGINAL_MODELS if ARGS.model_set == "original" else ADDITIONAL_MODELS + [OLLAMA_MODEL]
SECRET_WORDS = {"banana": [], "bicycle": ["bike"], "elephant": []}
MAX_Q = 20
TEMPERATURE = 0.7
N_PICKS = 50
WORKERS = 1
SEED = 519

ANSWERER_SYS = ("You are playing 20 Questions. The secret object is: {secret}. "
                "Answer each question about the secret object truthfully with exactly one word: "
                "Yes, No, or Sometimes. Never reveal the secret object.")
GUESSER_SYS = ("You are playing 20 Questions. Another player is thinking of a common object. "
               "You may ask up to 20 yes/no questions. Each turn, reply with exactly one line, either\n"
               "QUESTION: <a yes/no question>\nor\nGUESS: <one object>\n"
               "A guess uses up a turn. Try to identify the object in as few turns as possible.")
PICK_PROMPT = ("We are about to play 20 Questions and you will be the one thinking of the object. "
               "Choose a common physical object completely at random. "
               "Reply with only the object's name, one or two words, nothing else.")

_clients, _locks, _last = {}, {}, {}
_hf_models = {}
_hf_lock = threading.RLock()
_active_hf_pair = None

def prepare_hf_pair(*models):
    global _active_hf_pair
    needed = {m["model"] for m in models if m.get("backend") == "transformers"}
    with _hf_lock:
        if needed == _active_hf_pair:
            return
        stale = set(_hf_models) - needed
        for model_name in stale:
            del _hf_models[model_name]
        if stale:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _active_hf_pair = needed
def _pace(m):
    lock = _locks.setdefault(m["name"], threading.Lock())
    with lock:
        wait = _last.get(m["name"], 0) + m.get("gap", 3) - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[m["name"]] = time.time()

def fix_msgs(m, messages):
    if not m.get("no_system"):
        return messages
    sys_txt = "\n".join(x["content"] for x in messages if x["role"] == "system")
    rest = [x for x in messages if x["role"] != "system"]
    if rest and rest[0]["role"] == "user":
        return [{"role": "user", "content": sys_txt + "\n\n" + rest[0]["content"]}] + rest[1:]
    return [{"role": "user", "content": sys_txt}] + rest

def chat(m, messages):
    if MOCK:
        return mock_reply(messages)
    if m.get("backend") == "transformers":
        return hf_chat(m, messages)
    from openai import OpenAI, RateLimitError
    if m["name"] not in _clients:
        _clients[m["name"]] = OpenAI(base_url=m["base_url"], api_key=os.environ[m["key_env"]], max_retries=0)
    for attempt in range(8):
        _pace(m)
        try:
            r = _clients[m["name"]].chat.completions.create(
                model=m["model"], messages=fix_msgs(m, messages), temperature=TEMPERATURE, max_tokens=1024,
                extra_body=m.get("extra"))
            text = r.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            return re.sub(r"<think>.*$", "", text, flags=re.S).strip()
        except RateLimitError as e:
            msg = str(e).lower()
            if "per day" in msg or "(tpd)" in msg or "(rpd)" in msg:
                raise RuntimeError(f"model {m['name']} reached its DAILY limit; rerun tomorrow")
            try:
                wait = float(e.response.headers.get("retry-after", 0))
            except Exception:
                wait = 0
            wait = max(wait, 15 * (attempt + 1)) + random.random()
            print(f"  [{m['name']}] per-minute limit; waiting {wait:.0f}s")
            time.sleep(wait)
        except Exception as e:
            if "called a tool" in str(e):
                messages = messages + [{"role": "system", "content": "You have no tools. Reply with plain text only."}]
                print(f"  [{m['name']}] tried to call a tool; retrying with a plain-text reminder")
                continue
            status_code = getattr(e, "status_code", None)
            if status_code in (400, 401, 402, 403, 404, 422):
                raise RuntimeError(
                    f"{m['name']} request was rejected (HTTP {status_code}); "
                    "check the Hugging Face model access, token permissions, and billing."
                ) from e
            if m["key_env"] == "OLLAMA_API_KEY":
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"  [{m['name']}] {type(e).__name__}: {str(e)[:160]}; retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"local Ollama request failed for {m['model']}: {e}") from e
            wait = 5 * (attempt + 1)
            print(f"  [{m['name']}] {type(e).__name__}: {str(e)[:120]}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"model {m['name']} kept failing; rerun later")

def hf_chat(m, messages):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with _hf_lock:
        if m["model"] not in _hf_models:
            print(f"Loading {m['model']} from Hugging Face to local hardware (first run downloads weights)...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    m["model"], revision=m["revision"]
                )
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    m["model"],
                    revision=m["revision"],
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not load {m['model']} locally: {type(e).__name__}: {e}"
                ) from e
            _hf_models[m["model"]] = (tokenizer, model)

        tokenizer, model = _hf_models[m["model"]]
        try:
            template_kwargs = m.get("template_kwargs", {})
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **m.get("template_kwargs", {}),
            )
        inputs = inputs.to(model.device)
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            raise RuntimeError(
                f"Local generation failed for {m['model']}: {type(e).__name__}: {e}"
            ) from e

        prompt_length = inputs["input_ids"].shape[-1]
        text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return re.sub(r"<think>.*$", "", text, flags=re.S).strip()

def mock_reply(messages):
    sys_msg = messages[0]["content"]
    if sys_msg.startswith("You are playing 20 Questions. The secret"):
        return random.choice(["Yes", "No", "No", "Sometimes", "Maybe yes"])
    if sys_msg.startswith("You are playing 20 Questions. Another"):
        if random.random() < 0.2:
            return "GUESS: " + random.choice(list(SECRET_WORDS) + ["chair", "spoon"])
        return "QUESTION: Is it " + random.choice(["alive", "metal", "edible", "big"]) + "?"
    return random.choice(["Banana", "Chair", "Stapler", "Spoon", "Umbrella", "chair"])

def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).replace(" an ", " ").replace(" a ", " ").strip()

def parse_answer(text):
    w = norm(text).split()
    return w[0].capitalize() if w and w[0] in ("yes", "no", "sometimes") else None

def parse_move(move):
    text = re.sub(r"\(?Turns used:\s*\d+\s*/\s*\d+\.?\)?", " ", move, flags=re.I)
    m = re.search(r"(QUESTION|GUESS)\s*:\s*(.*?)(?=\s*(?:QUESTION|GUESS)\s*:|\n|$)", text, re.I | re.S)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return "QUESTION", text.strip().split("\n")[0].strip()

def valid_move(kind, body):
    words = re.findall(r"[A-Za-z]+", body)
    return len(words) >= (1 if kind == "GUESS" else 3)

def is_guess_question(body, accepted):
    q = norm(body)
    return any(re.fullmatch(rf"(is it|is the object|is the secret object|is your object)( the)? {re.escape(a)}s?", q)
               for a in accepted)

def play(answerer, guesser, secret, game_id=None):
    accepted = {norm(secret)} | {norm(s) for s in SECRET_WORDS[secret]}
    a_msgs = [{"role": "system", "content": ANSWERER_SYS.format(secret=secret)}]
    g_msgs = [{"role": "system", "content": GUESSER_SYS},
              {"role": "user", "content": "I am thinking of an object. Ask your first question."}]
    turns, invalid, bad_moves, won = [], 0, 0, False
    for t in range(1, MAX_Q + 1):
        for _ in range(3):
            move = chat(guesser, g_msgs)
            kind, body = parse_move(move)
            if valid_move(kind, body):
                break
            bad_moves += 1
            g_try = g_msgs + [{"role": "assistant", "content": move},
                              {"role": "user", "content": "Reply with exactly one line: QUESTION: <question> or GUESS: <object>."}]
            move = chat(guesser, g_try)
            kind, body = parse_move(move)
            if valid_move(kind, body):
                break
        g_msgs.append({"role": "assistant", "content": f"{kind}: {body}"})
        if not valid_move(kind, body):
            turns.append({"turn": t, "type": "invalid", "text": move[:200]})
            g_msgs.append({"role": "user", "content": f"That was not a valid move.\n(Turns used: {t}/{MAX_Q})"})
            continue
        if kind == "QUESTION" and is_guess_question(body, accepted):
            kind = "GUESS"
        if kind == "GUESS":
            guess = norm(body)
            correct = guess in accepted or any(a in guess.split() for a in accepted)
            turns.append({"turn": t, "type": "guess", "text": body, "correct": correct})
            if correct:
                won = True
                break
            g_msgs.append({"role": "user", "content": f"No, it is not {body}.\n(Turns used: {t}/{MAX_Q})"})
            continue
        a_msgs.append({"role": "user", "content": body})
        reply = chat(answerer, a_msgs)
        ans = parse_answer(reply)
        if ans is None:
            invalid += 1
            a_msgs.append({"role": "assistant", "content": reply})
            a_msgs.append({"role": "user", "content": "Answer with exactly one word: Yes, No, or Sometimes."})
            reply = chat(answerer, a_msgs)
            ans = parse_answer(reply) or "Sometimes"
        a_msgs.append({"role": "assistant", "content": ans})
        turns.append({"turn": t, "type": "question", "text": body, "answer": ans, "raw": reply})
        g_msgs.append({"role": "user", "content": f"{ans}.\n(Turns used: {t}/{MAX_Q})"})
    return {"game_id": game_id, "answerer": answerer["name"], "guesser": guesser["name"], "secret": secret,
            "won": won, "turns_used": len(turns), "invalid_answers": invalid, "bad_guesser_moves": bad_moves,
            "answer_turns": sum(x["type"] == "question" for x in turns), "turns": turns}

def pick_word(m):
    return norm(chat(m, [{"role": "user", "content": PICK_PROMPT}]).split("\n")[0])

def main():
    random.seed(SEED)
    if ARGS.pilot:
        return run_local_pilot()
    if ARGS.smoke_models:
        return smoke_models()

    if ARGS.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if not os.getenv("OLLAMA_API_KEY"):
        os.environ["OLLAMA_API_KEY"] = "ollama"
    if not MOCK:
        print("All Hugging Face checkpoints run locally; no hosted inference API or HF token is used.")

    games = []
    if os.path.exists(ARGS.results_file):
        with open(ARGS.results_file, encoding="utf-8") as f:
            games = [json.loads(line) for line in f if line.strip()]
    current = {m["name"] for m in MODELS}
    games = [g for g in games if g["answerer"] in current and g["guesser"] in current]
    done = {g["game_id"] for g in games if g.get("game_id")}
    jobs = [
        (a, g, w, repeat, f"{a['name']}|{g['name']}|{w}|{repeat:03d}")
        for a in MODELS for g in MODELS for w in SECRET_WORDS
        for repeat in range(ARGS.repeats)
        if f"{a['name']}|{g['name']}|{w}|{repeat:03d}" not in done
    ]
    print(f"{len(done)} games already saved; playing {len(jobs)} more...")
    failed = 0
    def run(job):
        try:
            prepare_hf_pair(*job[:2])
            return play(*job[:3], game_id=job[4])
        except RuntimeError as e:
            print("  skipped:", e)
            return None
    with ThreadPoolExecutor(WORKERS) as ex, open(ARGS.results_file, "a", encoding="utf-8") as f:
        for gm in ex.map(run, jobs):
            if gm is None:
                failed += 1
                continue
            games.append(gm)
            f.write(json.dumps(gm) + "\n"); f.flush()
    if failed:
        print(f"\n{failed} games could not finish (rate limits). Rerun the script later to finish them.")
        return

    names = [m["name"] for m in MODELS]
    wins = [g for g in games if g["won"]]
    n_ans = sum(g["answer_turns"] for g in games)
    print("\n===== Paste these into the proposal =====")
    print(f"[N_GAMES] = {len(games)}")
    print(f"[WIN_OVERALL] = {len(wins) / len(games):.0%}")
    print(f"[AVG_Q] = {sum(g['turns_used'] for g in wins) / max(len(wins), 1):.1f}")
    print(f"[INVALID_RATE] = {sum(g['invalid_answers'] for g in games) / max(n_ans, 1):.1%}")
    print("\nBad guesser moves (empty or broken questions) per guesser:")
    for g in names:
        print(f"  {g}: {sum(x.get('bad_guesser_moves', 0) for x in games if x['guesser'] == g)}")
    print("\nWin-rate table (rows = answerer, columns = guesser):")
    print("| Answerer / Guesser | " + " | ".join(names) + " |")
    for a in names:
        cells = []
        for g in names:
            sub = [x for x in games if x["answerer"] == a and x["guesser"] == g]
            cells.append(f"{sum(x['won'] for x in sub)}/{len(sub)}")
        print(f"| {a} | " + " | ".join(cells) + " |")
    print(f"\nFull game logs saved to {ARGS.results_file}.")

def smoke_models():
    if not os.getenv("OLLAMA_API_KEY"):
        os.environ["OLLAMA_API_KEY"] = "ollama"
    print("Smoke test uses local model inference only; no hosted APIs are called.")
    for model in MODELS:
        try:
            prepare_hf_pair(model)
            response = chat(model, [{"role": "user", "content": "Reply with exactly the word OK."}])
        except RuntimeError as e:
            print(f"FAILED {model['name']}: {e}")
            return 1
        print(f"OK {model['name']} ({model['model']}): {response[:120]!r}")
    return 0

def run_local_pilot():
    if ARGS.games < 1:
        raise ValueError("--games must be at least 1")
    if not ARGS.mock and not os.getenv("OLLAMA_API_KEY"):
        os.environ["OLLAMA_API_KEY"] = "ollama"

    local_model = next(m for m in MODELS if m["key_env"] == "OLLAMA_API_KEY")
    secrets = list(SECRET_WORDS)
    jobs = []
    for index in range(ARGS.games):
        secret = secrets[index % len(secrets)]
        repeat = index // len(secrets)
        game_id = f"{local_model['model']}|{secret}|{repeat:03d}"
        jobs.append((local_model, local_model, secret, game_id))

    existing = []
    if os.path.exists(ARGS.pilot_file):
        with open(ARGS.pilot_file, encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
    done = {g["game_id"] for g in existing if g.get("game_id")}
    todo = [job for job in jobs if job[3] not in done]
    print(f"Local model: {local_model['model']}")
    print(f"Pilot schedule: {ARGS.games} games across {len(secrets)} targets; "
          f"{len(done)} already saved, {len(todo)} remaining.")
    if ARGS.mock:
        print("Mock mode is enabled; these results are for pipeline validation only.")

    failed = []
    with open(ARGS.pilot_file, "a", encoding="utf-8") as f:
        for completed, job in enumerate(todo, start=1):
            try:
                game = play(*job[:3], game_id=job[3])
            except RuntimeError as e:
                failed.append(job[3])
                print(f"  FAILED {job[3]}: {e}")
                continue
            f.write(json.dumps(game) + "\n")
            f.flush()
            print(f"  Saved {len(done) + completed}/{ARGS.games}: "
                  f"{game['secret']} (repeat {job[3].rsplit('|', 1)[-1]}), "
                  f"{'win' if game['won'] else 'loss'}, {game['turns_used']} turns")

    with open(ARGS.pilot_file, encoding="utf-8") as f:
        games = [json.loads(line) for line in f if line.strip()]
    run_games = [g for g in games if g.get("game_id") in {job[3] for job in jobs}]
    if failed or len({g.get("game_id") for g in run_games}) != ARGS.games:
        print(f"Pilot incomplete: {len(run_games)}/{ARGS.games} saved. "
              "Fix the reported errors and rerun to resume.")
        return 1

    wins = sum(g["won"] for g in run_games)
    print(f"\nPilot complete: {len(run_games)} Llama self-play games; "
          f"{wins}/{len(run_games)} wins ({wins / len(run_games):.1%}).")
    print(f"Results saved to {ARGS.pilot_file}. This is a one-model exploratory pilot.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main() or 0)
