import json
import os
import re
import sys

import click
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.lm_inference import OpenAIModel, OpenRouterModel

PROMPT = (
    "Answer the following question about a Game Boy-era Pokémon game using only what you "
    "already know. Do not search, browse, or look anything up. Reply with a bare answer of "
    "1-2 sentences and nothing else.\n\n"
    "Game: {game}\n"
    "Question: {question}"
)


def clean(text):
    if text is None:
        return ""
    if text.startswith("Reasoning:") and "\nResponse: " in text:
        text = text.split("\nResponse: ", 1)[1]
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return re.sub(r"\s+", " ", text)


@click.command()
@click.option("--input_csv", default="results/contamination/pokemon_contamination_benchmark.csv")
@click.option("--model", default="google/gemini-3.6-flash")
@click.option("--out_dir", default="results/contamination")
@click.option("--limit", default=None, type=int)
@click.option("--batch", default=20, type=int)
@click.option("--max_new_tokens", default=2048, type=int)
@click.option("--provider", default="openrouter", type=click.Choice(["openrouter", "openai"]))
@click.option("--temperature", default="0.0")
def main(input_csv, model, out_dir, limit, batch, max_new_tokens, provider, temperature):
    temperature = None if temperature == "none" else float(temperature)
    df = pd.read_csv(input_csv)
    if limit is not None:
        df = df.head(limit)
    tag = model.split("/")[-1]
    raw_path = os.path.join(out_dir, f"answers_{tag}.jsonl")
    done = {}
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("raw") is not None:
                    done[rec["row"]] = rec
    lm = OpenRouterModel(model) if provider == "openrouter" else OpenAIModel(model)
    todo = [i for i in range(len(df)) if i not in done]
    print(f"{len(done)} cached, {len(todo)} to query", flush=True)
    for start in range(0, len(todo), batch):
        rows = todo[start:start + batch]
        prompts = [PROMPT.format(game=df.loc[i, "game"], question=df.loc[i, "question"]) for i in rows]
        try:
            outs = lm.infer(prompts, max_new_tokens=max_new_tokens, temperature=temperature)
            pairs = list(zip(rows, outs["output"], outs["meta"]["output_tokens"]))
        except Exception as e:
            print(f"batch failed ({e}); retrying rows individually", flush=True)
            pairs = []
            for i, p in zip(rows, prompts):
                try:
                    o = lm.infer(p, max_new_tokens=max_new_tokens, temperature=temperature)
                    pairs.append((i, o["output"], o["meta"]["output_tokens"]))
                except Exception as e2:
                    print(f"row {i} failed: {e2}", flush=True)
                    pairs.append((i, None, None))
        with open(raw_path, "a") as f:
            for i, raw, n_out in pairs:
                rec = {"row": i, "question_id": int(df.loc[i, "question_id"]), "raw": raw,
                       "output_tokens": n_out}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if raw is not None:
                    done[i] = rec
        print(f"{len(done)}/{len(df)} answered", flush=True)
    col = f"actual_answer_{tag}"
    out = df[["game", "question_id", "question", "expected_answer"]].copy()
    out[col] = [clean(done[i]["raw"]) if i in done else "" for i in range(len(df))]
    out_path = os.path.join(out_dir, f"pokemon_contamination_{tag}.csv")
    out.to_csv(out_path, index=False)
    n_empty = int((out[col] == "").sum())
    print(f"wrote {out_path}: {len(out) - n_empty}/{len(out)} non-empty", flush=True)


if __name__ == "__main__":
    main()
