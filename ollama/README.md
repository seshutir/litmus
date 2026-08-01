# LITMUS — Risk Identification for AI Use Cases

**LITMUS** identifies *risks* for a given AI *use case* and grounds each risk in
evidence. Given a `(use case, candidate risk)` pair it decides — with a
supporting/​non‑supporting verdict and a probability — whether the risk is
genuinely associated with that use case, backed by retrieved evidence.

Under the hood LITMUS uses [**FactReasoner**](./FactReasoner) as its reasoning
engine: it turns each risk claim into atomic sub‑claims, retrieves evidence for
them, extracts NLI relations, and runs a probabilistic inference engine
(**merlin**) over the resulting factuality graph. This checkout runs the whole
pipeline against a **local [Ollama](https://ollama.com) server** — no RITS
credentials, no cloud API keys (unless you opt into Google search for evidence).

The entry point is [`example_ai_incidents.py`](./example_ai_incidents.py). For
each `(use case, risk)` pair it:

1. builds a claim — `"<risk> is a risk associated with <use case>"`,
2. atomizes and revises it with the local LLM,
3. attaches evidence contexts from a **local Chroma vectorstore** (default) or
   **Google search**,
4. extracts NLI relations with **SIMBA‑UQ** and runs **merlin** over them, and
5. writes a factuality graph (`<risk><idx>.html` + `<risk><idx>.json`) and
   appends to an aggregated `output/fr_ai_incidents.json`.

This README documents the exact setup used here — the `litmus_neurips` conda
env, the local Ollama/merlin prerequisites, and the command that runs LITMUS
end‑to‑end.

---

## The command that works

Once the env is installed, Ollama is running, and the vectorstore is populated
(all covered below), this scores one use case fully offline:

```bash
python example_ai_incidents.py \
     --context-source chroma \
     --restart-index 0 --limit 1 \
     --top-k 3 --nli-temps 0.3 --nli-samples 3
```

`--nli-temps 0.3 --nli-samples 3 --top-k 3` are the **speed knobs** — on a local
Ollama server the NLI stage runs generations roughly serially and dominates
wall‑clock, so keeping them small is what makes a run finish in minutes rather
than stall. See [§5](#5-speed-knobs-and-the-nli-cost) for why.

---

## 1. Prerequisites

- **macOS (Apple Silicon)** — a prebuilt `merlin` binary (`arm64`) ships in
  [`merlin/build/merlin`](./merlin/build/merlin). The script points
  `MERLIN_PATH` at it automatically.
- **Conda / miniforge** — the env below is named `litmus_neurips` (Python
  **3.11**).
- **Ollama** installed and running with the default model pulled:
  ```bash
  # install: https://ollama.com/download   (or: brew install ollama)
  ollama serve                 # serves on http://localhost:11434
  ollama pull granite4:micro   # the default LLM (granite-4-0-micro)
  ```
  Verify the server is reachable:
  ```bash
  curl -s http://localhost:11434/api/tags | head -c 200
  ```

---

## 2. Installation

These are the exact steps used to stand up `litmus_neurips`:

```bash
# 1) Create / activate the conda env (Python 3.11)
conda create -n litmus_neurips python=3.11 -y
conda activate litmus_neurips

# 2) Install FactReasoner (LITMUS's reasoning engine) EDITABLE from the bundled
#    checkout. Editable (-e) so local edits under FactReasoner/src take effect on
#    the next run without reinstalling. This pins mellea==0.6.0; the 'rits' extra
#    is NOT needed for the Ollama path.
pip install -e ./FactReasoner

# 3) Runtime deps LITMUS needs that FactReasoner does not pull in:
pip install gravis                 # HTML factuality-graph export
pip install "setuptools<81"        # gravis imports pkg_resources (dropped in setuptools >= 81)
pip install sentence-transformers  # embedding function used by Chroma (read + ingest sides)
```

> **Why these three extras?**
> - `gravis` renders the factuality graph to HTML; it still imports the legacy
>   `pkg_resources`, so `setuptools` must be **< 81** or the import fails.
> - `sentence-transformers` provides the `all-MiniLM-L6-v2` embedding function
>   that both `ChromaReader` (read) and `ingest_vectorstore.py` (write) use — it
>   must be present on both sides so the vectors line up.

Everything below assumes the env is active. Equivalently, prefix any command
with `conda run -n litmus_neurips …`.

---

## 3. Populate the evidence vectorstore (do this before the chroma run)

LITMUS grounds each risk in evidence. With `--context-source chroma`, that
evidence comes from a **local Chroma vectorstore** at `./vectorstore_sae_google`
(collection `mydocs`). If the collection is empty, every claim is atomized and
then **skipped** with a warning — there is nothing to reason over.

`ChromaReader` only *reads* an existing collection (using the `all-MiniLM-L6-v2`
embedding function). Use the bundled
[`ingest_vectorstore.py`](./ingest_vectorstore.py) to build/refresh `mydocs`
with the **same** embedding function so reads line up with writes. It ingests
`.txt`, `.md`, and `.pdf` files (recursively), chunks them, and adds them.

Example — ingest the evidence for incident 0:

```bash
python ingest_vectorstore.py \
  --source ./incident_0 \
  --persist-dir ./vectorstore_sae_google \
  --collection mydocs
```

**Verify it worked** (a non‑zero count means the chroma run will find evidence):

```bash
python - <<'PY'
from fact_reasoner.core.retriever import ChromaReader
r = ChromaReader(collection_name="mydocs", persist_directory="./vectorstore_sae_google")
res = r.query("exposure to toxic content", n_results=3)
print("chunks returned:", len(res["documents"][0]))
PY
```

**Ingest options**

| Flag | Default | Description |
|------|---------|-------------|
| `--source DIR` | *(required)* | Directory of evidence files (`.txt`/`.md`/`.pdf`, recursive). |
| `--persist-dir DIR` | `./vectorstore_sae_google` | Chroma persist dir — must match `CHROMA_DIR` in the script. |
| `--collection NAME` | `mydocs` | Collection name — must match `COLLECTION` in the script. |
| `--chunk-size N` | `1000` | Characters per chunk. |
| `--chunk-overlap N` | `150` | Overlap between consecutive chunks. |
| `--reset` | off | Delete the collection before rebuilding. |

> Keep `--persist-dir` / `--collection` identical to `CHROMA_DIR`
> (`./vectorstore_sae_google`) and `COLLECTION` (`mydocs`) in
> `example_ai_incidents.py`. `ingest_vectorstore.py` handles `.txt`/`.md`/`.pdf`
> only — a PDF of the source document is enough to reason over an incident.

---

## 4. Running LITMUS

```bash
conda activate litmus_neurips   # (or prefix commands with: conda run -n litmus_neurips)

# The verified command — one use case, local Chroma evidence, fully offline:
python example_ai_incidents.py \
     --context-source chroma \
     --restart-index 0 --limit 1 \
     --top-k 3 --nli-temps 0.3 --nli-samples 3
```

Expected: atoms extracted + revised, contexts pulled from `mydocs`, SIMBA‑UQ NLI,
merlin runs, `output/fr_ai_incidents.json` appended, and
`<risk><idx>.html` + `<risk><idx>.json` written to the working directory.

### Command‑line options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `None` → `granite4:micro` | Ollama model id/tag. Any pulled tag works (e.g. `llama3.2:latest`) or a catalog alias (`llama3`). |
| `--context-source {chroma,google}` | `chroma` | Evidence source (see [§6](#6-evidence-sources)). |
| `--serper-cache-dir DIR` | `None` | Cache dir for the Google SearchAPI (`google` source only). |
| `--restart-index N` | `0` | First use‑case index to process. |
| `--limit N` | `1` | Number of use cases to process starting at `--restart-index`. |
| `--file-path PATH` | `output/fr_ai_incidents.json` | Aggregated results file. |
| `--top-k N` | `3` | Contexts retrieved per atom. Lower ⇒ fewer NLI pairs ⇒ faster. |
| `--nli-temps T [T ...]` | `0.3 0.7` | SIMBA‑UQ temperature schedule. Fewer ⇒ faster. |
| `--nli-samples N` | `3` | SIMBA‑UQ samples per temperature. Lower ⇒ faster. |
| `--no-progress` | off | Disable the tqdm progress bar on the NLI batch. |

---

## 5. Speed knobs and the NLI cost

Ollama does **not** expose token logprobs, so FactReasoner's default
`nli_method="logprobs"` (RITS / vLLM only) cannot be used. LITMUS uses
`nli_method="simbauq"` — a backend‑agnostic self‑consistency estimator (rouge
similarity + aggregation confidence). It is already wired in; no configuration
needed.

The cost: for **each** `(atom, context)` pair, SIMBA‑UQ runs

```
len(--nli-temps) × --nli-samples   LLM generations
```

With the defaults (`0.3 0.7` × `3`) that is 6 generations per pair; with many
atoms × `--top-k` contexts each, the batch balloons (this is what caused an
earlier run to stall at *"Running throttled batch of 135 requests"*). Because a
local Ollama server serves these roughly serially, the NLI stage dominates
wall‑clock. The three knobs shrink it:

- **`--top-k`** — fewer contexts per atom ⇒ fewer pairs.
- **`--nli-temps`** — a single temperature (`0.3`) halves the generations vs. the
  `0.3 0.7` default.
- **`--nli-samples`** — fewer samples per temperature.

The verified command uses `--top-k 3 --nli-temps 0.3 --nli-samples 3` = **3
generations per pair** — the smallest setting that still gives a stable verdict.

### The supporting / non‑supporting verdict trade‑off

For each atom LITMUS reports a verdict: **S** (supporting) when
`P(true) > P(false)` after merlin inference, else **NS** (non‑supporting). With
very few NLI samples the self‑consistency estimate is noisy: even a genuinely
supporting context can land just under `0.5` and read as **NS**. If a risk you
expect to be supported comes back NS, the usual causes are:

1. the supporting document is not in the vectorstore (e.g. only a PDF was
   ingested, not a `.docx`) — check the ingest count;
2. `--top-k` too low (a single evidence shot), or
3. `--nli-samples` too low (the estimate is under‑sampled).

Raising `--nli-samples` / `--top-k` (and, if needed, adding `0.7` back to
`--nli-temps`) trades speed for a more stable verdict.

---

## 6. Evidence sources

- **`chroma` (default, offline):** reads evidence from `./vectorstore_sae_google`
  (collection `mydocs`). Self‑contained — no network or API keys. Populate it
  first ([§3](#3-populate-the-evidence-vectorstore-do-this-before-the-chroma-run)).
- **`google`:** re‑retrieves evidence via Google SearchAPI at build time.
  Requires a key:
  ```bash
  export SERPER_API_KEY=<your-key>
  python example_ai_incidents.py --context-source google --limit 1
  ```

---

## 7. Inputs and outputs

**Inputs** (already present):
- `ai_use_cases/rephrased_incidents.json` — the use cases.
- `ai_use_cases/rephrased_incidents_risk_subdomain.json` — the paired candidate risks.

**Outputs:**
- `output/fr_ai_incidents.json` — aggregated risk results. Per use case, for each
  target atom it records parallel lists: `risks` (supporting context text),
  `summaries` (each context's synthetic summary), `probabilities`, and `type`.
- `<risk><idx>.html` — interactive factuality graph (gravis).
- `<risk><idx>.json` — full pipeline dump for that use case, including `results`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'fact_reasoner'` | `pip install -e ./FactReasoner` in the active env. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | `pip install "setuptools<81"`. |
| `sentence_transformers ... is not installed` | `pip install sentence-transformers`. |
| `No local Chroma evidence found ... Skipping` | The `mydocs` collection is empty — populate it ([§3](#3-populate-the-evidence-vectorstore-do-this-before-the-chroma-run)) or use `--context-source google`. |
| NLI batch stalls / huge request count | Lower `--top-k`, `--nli-samples`, and use a single `--nli-temps 0.3` ([§5](#5-speed-knobs-and-the-nli-cost)). |
| NS verdict despite supporting evidence | Confirm the doc is ingested; raise `--nli-samples` / `--top-k` ([§5](#the-supporting--non-supporting-verdict-trade-off)). |
| Connection refused to `localhost:11434` | Start Ollama: `ollama serve`, then `ollama pull granite4:micro`. |
| `--context-source google requires the SERPER_API_KEY` | `export SERPER_API_KEY=<your-key>`. |
