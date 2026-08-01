

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import gravis as gv

from mellea.backends import ModelOption

FACT_REASONER_DIR = "./FactReasoner"
os.environ.setdefault("FACT_REASONER", FACT_REASONER_DIR)
_src = os.path.join(FACT_REASONER_DIR, "src")
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)

# Local merlin build (arm64 Mach-O) shipped in this workspace.
DEFAULT_MERLIN_PATH = "./merlin/build/merlin"
MERLIN_PATH = os.environ.setdefault("MERLIN_PATH", DEFAULT_MERLIN_PATH)

from fact_reasoner import FactReasoner, build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, ChromaReader, SourceRetriever
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor

# Local Chroma vectorstore holding the PDF evidence.
CHROMA_DIR = "./vectorstore_sae_google"
COLLECTION = "mydocs"


def append_dict_to_file(new_dict: Dict[str, Any], filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON root must be a list, got {type(data).__name__}")
        except json.JSONDecodeError as e:
            print(f"Warning: {filename} is empty or not valid JSON. Starting new list. ({e})")
            data = []
    else:
        data = []

    data.append(new_dict)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Appended new dict to '{filename}'. New length: {len(data)}")


def build_components(
    model_id,
    context_source,
    serper_cache_dir,
    top_k=3,
    nli_temps=None,
    nli_samples=3,
    show_progress=True,
):


    if nli_temps is None:
        nli_temps = [0.3, 0.7]

    # Ollama backend. model_id=None -> shared default (granite-4-0-micro ->
    # granite4:micro). TEMPERATURE:0 preserves the original deterministic intent;
    # build_backend already applies MAX_NEW_TOKENS=4096 by default.
    backend = build_backend(
        "ollama",
        model_id=model_id,
        model_options={ModelOption.MAX_NEW_TOKENS: 4096, ModelOption.TEMPERATURE: 0},
    )

    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    context_summarizer = ContextSummarizer(backend)
    # Ollama has no logprobs -> simbauq self-consistency (rouge/aggregation
    # default). Fewer temps/samples = fewer generations per NLI pair.
    nli_extractor = NLIExtractor(
        backend,
        nli_method="simbauq",
        simbauq_temperatures=nli_temps,
        simbauq_n_per_temp=nli_samples,
        show_progress=show_progress,
    )

    # ContextRetriever now wraps a SourceRetriever + summarizer (new API).
    if context_source == "google":
        if not os.getenv("SERPER_API_KEY"):
            raise SystemExit(
                "--context-source google requires the SERPER_API_KEY environment "
                "variable to be set."
            )
        source = SourceRetriever(
            service_type="google",
            top_k=top_k,
            cache_dir=serper_cache_dir,
            fetch_text=True,
            num_workers=4,
        )
    else:  # "chroma"
        source = SourceRetriever(
            service_type="chromadb",
            collection_name=COLLECTION,
            persist_dir=CHROMA_DIR,
            top_k=top_k,
            num_workers=4,
        )
    context_retriever = ContextRetriever(
        retriever=source,
        context_summarizer=context_summarizer,
        num_workers=4,
    )

    reader = ChromaReader(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )

    return {
        "backend": backend,
        "atom_extractor": atom_extractor,
        "atom_reviser": atom_reviser,
        "context_summarizer": context_summarizer,
        "nli_extractor": nli_extractor,
        "context_retriever": context_retriever,
        "reader": reader,
        "top_k": top_k,
    }


async def fr_check(comp, context_source, INPUT, OUTPUT, file_path, usecase_index, RISK, USECASE):
    input = INPUT
    output = OUTPUT
    print(f"\n🔍 Claim: {output}")

    atom_extractor = comp["atom_extractor"]
    reader = comp["reader"]
    top_k = comp["top_k"]

    # Step 1: Extract atoms first.
    result = atom_extractor.run(output)
    atoms = [
        {
            "id": f"a{i}",
            "text": atom_text,
            "original": atom_text,
            "label": None,
            "contexts": [],
        }
        for i, atom_text in enumerate(result.values())
    ]

    # For the local Chroma path we pre-attach evidence and tell build() the
    # contexts already exist. For the google path we let build() retrieve.
    use_preattached = context_source == "chroma"

    contexts_atom_search = []
    if use_preattached:
        for atom in atoms:
            atom_claim = atom["text"]
            results = reader.query(atom_claim, n_results=top_k)
            retrieved_docs = results["documents"][0] if results.get("documents") else []

            for i, doc in enumerate(retrieved_docs):
                doc_text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                ctx_id = f"{atom['id']}_ctx_{i}"
                contexts_atom_search.append(
                    {
                        "id": ctx_id,
                        "title": "PDF Evidence",
                        "text": doc_text,
                        "snippet": doc_text[:200],
                        "link": "",
                    }
                )
                atom["contexts"].append(ctx_id)

        if not contexts_atom_search:
            print(
                f"⚠️  No local Chroma evidence found for use case {usecase_index}. "
                f"The '{COLLECTION}' collection in {CHROMA_DIR} appears empty — "
                "populate it or use --context-source google. Skipping."
            )
            return

    instance = {
        "input": INPUT,
        "output": OUTPUT,
        "topic": "",
        "atoms": atoms,
        "contexts": contexts_atom_search,
    }

    pipeline = FactReasoner(
        atom_extractor=comp["atom_extractor"],
        atom_reviser=comp["atom_reviser"],
        context_retriever=comp["context_retriever"],
        context_summarizer=comp["context_summarizer"],
        nli_extractor=comp["nli_extractor"],
        merlin_path=MERLIN_PATH,
    )

    pipeline.from_dict_with_contexts(instance)

    await pipeline.build(
        has_atoms=True,
        has_contexts=use_preattached,   # True: keep local evidence; False: retrieve
        revise_atoms=True,
        remove_duplicates=True,
        contexts_per_atom_only=False,
        rel_atom_context=True,
        rel_context_context=False,
        summarize_contexts=True,
    )

    # run reasoning graph
    results, marginals = pipeline.score()

    for key, atom in pipeline.atoms.items():
        print()
        print(f"Atom {key}: {atom.text}")
        print()
        for context in atom.contexts:
            print(f"  Context {context}: {pipeline.contexts[context].text[:]}")

    new_item = {usecase_index: {"intent": USECASE}}
    graph_response = pipeline.fact_graph

    for edge in graph_response.edges:
        value = new_item.get(pipeline.atoms[edge.target].get_text())
        if value is None:
            new_item[usecase_index][pipeline.atoms[edge.target].get_text()] = {
                "risks": [],
                "summaries": [],
                "probabilities": [],
                "type": [],
            }
        target_text = pipeline.atoms[edge.target].get_text()
        source_context = pipeline.contexts[edge.source]
        new_item[usecase_index][target_text]["risks"].append(
            source_context.get_text()
        )
        # Context summary (synthetic_summary) — same value written per-context in
        # the per-incident <RISK><idx>.json. Empty string when no summary.
        new_item[usecase_index][target_text]["summaries"].append(
            source_context.get_summary()
        )
        new_item[usecase_index][target_text]["probabilities"].append(edge.probability)
        new_item[usecase_index][target_text]["type"].append(edge.type)

    append_dict_to_file(new_item, file_path)

    graph_response = pipeline.fact_graph
    g = graph_response.as_digraph()
    fig = gv.d3(g, show_edge_label=True, edge_label_data_source="label", edge_curvature=0.2)

    filename = RISK + str(usecase_index)
    fig.export_html(filename + ".html")

    output_file = os.path.join(filename + ".json")
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FactReasoner over AI-incident use cases on local Ollama."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model id/tag. Default: build_backend default (granite4:micro).",
    )
    parser.add_argument(
        "--context-source",
        choices=["chroma", "google"],
        default="chroma",
        help="Evidence source: 'chroma' (local vectorstore, offline; default) or "
        "'google' (Google SearchAPI, needs SERPER_API_KEY).",
    )
    parser.add_argument(
        "--serper-cache-dir",
        default=None,
        help="Cache dir for the Google SearchAPI (only used with --context-source google).",
    )
    parser.add_argument("--restart-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of use cases to process starting at --restart-index.",
    )
    parser.add_argument("--file-path", default="output/fr_ai_incidents.json")
    # --- Speed knobs (see build_components) ---
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Contexts retrieved per atom. Lower = fewer NLI pairs = faster. Default 3.",
    )
    parser.add_argument(
        "--nli-temps",
        type=float,
        nargs="+",
        default=[0.3, 0.7],
        help="SIMBA-UQ temperature schedule. Fewer = faster. Default: 0.3 0.7.",
    )
    parser.add_argument(
        "--nli-samples",
        type=int,
        default=3,
        help="SIMBA-UQ samples per temperature. Lower = faster. Default 3. "
        "(Generations per NLI pair = len(nli-temps) * nli-samples.)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar on the NLI batch.",
    )
    args = parser.parse_args()

    comp = build_components(
        args.model,
        args.context_source,
        args.serper_cache_dir,
        top_k=args.top_k,
        nli_temps=args.nli_temps,
        nli_samples=args.nli_samples,
        show_progress=not args.no_progress,
    )

    with open("./ai_use_cases/rephrased_incidents.json") as f:
        use_cases = json.load(f)
    with open("./ai_use_cases/rephrased_incidents_risk_subdomain.json") as f:
        risks = json.load(f)

    start = args.restart_index
    for offset, use_case in enumerate(use_cases[start : start + args.limit]):
        usecase_index = offset + start
        print(usecase_index)
        USECASE = use_case
        RISK = risks[usecase_index]
        INPUT = RISK + " is a risk associated with " + USECASE
        OUTPUT = RISK + " is a risk associated with " + USECASE
        await fr_check(
            comp, args.context_source, INPUT, OUTPUT, args.file_path, usecase_index, RISK, USECASE
        )


if __name__ == "__main__":
    asyncio.run(main())
