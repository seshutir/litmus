from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS
import os
from pathlib import Path
import sys
import json
import gravis as gv
import json

os.environ["FACT_REASONER"] = "/Users/seshu/Documents/2026/risks_benchmark_v2/FactReasoner/"
os.environ["MERLIN_PATH"] = "/Users/seshu/Documents/2025/risks_benchmark/risk_scrapping/src/FactReasoner/merlin/build/merlin"
fr_parent_dir = os.getenv("FACT_REASONER")

sys.path.insert(0, fr_parent_dir)


from fact_reasoner import FactReasoner
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.summarizer import ContextSummarizer
from src.fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.retriever import ChromaReader
from fact_reasoner.baselines.factscore import FactScore


from src.fact_reasoner.core.query_builder import QueryBuilder

MERLIN_PATH = os.getenv("MERLIN_PATH")

import json
import os
from typing import Any, Dict, List


def append_dict_to_file(new_dict: Dict[str, Any], filename: str) -> None:
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



def fr_check(INPUT, OUTPUT, file_path, usecase_index, RISK, USECASE): 
    #############################################
    # 2) Claim
    #############################################

    input = INPUT 
    output = OUTPUT 
    print(f"\n🔍 Claim: {output}")

    #############################################
    # 4) Initialize FactReasoner
    #############################################

    model_id = "llama-3.3-70b-instruct"

    backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096, ModelOption.TEMPERATURE: 0},)

    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)

    CHROMA_DIR = "./vectorstore_sae_google"
    COLLECTION = "mydocs"

    context_retriever = ContextRetriever(
        service_type="google",
        collection_name="mydocs",
        persist_dir=CHROMA_DIR, 
        top_k=5
    )

    reader = ChromaReader(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_model="all-MiniLM-L6-v2"
    )


    context_summarizer = ContextSummarizer(backend)
    nli_extractor = NLIExtractor(backend)

    # Step 1: Extract atoms first
    result = atom_extractor.run(output)
    atoms = [
        {
            "id": f"a{i}",
            "text": atom_text,
            "original": atom_text,
            "label": None,
            "contexts": []  
        }
        for i, atom_text in enumerate(result.values())
    ]

    # Step 2: Initialize top-level context list
    contexts_atom_search = []

    # Step 3: Query the reader for each atom
    for atom in atoms:
        atom_claim = atom["text"]
        results = reader.query(atom_claim, n_results=10)
        retrieved_docs = results['documents'][0]

        for i, doc in enumerate(retrieved_docs):
            doc_text = doc.page_content if hasattr(doc, "page_content") else str(doc)

            ctx_id = f"{atom['id']}_ctx_{i}"
            contexts_atom_search.append({
                "id": ctx_id,
                "title": "PDF Evidence",
                "text": doc_text,
                "snippet": doc_text[:200],
                "link": ""
            })
            atom["contexts"].append(ctx_id)

    # Step 4: Build final instance JSON
    instance = {
        "input": INPUT,
        "output": OUTPUT,
        "topic": "",  
        "atoms": atoms,
        "contexts": contexts_atom_search
    }


    pipeline = FactReasoner(
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        nli_extractor=nli_extractor,
        merlin_path=MERLIN_PATH
    )

    pipeline.from_dict_with_contexts(instance)

    pipeline.build(
        has_atoms=True,
        has_contexts=False,
        revise_atoms=True,
        remove_duplicates=True,
        contexts_per_atom_only=False,
        rel_atom_context=True, 
        rel_context_context=False,
        summarize_contexts=False
    )

    # run reasoning graph
    results, marginals = pipeline.score()

    for key, atom in pipeline.atoms.items():
        print()
        print(f"Atom {key}: {atom.text}")
        print()
        for context in atom.contexts:
            print(f"  Context {context}: {pipeline.contexts[context].text[:]}")


    new_item = {usecase_index:{"intent": USECASE}}

    graph_response = pipeline.fact_graph


    for edge in graph_response.edges:
        value = new_item.get(pipeline.atoms[edge.target].get_text())
        if value is None:
            new_item[usecase_index][pipeline.atoms[edge.target].get_text()] = {"risks":[], "probabilities":[], "type": []}
        new_item[usecase_index][pipeline.atoms[edge.target].get_text()]["risks"].append(pipeline.contexts[edge.source].get_text())
        new_item[usecase_index][pipeline.atoms[edge.target].get_text()]["probabilities"].append(edge.probability)
        new_item[usecase_index][pipeline.atoms[edge.target].get_text()]["type"].append(edge.type)

    
    # Append the new item
    append_dict_to_file(new_item, file_path)

    graph_response = pipeline.fact_graph
    g = graph_response.as_digraph()
    fig = gv.d3(g, show_edge_label=True, edge_label_data_source='label', edge_curvature=0.2)

    filename = RISK + str(usecase_index) 
    fig.export_html(filename + ".html")

    output_file = os.path.join(filename+".json")
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)


file_path = "output/fr_ai_incidents.json"

with open("./ai_use_cases/rephrased_incidents.json") as f:
    use_cases = json.load(f)

with open("./ai_use_cases/rephrased_incidents_risk_subdomain.json") as f:
    risks = json.load(f)

RESTART_INDEX = 277
for usecase_index, use_case in enumerate(use_cases[RESTART_INDEX:RESTART_INDEX+23]):
    usecase_index1 = usecase_index + RESTART_INDEX
    print(usecase_index1)
    USECASE = use_case 
    RISK = risks[usecase_index1]
    INPUT = RISK + " is a risk associated with " +  USECASE
    OUTPUT = RISK + " is a risk associated with " +  USECASE
    fr_check(INPUT, OUTPUT, file_path, usecase_index1, RISK, USECASE)


# RESTART_INDEX = 100
# for usecase_index, use_case in enumerate(use_cases[RESTART_INDEX:RESTART_INDEX+1]):
#     usecase_index1 = usecase_index + RESTART_INDEX
#     print(usecase_index1)
#     USECASE = "Machine learning to integrate renewable energy sources like solar and wind power into the grid by forecasting variable weather patterns and optimizing energy storage systems, ensuring grid stability and maximizing renewable energy utilization." 
#     RISK = "Hallucination"
#     INPUT = RISK + " is a risk associated with " +  USECASE
#     OUTPUT = RISK + " is a risk associated with " +  USECASE
#     fr_check(INPUT, OUTPUT, file_path, usecase_index1, RISK, USECASE)