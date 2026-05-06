# Litmus: AI Risk Assessment Framework

Litmus is a framework for assessing and analyzing AI-related risks across different use cases. It leverages fact reasoning, natural language inference, and knowledge graphs to systematically evaluate how specific risks manifest in AI systems.

## Overview

This project combines multiple components to:
- Extract factual claims from AI use cases
- Retrieve relevant evidence from document stores
- Build knowledge graphs representing relationships between risks and evidence
- Score and visualize risk factors associated with different AI applications

## Quick Start

### Prerequisites

1. **Conda Environment**: Set up the `fr_mellea` conda environment (see [Environment Setup](#environment-setup))

2. **LLM Backend**: You must configure an LLM backend independently. The framework supports multiple backends through the `mellea` library:
   - Llama (via RITS)
   - Other backends supported by mellea

3. **API Keys**: Set up required environment variables:
   ```bash
   export SERPER_API_KEY="your-serper-api-key"
   ```
   The `SERPER_API_KEY` is used by the `ContextRetriever` for Google search results when gathering literature and evidence.

### Environment Setup

Create and activate the conda environment:

```bash
# Create environment from the pre-configured fr_mellea environment
conda activate fr_mellea

# Or manually install from requirements.txt
pip install -r requirements.txt
```

### Running the Benchmark

```bash
python benchmark.py
```

The script will:
1. Load use cases from `ai_use_cases/rephrased_google_use_cases.json`
2. For each risk-usecase pair, analyze the connection
3. Generate visualizations and JSON outputs in the `output/` directory

## Project Structure

```
litmus/
├── benchmark.py                          # Main benchmark script
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
├── FactReasoner/                         # Fact reasoning engine
├── merlin/                               # Knowledge graph builder
├── vectorstore_sae_google/               # Chroma vector store for document embeddings
├── ai_use_cases/                         # Use case definitions
│   └── rephrased_google_use_cases.json   # JSON file with AI use cases
└── output/                               # Generated outputs (risk graphs and JSON)
```

## Key Components

### FactReasoner
A reasoning engine that:
- Extracts atomic facts from claims using `Atomizer`
- Revises and validates atoms using `Reviser`
- Retrieves supporting context using `ContextRetriever`
- Summarizes context using `ContextSummarizer`
- Performs natural language inference using `NLIExtractor`

### Risk Definitions
The framework defines risks in a hierarchical structure (currently focused on risk "1.1: Unfair discrimination and misrepresentation"). Risk-usecase mappings are defined in `benchmark.py` and can be extended.

### Output

For each risk-usecase analysis, the system generates:

1. **HTML Visualization** (`{RISK}{INDEX}.html`): 
   - Interactive directed graph showing relationships between atoms (claims) and contexts (evidence)
   - Edges are labeled with probability scores and relationship types

2. **JSON Output** (`{RISK}{INDEX}.json`):
   - Structured reasoning graph with:
     - All extracted atoms and their relationships
     - Retrieved contexts and supporting evidence
     - Scoring results and marginal probabilities
     - Edge types and confidence scores

3. **Analysis Summary** (`fr_{RISK}_sources_rephrased_google_use_cases.json`):
   - Aggregated results linking risks to specific evidence
   - Probability scores for each risk-evidence connection
   - Relationship types between atoms and contexts

## Configuration

### LLM Backend Configuration

The default configuration uses Llama 3.3 70B Instruct via RITS backend:

```python
model_id = "llama-3.3-70b-instruct"
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT,
    model_options={
        ModelOption.MAX_NEW_TOKENS: 4096,
        ModelOption.TEMPERATURE: 0
    }
)
```

To use a different backend, modify the `backend` initialization in `benchmark.py`.

### Vector Store Configuration

Documents are embedded and stored using Chroma:
- **Directory**: `./vectorstore_sae_google`
- **Collection**: `mydocs`
- **Embedding Model**: `all-MiniLM-L6-v2`

### Risk and Use Case Mapping

Edit the `risks_dict` and `label_indices_dict` in `benchmark.py` to:
- Add new risk categories
- Map use case indices to specific risks
- Change risk descriptions

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- **LLM & NLP**: transformers, torch, torchvision
- **Reasoning**: mellea, mellea-ibm
- **Vector Database**: chromadb
- **Graph Processing**: gravis, networkx
- **Utilities**: pydantic, pyyaml, requests

## Environment Variables

Required:
- `SERPER_API_KEY`: API key for Serper (Google search results)
- `FACT_REASONER`: Path to FactReasoner directory (set to `./FactReasoner/` in benchmark.py)
- `MERLIN_PATH`: Path to Merlin executable (set to `./merlin/build/merlin` in benchmark.py)

## Data Flow

```
Use Case Input
    ↓
FactReasoner Pipeline
    ├── Atomize: Extract factual claims
    ├── Retrieve: Find supporting evidence
    ├── Build Graph: Connect atoms and contexts
    └── Score: Calculate confidence scores
    ↓
Output Generation
    ├── Fact Graph JSON
    ├── Interactive Visualization (HTML)
    └── Risk-Evidence Summary
```

## Extending the Framework

### Adding New Risks

1. Add to `risks_dict` with risk ID and description
2. Add index mappings to `label_indices_dict`
3. Run `benchmark.py` to generate analysis

### Adding New Use Cases

1. Add entries to `ai_use_cases/rephrased_google_use_cases.json`
2. Update `label_indices_dict` to include new use case indices
3. Rerun analysis

### Customizing the LLM Backend

Replace the `RITSBackend` initialization with your preferred backend, ensuring it's compatible with the `mellea` library API.

## Troubleshooting

### Missing SERPER_API_KEY
```
Error: SERPER_API_KEY not found
```
Set the environment variable:
```bash
export SERPER_API_KEY="your-key"
```

### Backend Connection Issues
Ensure your LLM backend is properly configured and accessible. Check backend-specific configuration in the respective library documentation.

### Vector Store Not Found
Ensure the `vectorstore_sae_google/` directory exists with pre-populated documents. If missing, the system will attempt to create it during the first run.

## Output Example

A typical output structure:
```json
{
  "input": "Risk description and use case",
  "output": "Same as input",
  "atoms": [
    {
      "id": "a0",
      "text": "Extracted claim",
      "contexts": ["a0_ctx_0", "a0_ctx_1"]
    }
  ],
  "contexts": [
    {
      "id": "a0_ctx_0",
      "title": "PDF Evidence",
      "text": "Supporting document text",
      "snippet": "Relevant excerpt..."
    }
  ],
  "results": [/* scoring results */]
}
```

