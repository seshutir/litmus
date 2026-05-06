# Benchmark.py Documentation

## Overview

`benchmark.py` is the main entry point for the Litmus risk assessment framework. It orchestrates the entire pipeline for analyzing AI risks across different use cases by:

1. Loading pre-defined use cases and risk categories
2. Mapping risks to specific use cases
3. Running the FactReasoner pipeline for each risk-usecase pair
4. Generating visualizations and structured outputs

## Main Execution Flow

```
benchmark.py
├── Load Risk Definitions (risks_dict)
├── Load Risk-UseCase Mappings (label_indices_dict)
├── Load AI Use Cases (ai_use_cases/rephrased_google_use_cases.json)
└── For each risk in risks_dict:
    └── For each use case index in label_indices_dict[risk]:
        └── Call fr_check() to analyze the risk-usecase pair
```

## Core Functions

### `append_dict_to_file(new_dict: Dict[str, Any], filename: str) -> None`

**Purpose**: Append analysis results to a JSON file in list format.

**Parameters**:
- `new_dict`: Dictionary containing the analysis results to append
- `filename`: Path to the JSON file

**Behavior**:
- If file exists and is valid JSON, loads existing list
- Appends new dictionary to the list
- Writes back to file with formatting (indent=4)
- Prints confirmation with new list length
- Handles edge cases: missing files, invalid JSON, empty files

**Error Handling**:
- Catches `json.JSONDecodeError` and starts with empty list
- Validates that JSON root is a list (not dict or other types)

**Example Usage**:
```python
result = {"1.1": {"intent": "Use case text", "risk_data": {...}}}
append_dict_to_file(result, "output/results.json")
```

### `fr_check(INPUT, OUTPUT, file_path, usecase_index, RISK, USECASE) -> None`

**Purpose**: Main risk analysis function. Orchestrates the complete FactReasoner pipeline for a single risk-usecase pair.

**Parameters**:
- `INPUT` (str): Description of the risk and use case (typically: "{RISK} is a risk associated with {USECASE}")
- `OUTPUT` (str): The claim to be analyzed (same as INPUT in current implementation)
- `file_path` (str): Path to JSON file for storing results
- `usecase_index` (int): Index of the use case being analyzed
- `RISK` (str): Risk description (from risks_dict)
- `USECASE` (str): Use case description (from ai_use_cases JSON)

**Process**:

1. **Initialize Backend** (lines 69-72):
   - Sets up Llama 3.3 70B Instruct via RITS backend
   - Configures max tokens (4096) and temperature (0 for deterministic output)

2. **Initialize Components** (lines 74-95):
   - **Atomizer**: Extracts atomic facts from claims
   - **Reviser**: Validates and revises extracted atoms
   - **ContextRetriever**: Retrieves relevant evidence (Google search)
   - **ChromaReader**: Queries vector store for document embeddings
   - **ContextSummarizer**: Summarizes retrieved contexts
   - **NLIExtractor**: Performs natural language inference

3. **Extract Atoms** (lines 98-108):
   - Runs atomizer on OUTPUT claim
   - Converts atoms to structured format with IDs
   - Initializes empty contexts list for each atom

4. **Retrieve Context** (lines 114-130):
   - For each atom, queries ChromaReader with top 10 results
   - Creates context objects with:
     - Unique ID: `{atom_id}_ctx_{index}`
     - Title: "PDF Evidence"
     - Text: Full document content
     - Snippet: First 200 characters
     - Link: Empty string (for extensibility)

5. **Build Instance** (lines 132-139):
   - Creates structured JSON with:
     - Input/output text
     - All atoms with their contexts
     - Context metadata

6. **Initialize Pipeline** (lines 142-149):
   - Creates FactReasoner with all components
   - Passes instance data to pipeline

7. **Build Reasoning Graph** (lines 151-162):
   - Calls `pipeline.build()` with configuration:
     - `has_atoms=True`: Include extracted atoms
     - `has_contexts=False`: Contexts handled separately
     - `revise_atoms=True`: Validate extracted atoms
     - `remove_duplicates=True`: Remove redundant atoms
     - `contexts_per_atom_only=False`: Cross-atom contexts allowed
     - `rel_atom_context=True`: Build atom-context relationships
     - `rel_context_context=False`: Skip context-context links
     - `summarize_contexts=True`: Summarize evidence

8. **Score Results** (line 165):
   - Runs reasoning engine to compute scores
   - Returns results and marginal probabilities

9. **Print Atom-Context Relationships** (lines 167-172):
   - Debugging output showing extracted atoms and their evidence

10. **Build Risk-Evidence Map** (lines 175-186):
    - Creates dictionary structure mapping risks to evidence
    - For each graph edge:
      - Links target atom (risk) to source context (evidence)
      - Records probability score
      - Records relationship type
    - Structure: `{usecase_index: {intent: ..., risk_text: {risks: [...], probabilities: [...], type: [...]}}}`

11. **Generate Outputs** (lines 190-203):
    - **JSON Results** (`append_dict_to_file`): Append risk-evidence map
    - **HTML Visualization** (lines 192-197):
      - Converts fact graph to directed graph
      - Creates interactive D3 visualization with edge labels
      - Exports to `{RISK}{INDEX}.html`
    - **JSON Export** (lines 199-203):
      - Serializes complete pipeline to JSON
      - Includes scoring results
      - Exports to `{RISK}{INDEX}.json`

**Output Files Generated**:

For each call with `RISK="1.1 Unfair discrimination..."` and `usecase_index=0`:

1. **1.1 Unfair discrimination...0.html**
   - Interactive graph visualization
   - Nodes represent atoms (claims) and contexts (evidence)
   - Edges show relationships with probability labels

2. **1.1 Unfair discrimination...0.json**
   - Complete reasoning graph structure
   - All atoms with extracted text
   - All contexts with evidence
   - Scoring results and probabilities

3. **output/fr_1.1_sources_rephrased_google_use_cases.json** (appended)
   - Aggregated results
   - Maps risks to evidence with probabilities

## Configuration Dictionaries

### `risks_dict`

**Purpose**: Maps risk IDs to human-readable descriptions.

**Current State**: Only risk "1.1" is active. Other risks are commented out.

**Structure**:
```python
risks_dict = {
    "1.1": "Unfair discrimination and misrepresentation",
    # "1.2": "Exposure to toxic content",
    # ... more risks
}
```

**To Add New Risk**:
1. Uncomment or add new entry
2. Ensure corresponding entry in `label_indices_dict`
3. Rerun benchmark.py

### `label_indices_dict`

**Purpose**: Maps risk IDs to use case indices for analysis.

**Structure**:
```python
label_indices_dict = {
    "1.1": [0, 1],  # Analyze risk 1.1 for use cases 0 and 1
    # Other risks...
}
```

**Current Active Mappings**:
- Risk "1.1": Use cases 0, 1

**To Modify**:
1. Update the list of indices for a risk
2. Ensure indices exist in the loaded use cases JSON
3. Rerun benchmark.py

## Data Sources

### Use Cases
- **File**: `ai_use_cases/rephrased_google_use_cases.json`
- **Format**: JSON array of use case strings
- **Usage**: Loaded once at startup, indexed by position

**Example**:
```json
[
  "First use case description",
  "Second use case description",
  ...
]
```

### Vector Store
- **Path**: `./vectorstore_sae_google/`
- **Type**: Chroma persistent directory
- **Collection**: `mydocs`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Contents**: Pre-indexed documents for context retrieval

## Environment Variables

### Required
- `FACT_REASONER`: Path to FactReasoner directory (set to `./FactReasoner/`)
- `MERLIN_PATH`: Path to Merlin executable (set to `./merlin/build/merlin`)
- `SERPER_API_KEY`: API key for Google search (used by ContextRetriever)

### Auto-Set in Script
- `FACT_REASONER`: Automatically set to `./FactReasoner/`
- `MERLIN_PATH`: Automatically set to `./merlin/build/merlin`

## Key Design Decisions

### Why Separate INPUT and OUTPUT?
Currently, `INPUT` and `OUTPUT` are identical. This provides flexibility for future workflows where the output might be a processed version of the input.

### Why Split Atoms and Contexts?
The pipeline separates claim extraction from evidence retrieval, allowing:
- Targeted fact-checking of specific claims
- Reuse of evidence across multiple claims
- Better separation of concerns

### Why ChromaReader Instead of ContextRetriever?
ChromaReader provides direct access to the vector store with N results, while ContextRetriever is Google-based. ChromaReader is preferred for accessing pre-indexed documents.

### Configuration Choices

**Temperature = 0**: Ensures deterministic outputs for reproducibility during fact extraction and reasoning.

**MAX_NEW_TOKENS = 4096**: Allows detailed reasoning and explanation generation while limiting token sprawl.

**remove_duplicates = True**: Prevents duplicate atoms from inflating the reasoning graph.

**summarize_contexts = True**: Reduces context length for more focused reasoning.

## Main Execution Loop

The main execution is at the bottom of the file (lines 274-286):

```python
# Load use cases
with open("./ai_use_cases/rephrased_google_use_cases.json") as f:
    use_cases = json.load(f)

# Iterate through all risk-usecase pairs
for risk in risks_dict.keys():
    file_path = "output/fr_" + risk + "_sources_rephrased_google_use_cases.json"
    for usecase_index in label_indices_dict[risk]:
        # Prepare inputs
        usecase_index1 = usecase_index
        USECASE = use_cases[usecase_index]
        RISK = risks_dict[risk]
        INPUT = RISK + " is a risk associated with " + USECASE
        OUTPUT = RISK + " is a risk associated with " + USECASE
        
        # Run analysis
        print(risk, usecase_index1)
        fr_check(INPUT, OUTPUT, file_path, usecase_index1, RISK, USECASE)
```

## Common Modifications

### Analyze Only Specific Risks
```python
# Temporarily modify label_indices_dict
label_indices_dict = {
    "1.1": [0, 1],  # Only analyze risk 1.1
}
```

### Analyze More Use Cases
```python
label_indices_dict = {
    "1.1": [0, 1, 2, 3, 4, 5],  # Expand to more use cases
}
```

### Change LLM Backend
```python
# In fr_check(), replace:
backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, ...)
# With your preferred backend following mellea API
```

### Modify Output Path
```python
# Change file_path construction:
file_path = "custom_output/my_analysis_" + risk + ".json"
```

## Debugging

### Print Atoms and Contexts
The script prints extracted atoms and their associated contexts:
```python
for key, atom in pipeline.atoms.items():
    print(f"Atom {key}: {atom.text}")
    for context in atom.contexts:
        print(f"  Context {context}: {pipeline.contexts[context].text[:]}")
```

### Inspect Generated JSON
View `{RISK}{INDEX}.json` to see complete reasoning graphs with all relationships.

### Check HTML Visualization
Open `{RISK}{INDEX}.html` in a browser to interact with the reasoning graph.

## Performance Considerations

- **Batch Processing**: The loop processes multiple risk-usecase pairs sequentially
- **I/O**: JSON appends are done per-analysis (could be optimized to batch)
- **Memory**: Each FactReasoner instance is created fresh per analysis
- **LLM Calls**: Temperature=0 ensures consistent inference, but varies with claim complexity

## Troubleshooting

### Atomizer Failures
- **Cause**: Weak fact extraction from the claim
- **Solution**: Ensure OUTPUT is a clear, factual statement

### No Contexts Retrieved
- **Cause**: Vector store empty or embedding model mismatch
- **Solution**: Verify `vectorstore_sae_google` exists and is populated

### JSON Append Errors
- **Cause**: Invalid existing JSON in output file
- **Solution**: Delete corrupted JSON file and rerun (append_dict_to_file will create fresh)

### Merlin Path Issues
- **Cause**: Invalid MERLIN_PATH environment variable
- **Solution**: Verify path exists: `ls ./merlin/build/merlin`

## Extension Points

1. **Add New Risk Categories**: Modify `risks_dict` and `label_indices_dict`
2. **Change Output Format**: Modify the JSON structure in `fr_check()`
3. **Add Custom Scoring**: Extend post-processing after `pipeline.score()`
4. **Change Retrieval Strategy**: Replace `ChromaReader` with custom retriever
