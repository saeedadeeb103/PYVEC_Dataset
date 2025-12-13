# PyViz - Python Visualization Dataset Pipeline

A parallel pipeline to the TikZ extraction system, designed to extract Python visualization code (matplotlib, seaborn, plotly) from multiple sources and build a dataset for text-to-visualization generation.

## Project Structure

```
PyViz/
├── ArXiv/
│   └── arxiv_extractor.py      # Extract Python viz code from ArXiv papers
├── GitHub/
│   ├── github_extractor.py     # Search GitHub for repos with viz code
│   ├── github_clone_compress.py # Clone and compress found repos
│   └── github_cleaner.py       # Extract viz code from cloned repos
├── Stack/
│   └── stackexchange_extractor.py  # Extract from StackOverflow Q&A
├── Debug/
│   ├── llm_debug.py            # Use LLM to fix broken viz code
│   └── llm_debug.job           # SLURM job script
├── Describe/
│   ├── vlm_captioning.py       # Generate captions for viz images
│   └── vlm_captioning.job      # SLURM job script
├── postprocess_code.py         # Main post-processing pipeline
└── README.md
```

## Data Sources

### 1. ArXiv Papers
Extracts Python visualization code from ArXiv paper source files (`.py`, `.ipynb`).

```bash
cd ArXiv
python arxiv_extractor.py
```

### 2. GitHub Repositories
Searches and clones GitHub repos containing matplotlib/seaborn/plotly code.

```bash
cd GitHub
# Step 1: Find repos
python github_extractor.py

# Step 2: Clone and compress
python github_clone_compress.py

# Step 3: Extract viz code
python github_cleaner.py
```

### 3. StackOverflow
Extracts visualization code from StackOverflow Q&A posts tagged with matplotlib, seaborn, plotly.

```bash
cd Stack
python stackexchange_extractor.py
```

## Post-Processing

The main post-processing script extracts and cleans visualization code:

```bash
python postprocess_code.py
```

Features:
- Detects matplotlib, seaborn, and plotly code
- Extracts code from markdown blocks and Jupyter notebooks
- Adds missing imports automatically
- Extracts captions and text mentions

## Captioning Pipeline

Generate detailed descriptions for visualization images using a VLM:

```bash
cd Describe
python vlm_captioning.py \
    --model_id "your-vlm-model" \
    --batch_size 8 \
    --dataset_path "path/to/images" \
    --max_retries 3
```

## Debugging Pipeline

Fix broken visualization code using an LLM:

```bash
cd Debug
python llm_debug.py \
    --model_id "your-llm-model" \
    --batch_size 16 \
    --dataset_path "path/to/broken/code" \
    --max_retries 3
```

## Supported Libraries

| Library | Import Patterns | Plot Calls |
|---------|-----------------|------------|
| **matplotlib** | `import matplotlib`, `from matplotlib` | `plt.plot`, `plt.scatter`, `plt.bar`, `plt.hist`, `ax.plot`, etc. |
| **seaborn** | `import seaborn`, `from seaborn` | `sns.lineplot`, `sns.heatmap`, `sns.pairplot`, etc. |
| **plotly** | `import plotly`, `from plotly` | `px.scatter`, `px.line`, `go.Figure`, `fig.add_trace`, etc. |

## Output Format

Each extracted code sample includes:
- `code`: The Python visualization code
- `caption`: Description/context (if available)
- `library`: Detected visualization library (matplotlib, seaborn, plotly)
- `file_id`: Unique identifier
- `meta`: Source metadata (arxiv_id, repo, etc.)

## Requirements

```
datasets
tqdm
nltk
lxml
beautifulsoup4
httpx
aiohttp
openai
requests
```

## Directory Structure for Data

```
datikz_pyviz/
├── oai/                    # OAI resumption tokens
├── tarballs/               # Cached ArXiv tarballs
├── arxiv_extracted/        # Extracted ArXiv code
├── github_tarballs/        # Cloned GitHub repos
├── github_extracted/       # Extracted GitHub code
├── loaders/
│   └── github/
│       ├── seen_repos.json
│       ├── excluded_repos.json
│       └── github_key.json
├── raw/                    # Raw StackExchange dumps
└── data/                   # Final processed data
```
