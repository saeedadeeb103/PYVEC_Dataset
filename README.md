# PYVEC Dataset

**Python Visualization Executable Code Dataset**

A curated collection of **3,802 validated, executable Python visualization snippets** extracted from Kaggle, StackOverflow, GitHub, and official galleries. Every snippet is execution-tested to guarantee it produces a matplotlib/seaborn figure.

## 📊 Dataset Overview

| Source | Snippets | With Captions |
|--------|----------|---------------|
| **Kaggle** | 2,614 | 0 |
| **StackOverflow** | 605 | 0 |
| **Galleries** (Matplotlib/Seaborn) | 358 | 358 |
| **GitHub** | 225 | ~50 |
| **TOTAL** | **3,802** | **~408** |

### Key Features
- ✅ **100% executable** — All snippets validated by actual execution
- ✅ **100% produce figures** — Verified via `plt.get_fignums()`
- ✅ **Self-contained** — No external file dependencies or undefined variables
- ✅ **Synthetic data injection** — Incomplete code automatically fixed with sample DataFrames
- ✅ **Deduplicated** — MD5 hash checking removes duplicates

## 📁 Project Structure

```
pyvec_dataset/
├── Kaggle/
│   ├── kaggle_smart_extractor.py        # Kaggle notebook extractor
│   ├── kaggle_smart_validated.json      # 2,614 validated snippets
│   └── notebooks/
│       ├── seen_kernels_smart.json      # Tracks processed notebooks
│       └── completed_queries.json       # Tracks search queries (19/68 done)
├── StackOverflow/
│   ├── stackoverflow_extractor.py       # StackOverflow API extractor
│   ├── stackoverflow_validated.json     # 605 validated snippets
│   └── seen_questions.json              # Tracks processed questions
├── GitHub/
│   ├── github_smart_extractor.py        # GitHub tutorial repo extractor
│   ├── github_smart_validated.json      # 225 validated snippets
│   ├── seen_repos_smart.json            # Tracks processed repos
│   └── github_key.json                  # GitHub API keys (add your own)
├── Galleries/
│   ├── gallery_extractor.py             # Official gallery scraper
│   └── gallery_validated.json           # 358 validated snippets
├── Viewer/
│   ├── backend/app.py                   # Flask API for browsing snippets
│   ├── backend/requirements.txt         # Backend dependencies
│   └── frontend/index.html              # Web UI
├── all_validated_snippets.json          # Combined dataset (3,802 snippets)
├── EXTRACTION_PIPELINE.md               # Detailed methodology
└── MINDMAP.txt                          # Visual extraction pipeline
```

## 🚀 Quick Start

### Use the Dataset

The complete dataset is in `all_validated_snippets.json`:

```python
import json

# Load all snippets
with open('all_validated_snippets.json', 'r') as f:
    snippets = json.load(f)

print(f"Total snippets: {len(snippets)}")

# Example snippet structure
snippet = snippets[0]
print(snippet['code'])          # Python code
print(snippet['source_type'])   # Kaggle/StackOverflow/GitHub/Galleries
print(snippet.get('caption'))   # Description (if available)
```

### Run the Viewer

Browse snippets in a web interface:

```bash
cd Viewer/backend
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## 🔄 Extract More Data

All extractors support **resumable execution** — they track progress and skip already-processed items.

### Prerequisites

```bash
# Create conda environment
conda create -n pyvec python=3.10
conda activate pyvec

# Install dependencies
pip install kaggle requests beautifulsoup4 lxml tqdm numpy pandas matplotlib seaborn
```

### 1. Kaggle Extractor

**Status:** 19/68 queries completed, 2,614 snippets extracted

```bash
cd Kaggle

# Setup Kaggle API credentials (one-time)
# 1. Go to https://www.kaggle.com/settings
# 2. Create API token → downloads kaggle.json
# 3. Move to ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Run extractor (resumes from query #20)
python kaggle_smart_extractor.py
```

**How it works:**
- Searches 68 queries (matplotlib, seaborn, titanic, iris, etc.)
- Downloads notebooks, extracts plot cells
- Tracks imports and data setup across cells
- Injects synthetic DataFrames when needed
- Validates by executing with 5s timeout
- Saves progress: automatically resumes after rate limits

**Expected yield:** ~2,000 more snippets from remaining 49 queries

### 2. StackOverflow Extractor

**Status:** 324 questions processed, 605 snippets extracted

```bash
cd StackOverflow

# No authentication needed
python stackoverflow_extractor.py
```

**How it works:**
- Uses StackExchange API (300 requests/day limit)
- Searches tags: matplotlib, seaborn, combinations
- Extracts code blocks from high-score answers
- Validates execution with synthetic data
- Auto-retries on rate limit (429 errors)

**Expected yield:** Continues until API exhaustion, then resume next day

### 3. GitHub Extractor

**Status:** 567 repos processed, 225 snippets extracted

```bash
cd GitHub

# Setup GitHub API key (required)
# Create token at: https://github.com/settings/tokens
# Needs: public_repo scope
echo '["your_github_token_here"]' > github_key.json

# Run extractor
python github_smart_extractor.py
```

**How it works:**
- Searches tutorial/example repositories
- Downloads .py files, parses with AST
- Extracts plot functions and standalone blocks
- Validates with 10s timeout
- Rotates API keys to avoid rate limits (5,000/hour per key)

**Expected yield:** Continuous extraction from new repos

### 4. Gallery Extractor

**Status:** Complete (358/361 examples = 99.2% success)

```bash
cd Galleries

# No authentication needed
python gallery_extractor.py
```

**How it works:**
- Scrapes official Matplotlib & Seaborn galleries
- Downloads raw .py files from GitHub
- Cleans sphinx directives and docstrings
- Extracts docstrings as captions
- 100% success rate (official examples)

**Expected yield:** Only new examples when galleries update

### Combine Extracted Data

After running extractors, merge into final dataset:

```bash
python -c "
import json

sources = {
    'Kaggle': 'Kaggle/kaggle_smart_validated.json',
    'StackOverflow': 'StackOverflow/stackoverflow_validated.json',
    'GitHub': 'GitHub/github_smart_validated.json',
    'Galleries': 'Galleries/gallery_validated.json'
}

all_snippets = []
for name, path in sources.items():
    with open(path) as f:
        snippets = json.load(f)
        for s in snippets:
            s['source_type'] = name
        all_snippets.extend(snippets)
        print(f'{name}: {len(snippets)} snippets')

with open('all_validated_snippets.json', 'w') as f:
    json.dump(all_snippets, f, indent=2, ensure_ascii=False)

print(f'TOTAL: {len(all_snippets)} snippets')
"
```

## 📋 Dataset Schema

```json
{
  "code": "import matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(0, 10, 100)\nplt.plot(x, np.sin(x))\nplt.show()",
  "caption": "Simple sine wave plot",
  "library": "matplotlib",
  "source_type": "Kaggle",
  "source": "https://www.kaggle.com/username/notebook",
  "file_id": "kaggle_123456_0"
}
```

| Field | Type | Coverage | Description |
|-------|------|----------|-------------|
| `code` | string | 100% | Python source code |
| `caption` | string | ~11% | Description (Galleries only) |
| `library` | string | 100% | matplotlib, seaborn, or both |
| `source_type` | string | 100% | Kaggle, StackOverflow, GitHub, or Galleries |
| `source` | string | 100% | Original URL or reference |
| `file_id` | string | 100% | Unique identifier |

## 🔬 Validation Pipeline

Every snippet passes through:

1. **Code Execution Test**
   - Sets matplotlib backend: `matplotlib.use('Agg')`
   - 5-10 second timeout
   - Catches all exceptions

2. **Figure Verification**
   - Checks `plt.get_fignums() > 0`
   - Must create at least one figure

3. **Synthetic Data Injection**
   - Detects undefined variables: `df`, `data`, `x`, `y`
   - Injects 100-row DataFrame with numeric, categorical, temporal columns
   - Makes incomplete code runnable

4. **Bad Pattern Filtering**
   - Rejects: file I/O (`read_csv`, `read_excel`)
   - Rejects: network calls (`requests`, `urllib`)
   - Rejects: ML frameworks (`torch`, `tensorflow`)

5. **Deduplication**
   - MD5 hash of code
   - Removes exact duplicates

## 📖 Documentation

- **`EXTRACTION_PIPELINE.md`** — Detailed technical methodology, API usage, validation algorithms
- **`MINDMAP.txt`** — Visual ASCII tree of the extraction pipeline

## 🛠️ Requirements

```txt
# Core
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
tqdm>=4.66.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization (for validation)
matplotlib>=3.7.0
seaborn>=0.12.0

# APIs
kaggle>=1.5.0
PyGithub>=2.1.0

# Web viewer
flask>=2.3.0
flask-cors>=4.0.0
```

## 📊 Statistics

### Extraction Yield
| Source | Items Processed | Snippets | Yield Rate |
|--------|----------------|----------|------------|
| Kaggle | 2,406 notebooks | 2,614 | 1.09/notebook |
| StackOverflow | 324 questions | 605 | 1.87/question |
| GitHub | 567 repos | 225 | 0.40/repo |
| Galleries | 361 examples | 358 | 99.2% |

### Code Quality
- **100% executable** — No syntax errors
- **100% produce figures** — Verified output
- **0% duplicates** — MD5 deduplication
- **~11% captioned** — Galleries + some GitHub

## 🎯 Use Cases

- **Text-to-visualization generation** — Train models to generate viz code
- **Code completion** — Autocomplete for matplotlib/seaborn
- **Documentation examples** — Real-world usage patterns
- **Education** — Learn visualization best practices
- **Benchmarking** — Test code generation models

## 📝 Citation

If you use this dataset, please cite:

```bibtex
@dataset{pyvec2024,
  title={PYVEC: Python Visualization Executable Code Dataset},
  author={Your Name},
  year={2024},
  url={https://github.com/saeedadeeb103/PYVEC_Dataset},
  note={3,802 validated Python visualization snippets from Kaggle, StackOverflow, GitHub, and official galleries}
}
```

## 📄 License

Individual snippets retain their original licenses:
- **Kaggle**: Apache 2.0 (Kaggle Terms)
- **StackOverflow**: CC BY-SA 4.0
- **GitHub**: Repository-specific licenses
- **Galleries**: BSD-3-Clause (matplotlib), BSD-3-Clause (seaborn)

This dataset compilation is provided for research and educational purposes.

## 🤝 Contributing

To add more snippets:
1. Run the extractors as described above
2. Validate new snippets are executable
3. Merge into `all_validated_snippets.json`
4. Submit a pull request

## ⚠️ Notes

- **API Keys Required:**
  - Kaggle: `~/.kaggle/kaggle.json`
  - GitHub: `GitHub/github_key.json`
  - StackOverflow: No key needed (public API)

- **Rate Limits:**
  - Kaggle: ~200/hour (auto-retry)
  - StackOverflow: 300/day (resume next day)
  - GitHub: 5,000/hour per key (use multiple keys)
  - Galleries: No limit

- **Resumability:**
  All extractors track progress and can be stopped/restarted without data loss
