"""
Kaggle Smart Extractor

Extracts COMPLETE, runnable visualization snippets from Kaggle notebooks.
Key improvements:
1. Tracks imports and data setup across cells
2. Builds complete snippets with all dependencies
3. Validates each snippet before saving
4. Injects synthetic data when needed
"""

import os
import re
import json
import time
import signal
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from tqdm import tqdm

# Set matplotlib backend before importing
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
OUTPUT_DIR = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/Kaggle/notebooks"
OUTPUT_FILE = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/Kaggle/kaggle_smart_validated.json"

# Search queries - use simple terms that Kaggle API accepts well
SEARCH_QUERIES = [
    # Core visualization queries (known to work)
    "matplotlib",
    "seaborn",
    "visualization",
    "EDA",
    "plotting",
    "charts",
    "graphs",
    "analysis",
    # Dataset-specific (popular on Kaggle)
    "titanic",
    "iris",
    "housing",
    "netflix",
    "spotify",
    "covid",
    "sales",
    "customer",
    "stock",
    "weather",
    "crime",
    "healthcare",
    "finance",
    "sports",
    "movies",
    "music",
    "food",
    "wine",
    "cars",
    "flights",
    "hotel",
    "ecommerce",
    "retail",
    "marketing",
    "survey",
    "census",
    "diabetes",
    "heart",
    "cancer",
    "loan",
    "credit",
    "fraud",
    "churn",
    "sentiment",
    "twitter",
    "amazon",
    "imdb",
    "yelp",
    "airbnb",
    "uber",
    "bike",
    "energy",
    "pollution",
    "climate",
    "earthquake",
    "population",
    "education",
    "salary",
    "jobs",
    "startup",
    "bitcoin",
    "crypto",
    "nba",
    "fifa",
    "olympics",
    "pokemon",
    "anime",
    "games",
]

# Plot patterns
PLOT_PATTERNS = [
    r'plt\.(?:plot|scatter|bar|barh|hist|pie|boxplot|violinplot|imshow|contour|heatmap|fill|stem|step|errorbar|stackplot|hexbin)\s*\(',
    r'sns\.(?:lineplot|scatterplot|barplot|histplot|heatmap|boxplot|violinplot|pairplot|jointplot|kdeplot|regplot|countplot|catplot|stripplot|swarmplot|pointplot|lmplot|relplot|displot)\s*\(',
    r'ax\.(?:plot|scatter|bar|hist|imshow|pie|boxplot)\s*\(',
    r'\.plot\s*\([^)]*kind\s*=',
    r'fig,\s*ax',
]

# Patterns that indicate file/network dependencies
BAD_PATTERNS = [
    r'\.read_csv\s*\(["\'][^"\']+["\']',  # Reading specific files
    r'\.read_excel\s*\(',
    r'\.read_json\s*\(',
    r'\.read_sql',
    r'open\s*\(["\']',
    r'requests\.',
    r'urllib\.',
    r'wget\.',
    r'kaggle\.',
    r'drive\.mount',
    r'!pip\s+install',
    r'!wget',
    r'!curl',
    r'Image\s*\(',
    r'cv2\.',
    r'tensorflow',
    r'torch\.',
    r'keras\.',
]

# Standard imports we provide
STANDARD_IMPORTS = """import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
"""

# Synthetic data templates
SYNTHETIC_DATA = """
# Synthetic sample data
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100) * 2 + 1,
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'group': np.random.choice(['Group1', 'Group2'], 100),
    'value': np.random.randint(10, 100, 100),
    'score': np.random.uniform(0, 100, 100),
    'count': np.random.poisson(5, 100),
    'date': pd.date_range('2020-01-01', periods=100, freq='D'),
})
data = df.copy()
dataset = df.copy()
"""


def has_plot_call(code: str) -> bool:
    """Check if code has a plotting call."""
    return any(re.search(p, code) for p in PLOT_PATTERNS)


def has_bad_pattern(code: str) -> bool:
    """Check for patterns that indicate external dependencies."""
    return any(re.search(p, code, re.IGNORECASE) for p in BAD_PATTERNS)


def extract_variable_assignments(code: str) -> Dict[str, str]:
    """Extract variable assignments from code."""
    assignments = {}
    
    # Simple assignment: var = ...
    for match in re.finditer(r'^(\w+)\s*=\s*(.+?)(?:\n|$)', code, re.MULTILINE):
        var_name = match.group(1)
        assignments[var_name] = match.group(0)
    
    return assignments


def get_used_variables(code: str) -> Set[str]:
    """Get variables used in code."""
    # Find all word tokens that look like variable names
    tokens = set(re.findall(r'\b([a-zA-Z_]\w*)\b', code))
    
    # Remove Python keywords and common functions
    keywords = {
        'import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif',
        'for', 'while', 'in', 'not', 'and', 'or', 'True', 'False', 'None',
        'try', 'except', 'finally', 'with', 'lambda', 'pass', 'break', 'continue',
        'print', 'len', 'range', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float',
        'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
    }
    
    return tokens - keywords


def test_snippet(code: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """Test if a snippet executes and produces a figure."""
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.ioff()
    plt.close('all')
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Timeout")
    
    exec_globals = {
        '__builtins__': __builtins__,
        'np': np,
        'pd': pd,
        'plt': plt,
        'matplotlib': matplotlib,
        'numpy': np,
        'pandas': pd,
    }
    
    try:
        import seaborn as sns
        exec_globals['sns'] = sns
        exec_globals['seaborn'] = sns
    except ImportError:
        pass
    
    # Clean up code
    modified = code.replace('plt.show()', 'pass').replace('fig.show()', 'pass')
    modified = re.sub(r"plt\.savefig\s*\([^)]*\)", "pass", modified)
    modified = re.sub(r"fig\.savefig\s*\([^)]*\)", "pass", modified)
    modified = re.sub(r"print\s*\([^)]*\)", "pass", modified)
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        exec(modified, exec_globals)
        
        signal.alarm(0)
        
        if plt.get_fignums():
            plt.close('all')
            return True, ""
        else:
            return False, "No figure"
            
    except TimeoutError:
        signal.alarm(0)
        plt.close('all')
        return False, "Timeout"
    except Exception as e:
        signal.alarm(0)
        plt.close('all')
        return False, str(e)[:80]
    finally:
        signal.alarm(0)


def build_complete_snippet(plot_cell: str, context_cells: List[str]) -> str:
    """Build a complete snippet with imports and context."""
    
    # Start with standard imports
    parts = [STANDARD_IMPORTS]
    
    # Check if we need synthetic data
    needs_data = False
    data_vars = ['df', 'data', 'dataset', 'train', 'test']
    
    for var in data_vars:
        if re.search(rf'\b{var}\b', plot_cell):
            # Check if it's defined in context
            defined_in_context = False
            for ctx in context_cells:
                if re.search(rf'\b{var}\s*=', ctx):
                    defined_in_context = True
                    break
            
            if not defined_in_context:
                needs_data = True
                break
    
    if needs_data:
        parts.append(SYNTHETIC_DATA)
    
    # Add relevant context (imports, small data setups)
    for ctx in context_cells:
        ctx = ctx.strip()
        if not ctx:
            continue
        
        # Skip cells with bad patterns
        if has_bad_pattern(ctx):
            continue
        
        # Include import statements
        if ctx.startswith('import ') or ctx.startswith('from '):
            # Skip if already in standard imports
            if 'matplotlib' not in ctx and 'numpy' not in ctx and 'pandas' not in ctx and 'seaborn' not in ctx:
                parts.append(ctx)
        
        # Include small variable assignments that might be needed
        elif '=' in ctx and len(ctx) < 200 and not has_bad_pattern(ctx):
            # Check if any variable from this cell is used in plot_cell
            assigned_vars = set(re.findall(r'^(\w+)\s*=', ctx, re.MULTILINE))
            used_in_plot = get_used_variables(plot_cell)
            
            if assigned_vars & used_in_plot:
                parts.append(ctx)
    
    # Add the plot cell
    parts.append(plot_cell.strip())
    
    # Ensure plt.show() at the end
    full_code = '\n\n'.join(parts)
    if 'plt.show()' not in full_code:
        full_code += '\nplt.show()'
    
    return full_code


def extract_from_notebook(notebook_path: str) -> List[Dict]:
    """Extract validated snippets from a notebook."""
    snippets = []
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except:
        return []
    
    # Extract all code cells
    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code.strip())
    
    if not code_cells:
        return []
    
    # Find cells with plot calls
    for i, cell in enumerate(code_cells):
        if not has_plot_call(cell):
            continue
        
        if has_bad_pattern(cell):
            continue
        
        # Get context from previous cells
        context = code_cells[:i]
        
        # Build complete snippet
        complete_code = build_complete_snippet(cell, context)
        
        # Test it
        success, error = test_snippet(complete_code)
        
        if success:
            snippets.append({
                'code': complete_code,
                'caption': '',
                'library': detect_library(complete_code),
            })
    
    return snippets


def detect_library(code: str) -> str:
    """Detect visualization library."""
    libs = []
    if 'plt.' in code or 'matplotlib' in code:
        libs.append('matplotlib')
    if 'sns.' in code or 'seaborn' in code:
        libs.append('seaborn')
    return ','.join(libs) if libs else 'matplotlib'


def search_kernels(query: str, page: int = 1, page_size: int = 20) -> List[Dict]:
    """Search for Kaggle kernels."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        kernels = api.kernels_list(
            search=query,
            page=page,
            page_size=page_size,
        )
        
        return [{'ref': k.ref, 'title': k.title} for k in kernels]
    except Exception as e:
        if '429' in str(e):
            logging.warning("Rate limited, waiting 60 seconds...")
            time.sleep(60)
            return search_kernels(query, page, page_size)
        logging.error(f"Search failed: {e}")
        return []


def download_kernel(kernel_ref: str, output_dir: str) -> Optional[str]:
    """Download a Kaggle kernel."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        os.makedirs(output_dir, exist_ok=True)
        api.kernels_pull(kernel_ref, path=output_dir, metadata=False)
        
        for f in os.listdir(output_dir):
            if f.endswith('.ipynb'):
                return os.path.join(output_dir, f)
        
        return None
    except Exception as e:
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_snippets = []
    seen_hashes = set()
    seen_kernels = set()
    seen_queries = set()
    
    # Load progress
    seen_file = os.path.join(OUTPUT_DIR, "seen_kernels_smart.json")
    if os.path.exists(seen_file):
        with open(seen_file, 'r') as f:
            seen_kernels = set(json.load(f))
    
    # Load completed queries
    queries_file = os.path.join(OUTPUT_DIR, "completed_queries.json")
    if os.path.exists(queries_file):
        with open(queries_file, 'r') as f:
            seen_queries = set(json.load(f))
        logging.info(f"Already completed {len(seen_queries)} queries")
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            all_snippets = json.load(f)
            for s in all_snippets:
                seen_hashes.add(hashlib.md5(s['code'].encode()).hexdigest())
        logging.info(f"Resuming with {len(all_snippets)} existing snippets")
    
    logging.info("Starting smart extraction from Kaggle...")
    
    # Filter to only new queries
    new_queries = [q for q in SEARCH_QUERIES if q not in seen_queries]
    logging.info(f"Found {len(new_queries)} new queries to search")
    
    for query in new_queries:
        logging.info(f"\nSearching: {query}")
        
        for page in range(1, 6):  # 5 pages per query
            kernels = search_kernels(query, page=page, page_size=20)
            
            if not kernels:
                break
            
            for kernel in tqdm(kernels, desc=f"'{query}' p{page}", leave=False):
                kernel_ref = kernel['ref']
                
                if kernel_ref in seen_kernels:
                    continue
                seen_kernels.add(kernel_ref)
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    notebook_path = download_kernel(kernel_ref, tmpdir)
                    
                    if not notebook_path:
                        continue
                    
                    snippets = extract_from_notebook(notebook_path)
                    
                    for snippet in snippets:
                        code_hash = hashlib.md5(snippet['code'].encode()).hexdigest()
                        if code_hash in seen_hashes:
                            continue
                        seen_hashes.add(code_hash)
                        
                        snippet['file_id'] = f"{kernel_ref.replace('/', '_')}_{len(all_snippets)}"
                        snippet['source'] = kernel_ref
                        all_snippets.append(snippet)
                    
                    if snippets:
                        logging.info(f"  ✓ {kernel_ref}: {len(snippets)} validated snippets")
                
                time.sleep(0.5)  # Be nice to API
        
        # Mark query as completed
        seen_queries.add(query)
        
        # Save progress
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_snippets, f, indent=2, ensure_ascii=False)
        
        with open(seen_file, 'w') as f:
            json.dump(list(seen_kernels), f)
        
        with open(queries_file, 'w') as f:
            json.dump(list(seen_queries), f)
        
        logging.info(f"Progress: {len(all_snippets)} validated snippets from {len(seen_kernels)} notebooks")
    
    logging.info(f"\n{'='*50}")
    logging.info(f"✅ Extracted {len(all_snippets)} VALIDATED snippets!")
    logging.info(f"   From {len(seen_kernels)} Kaggle notebooks")
    logging.info(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
