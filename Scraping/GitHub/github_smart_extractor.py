"""
GitHub Smart Snippet Extractor

Extracts self-contained, validated visualization FUNCTIONS from GitHub.
Targets tutorial/example repositories and extracts individual plotting functions.

Strategy:
1. Search for tutorial/example repos (not random projects)
2. Parse Python files with AST to extract individual functions
3. Find functions that create plots
4. Build complete snippets with dependencies
5. Validate each snippet executes and produces a figure
"""

import os
import re
import ast
import json
import time
import signal
import logging
import hashlib
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm
import requests

# Set matplotlib backend before importing
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import quality gate from repo root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from clean_scraped_data import build_quality_gate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
OUTPUT_DIR = str(Path(__file__).resolve().parent)
OUTPUT_FILE = str(Path(OUTPUT_DIR) / "github_smart_validated.json")
SEEN_FILE = str(Path(OUTPUT_DIR) / "seen_repos_smart.json")
KEYS_FILE = str(Path(OUTPUT_DIR) / "github_key.json")

# Search queries targeting TUTORIAL/EXAMPLE repos
SEARCH_QUERIES = [
    "matplotlib examples tutorial",
    "matplotlib tutorial language:python",
    "seaborn examples tutorial",
    "seaborn tutorial language:python",
    "data visualization tutorial python",
    "python plotting examples",
    "matplotlib gallery",
    "seaborn gallery",
    "python charts examples",
    "pandas visualization tutorial",
    "matplotlib cookbook",
    "python data visualization examples",
    "matplotlib beginner",
    "seaborn beginner tutorial",
    "python plot examples",
    "python visualization notebook examples",
    "matplotlib cheatsheet examples language:python",
    "seaborn plotting examples language:python",
    "pandas plotting tutorial language:python",
    "python charts cookbook language:python",
    "data viz demo language:python",
    "exploratory data analysis visualization language:python",
    "plotly matplotlib seaborn examples language:python",
    "python statistical plots examples",
]
MAX_PAGES_PER_QUERY = int(os.getenv("GITHUB_MAX_PAGES", "8"))
REPOS_PER_PAGE = int(os.getenv("GITHUB_REPOS_PER_PAGE", "30"))
MAX_FILES_PER_REPO = int(os.getenv("GITHUB_MAX_FILES_PER_REPO", "25"))
MAX_DIRS_PER_LEVEL = int(os.getenv("GITHUB_MAX_DIRS_PER_LEVEL", "20"))
COOLDOWN_BASE_SECONDS = float(os.getenv("GITHUB_COOLDOWN_BASE_SECONDS", "2"))
COOLDOWN_MAX_RETRIES = int(os.getenv("GITHUB_COOLDOWN_MAX_RETRIES", "4"))
COOLDOWN_MAX_SECONDS = float(os.getenv("GITHUB_COOLDOWN_MAX_SECONDS", "60"))

# Standard imports
STANDARD_IMPORTS = """import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
"""

# Synthetic data for missing variables
SYNTHETIC_DATA = """
# Sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.randn(100) * 0.1
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100) * 2 + 1,
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'group': np.random.choice(['Group1', 'Group2'], 100),
    'value': np.random.randint(10, 100, 100),
    'score': np.random.uniform(0, 100, 100),
})
data = df.copy()
"""

# Plot patterns
PLOT_CALLS = {
    'plt.plot', 'plt.scatter', 'plt.bar', 'plt.barh', 'plt.hist', 'plt.pie',
    'plt.boxplot', 'plt.violinplot', 'plt.imshow', 'plt.contour', 'plt.contourf',
    'plt.heatmap', 'plt.fill', 'plt.fill_between', 'plt.stem', 'plt.step',
    'plt.errorbar', 'plt.stackplot', 'plt.hexbin', 'plt.pcolormesh',
    'sns.lineplot', 'sns.scatterplot', 'sns.barplot', 'sns.histplot',
    'sns.heatmap', 'sns.boxplot', 'sns.violinplot', 'sns.pairplot',
    'sns.jointplot', 'sns.kdeplot', 'sns.regplot', 'sns.countplot',
    'sns.catplot', 'sns.stripplot', 'sns.swarmplot', 'sns.pointplot',
    'sns.lmplot', 'sns.relplot', 'sns.displot', 'sns.clustermap',
    'ax.plot', 'ax.scatter', 'ax.bar', 'ax.hist', 'ax.imshow', 'ax.pie',
}

# Bad patterns - skip files/functions with these
BAD_PATTERNS = [
    r'\.read_csv\s*\(["\'][^"\']+["\']',
    r'\.read_excel\s*\(',
    r'open\s*\(["\']',
    r'requests\.',
    r'urllib\.',
    r'cv2\.',
    r'tensorflow',
    r'torch\.',
    r'keras\.',
    r'sklearn\.datasets\.load_',
    r'input\s*\(',
    r'argparse',
    r'sys\.argv',
]


def load_json_with_fallback(path: str, default):
    """Load JSON with encoding fallbacks for Windows/local legacy files."""
    encodings = ("utf-8", "utf-8-sig", "cp1252")
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception as e:
            last_err = e
    logging.warning(f"Could not load JSON from {path}: {last_err}")
    return default


def load_github_keys():
    """Load GitHub API keys."""
    try:
        data = load_json_with_fallback(KEYS_FILE, {})
        keys = [v for k, v in data.get('keys', {}).items() if v]
        return keys if keys else [None]
    except Exception:
        return [None]


class GitHubAPI:
    def __init__(self):
        self.keys = load_github_keys()
        self.key_idx = 0
        self.session = requests.Session()
    
    def _get_headers(self):
        key = self.keys[self.key_idx] if self.keys else None
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if key:
            headers['Authorization'] = f'token {key}'
        return headers
    
    def _rotate_key(self):
        self.key_idx = (self.key_idx + 1) % len(self.keys)

    def _cooldown_seconds(self, attempt: int) -> float:
        return min(COOLDOWN_BASE_SECONDS * (2 ** attempt), COOLDOWN_MAX_SECONDS)

    def _request(self, url: str, params: Dict = None, timeout: int = 30):
        """HTTP GET with Kaggle-like capped cooldown retries."""
        params = params or {}
        for attempt in range(COOLDOWN_MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, headers=self._get_headers(), params=params, timeout=timeout)
                if resp.status_code == 200:
                    return resp

                # Retry only if likely transient.
                retryable = resp.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                if resp.status_code == 403:
                    # Retry 403 only when we're truly rate-limited.
                    remaining = resp.headers.get("X-RateLimit-Remaining", "")
                    if remaining == "0":
                        retryable = True
                        self._rotate_key()
                    else:
                        return resp

                if not retryable:
                    return resp

                if attempt >= COOLDOWN_MAX_RETRIES:
                    return resp

                wait_s = self._cooldown_seconds(attempt)
                logging.warning(f"GET retry ({resp.status_code}) attempt={attempt+1}/{COOLDOWN_MAX_RETRIES} wait={wait_s:.1f}s url={url}")
                time.sleep(wait_s)
            except Exception as e:
                if attempt >= COOLDOWN_MAX_RETRIES:
                    raise
                wait_s = self._cooldown_seconds(attempt)
                logging.warning(f"GET exception attempt={attempt+1}/{COOLDOWN_MAX_RETRIES} wait={wait_s:.1f}s url={url} err={e}")
                time.sleep(wait_s)
        return None
    
    def search_repos(self, query: str, page: int = 1, per_page: int = 30) -> List[Dict]:
        """Search for repositories."""
        url = "https://api.github.com/search/repositories"
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'page': page,
            'per_page': per_page,
        }
        
        try:
            resp = self._request(url, params=params, timeout=30)
            if resp is not None and resp.status_code == 200:
                return resp.json().get('items', [])
            status = resp.status_code if resp is not None else "None"
            logging.warning(f"Search failed: {status}")
            return []
        except Exception as e:
            logging.error(f"Request failed after retries: {e}")
            return []
    
    def get_repo_contents(self, owner: str, repo: str, path: str = "") -> List[Dict]:
        """Get repository contents."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        try:
            resp = self._request(url, timeout=30)
            if resp is not None and resp.status_code == 200:
                return resp.json() if isinstance(resp.json(), list) else [resp.json()]
            return []
        except Exception:
            return []
    
    def get_file_content(self, url: str) -> Optional[str]:
        """Get raw file content."""
        try:
            # Convert to raw URL
            raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            resp = self._request(raw_url, timeout=30)
            if resp is not None and resp.status_code == 200:
                return resp.text
            return None
        except Exception:
            return None


def has_plot_call(code: str) -> bool:
    """Check if code has a plotting call."""
    for call in PLOT_CALLS:
        if call in code:
            return True
    return False


def has_bad_pattern(code: str) -> bool:
    """Check for patterns that indicate external dependencies."""
    return any(re.search(p, code, re.IGNORECASE) for p in BAD_PATTERNS)


def extract_functions_with_plots(source_code: str) -> List[Tuple[str, str, str]]:
    """
    Extract functions that contain plot calls.
    Returns list of (function_name, function_code, docstring)
    """
    functions = []
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function source
            try:
                func_lines = source_code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_lines)
            except:
                continue
            
            # Check if function has plot calls
            if not has_plot_call(func_code):
                continue
            
            # Skip if has bad patterns
            if has_bad_pattern(func_code):
                continue
            
            # Get docstring
            docstring = ast.get_docstring(node) or ""
            
            # Skip very short or very long functions
            if len(func_code) < 50 or len(func_code) > 3000:
                continue
            
            functions.append((node.name, func_code, docstring))
    
    return functions


def extract_standalone_plot_blocks(source_code: str) -> List[str]:
    """
    Extract standalone plot code blocks (not in functions).
    Looks for consecutive lines that create a plot.
    """
    blocks = []
    lines = source_code.split('\n')
    
    current_block = []
    in_plot_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines and comments at start
        if not current_block and (not stripped or stripped.startswith('#')):
            if stripped.startswith('#') and 'example' in stripped.lower():
                current_block.append(line)
            continue
        
        # Check if line is part of plotting code
        is_plot_line = (
            'plt.' in line or 
            'sns.' in line or 
            'ax.' in line or
            'fig' in line or
            'import' in line or
            stripped.startswith('#') or
            '=' in line and any(x in line for x in ['np.', 'pd.', 'range(', 'linspace', 'random'])
        )
        
        if is_plot_line:
            current_block.append(line)
            if has_plot_call(line):
                in_plot_block = True
        elif in_plot_block and current_block:
            # End of block
            block_code = '\n'.join(current_block)
            if has_plot_call(block_code) and not has_bad_pattern(block_code):
                if 50 < len(block_code) < 2000:
                    blocks.append(block_code)
            current_block = []
            in_plot_block = False
        elif current_block and len(current_block) > 20:
            # Block too long without plot, reset
            current_block = []
            in_plot_block = False
    
    # Don't forget last block
    if current_block and in_plot_block:
        block_code = '\n'.join(current_block)
        if has_plot_call(block_code) and not has_bad_pattern(block_code):
            if 50 < len(block_code) < 2000:
                blocks.append(block_code)
    
    return blocks


def build_complete_snippet(code: str, is_function: bool = False) -> str:
    """Build a complete, runnable snippet."""
    parts = [STANDARD_IMPORTS]
    
    # Check if needs sample data
    needs_data = False
    data_vars = ['df', 'data', 'x', 'y', 'X', 'Y']
    for var in data_vars:
        if re.search(rf'\b{var}\b', code) and not re.search(rf'\b{var}\s*=', code):
            needs_data = True
            break
    
    if needs_data:
        parts.append(SYNTHETIC_DATA)
    
    if is_function:
        # Add function definition and call it
        parts.append(code)
        # Extract function name and call it
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            func_name = match.group(1)
            # Check function parameters
            param_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', code)
            if param_match:
                params = param_match.group(1).strip()
                if not params or params == 'self':
                    parts.append(f"\n{func_name}()")
                elif 'df' in params or 'data' in params:
                    parts.append(f"\n{func_name}(df)")
                elif 'x' in params and 'y' in params:
                    parts.append(f"\n{func_name}(x, y)")
                elif 'x' in params:
                    parts.append(f"\n{func_name}(x)")
                else:
                    # Try calling with no args
                    parts.append(f"\n{func_name}()")
    else:
        parts.append(code)
    
    full_code = '\n\n'.join(parts)
    
    # Ensure plt.show() at the end
    if 'plt.show()' not in full_code:
        full_code += '\nplt.show()'
    
    return full_code


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


def detect_library(code: str) -> str:
    """Detect visualization library."""
    libs = []
    if 'plt.' in code or 'matplotlib' in code:
        libs.append('matplotlib')
    if 'sns.' in code or 'seaborn' in code:
        libs.append('seaborn')
    return ','.join(libs) if libs else 'matplotlib'


def find_python_files(api: GitHubAPI, owner: str, repo: str, path: str = "", depth: int = 0) -> List[str]:
    """Recursively find Python files in a repo."""
    if depth > 4:  # Limit depth
        return []
    
    python_files = []
    contents = api.get_repo_contents(owner, repo, path)
    
    dir_count = 0
    for item in contents:
        if item.get('type') == 'file' and item.get('name', '').endswith('.py'):
            # Prioritize example/tutorial files
            name_lower = item['name'].lower()
            if any(x in name_lower for x in ['example', 'tutorial', 'demo', 'plot', 'viz', 'chart', 'graph']):
                python_files.insert(0, item.get('download_url', ''))
            else:
                python_files.append(item.get('download_url', ''))
        elif item.get('type') == 'dir':
            if dir_count >= MAX_DIRS_PER_LEVEL:
                continue
            dir_count += 1
            # Prioritize example directories
            dir_name = item.get('name', '').lower()
            if any(x in dir_name for x in ['example', 'tutorial', 'demo', 'gallery', 'notebook']):
                python_files.extend(find_python_files(api, owner, repo, item['path'], depth + 1))
            elif depth < 2:
                # Traverse only likely source dirs to avoid huge crawls.
                if not any(x in dir_name for x in ['src', 'plot', 'viz', 'chart', 'doc', 'sample', 'lesson']):
                    continue
                python_files.extend(find_python_files(api, owner, repo, item['path'], depth + 1))
    
    return python_files[:50]  # Pre-limit candidate files per repo


def process_repo(api: GitHubAPI, repo: Dict, seen_hashes: Set[str], quality_gate) -> List[Dict]:
    """Process a single repository."""
    snippets = []
    
    owner = repo.get('owner', {}).get('login', '')
    repo_name = repo.get('name', '')
    
    if not owner or not repo_name:
        return []
    
    # Find Python files
    python_files = find_python_files(api, owner, repo_name)
    if not python_files:
        return []

    logging.info(f"  -> scanning {owner}/{repo_name} ({min(len(python_files), MAX_FILES_PER_REPO)} files)")
    
    for file_url in tqdm(python_files[:MAX_FILES_PER_REPO], desc=f"Files {owner}/{repo_name}", leave=False):
        if not file_url:
            continue
        
        # Get file content
        try:
            resp = api._request(file_url, timeout=30)
            if resp is None or resp.status_code != 200:
                continue
            source_code = resp.text
        except:
            continue
        
        # Skip large files
        if len(source_code) > 50000:
            continue
        
        # Extract functions with plots
        functions = extract_functions_with_plots(source_code)
        for func_name, func_code, docstring in functions:
            complete_code = build_complete_snippet(func_code, is_function=True)
            
            # Test it
            success, error = test_snippet(complete_code)
            
            if success:
                code_hash = hashlib.md5(complete_code.encode()).hexdigest()
                if code_hash in seen_hashes:
                    continue
                seen_hashes.add(code_hash)
                
                snippet = {
                    'code': complete_code,
                    'caption': docstring[:200] if docstring else '',
                    'library': detect_library(complete_code),
                    'source': f"https://github.com/{owner}/{repo_name}",
                }
                snippet = quality_gate.clean_and_validate(snippet)
                if snippet:
                    snippets.append(snippet)
        
        # Extract standalone plot blocks
        blocks = extract_standalone_plot_blocks(source_code)
        for block in blocks:
            complete_code = build_complete_snippet(block, is_function=False)
            
            success, error = test_snippet(complete_code)
            
            if success:
                code_hash = hashlib.md5(complete_code.encode()).hexdigest()
                if code_hash in seen_hashes:
                    continue
                seen_hashes.add(code_hash)
                
                snippet = {
                    'code': complete_code,
                    'caption': '',
                    'library': detect_library(complete_code),
                    'source': f"https://github.com/{owner}/{repo_name}",
                }
                snippet = quality_gate.clean_and_validate(snippet)
                if snippet:
                    snippets.append(snippet)
        
        time.sleep(0.2)  # Be nice
    
    return snippets


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    quality_gate = build_quality_gate()

    api = GitHubAPI()
    all_snippets = []
    seen_hashes = set()
    seen_repos = set()

    # Load progress
    if os.path.exists(SEEN_FILE):
        seen_repos = set(load_json_with_fallback(SEEN_FILE, []))
    
    if os.path.exists(OUTPUT_FILE):
        all_snippets = load_json_with_fallback(OUTPUT_FILE, [])
        for s in all_snippets:
            code = s.get("code", "")
            if code:
                seen_hashes.add(hashlib.md5(code.encode()).hexdigest())
        logging.info(f"Resuming with {len(all_snippets)} existing snippets")

    logging.info("Starting GitHub smart extraction...")
    logging.info(
        f"Config -> queries={len(SEARCH_QUERIES)}, pages/query={MAX_PAGES_PER_QUERY}, "
        f"repos/page={REPOS_PER_PAGE}, files/repo={MAX_FILES_PER_REPO}"
    )

    total_repos_seen = 0
    total_repos_processed = 0
    total_new_snippets = 0

    for query in tqdm(SEARCH_QUERIES, desc="GitHub queries"):
        logging.info(f"\nSearching query: {query}")

        for page in tqdm(range(1, MAX_PAGES_PER_QUERY + 1), desc=f"Pages for '{query}'", leave=False):
            repos = api.search_repos(query, page=page, per_page=REPOS_PER_PAGE)

            if not repos:
                logging.info(f"  No more repos for query='{query}' page={page}")
                break

            for repo in tqdm(repos, desc=f"Repos '{query}' p{page}", leave=False):
                repo_full_name = repo.get('full_name', '')
                total_repos_seen += 1

                if repo_full_name in seen_repos:
                    continue
                seen_repos.add(repo_full_name)

                # Process repo (isolate failures so long runs don't stop)
                try:
                    snippets = process_repo(api, repo, seen_hashes, quality_gate)
                except Exception as e:
                    logging.warning(f"Skipping repo after error {repo_full_name}: {e}")
                    snippets = []
                total_repos_processed += 1

                if snippets:
                    for i, snippet in enumerate(snippets):
                        snippet['file_id'] = f"gh_{repo_full_name.replace('/', '_')}_{len(all_snippets) + i}"
                    all_snippets.extend(snippets)
                    total_new_snippets += len(snippets)
                    logging.info(f"  + {repo_full_name}: kept {len(snippets)} snippets")

                time.sleep(0.5)

        # Save progress
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_snippets, f, indent=2, ensure_ascii=False)

        with open(SEEN_FILE, 'w') as f:
            json.dump(list(seen_repos), f)

        logging.info(
            f"Checkpoint -> total_snippets={len(all_snippets)}, seen_repos={len(seen_repos)}, "
            f"repos_seen_this_run={total_repos_seen}, repos_processed_this_run={total_repos_processed}, "
            f"new_snippets_this_run={total_new_snippets}"
        )

        time.sleep(2)  # Rate limit

    logging.info(f"\n{'='*50}")
    logging.info(f"Done: extracted {len(all_snippets)} validated snippets from GitHub")
    logging.info(f"   From {len(seen_repos)} repositories")
    logging.info(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
