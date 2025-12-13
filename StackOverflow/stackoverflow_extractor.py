"""
StackOverflow Smart Validated Extractor

Extracts self-contained, validated Python visualization snippets from StackOverflow.
Uses the StackExchange API to search for questions with matplotlib/seaborn tags.
"""

import os
import re
import json
import time
import signal
import logging
import hashlib
import html
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import requests

# Set matplotlib backend before importing
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
OUTPUT_DIR = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/StackOverflow"
OUTPUT_FILE = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/StackOverflow/stackoverflow_validated.json"
SEEN_FILE = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/StackOverflow/seen_questions.json"

# StackExchange API
API_BASE = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"

# Tags to search
SEARCH_TAGS = [
    "matplotlib",
    "seaborn", 
    "matplotlib;plot",
    "matplotlib;bar-chart",
    "matplotlib;histogram",
    "matplotlib;scatter-plot",
    "matplotlib;heatmap",
    "seaborn;heatmap",
    "matplotlib;pie-chart",
    "matplotlib;boxplot",
    "python;data-visualization",
    "pandas;matplotlib",
    "matplotlib;subplot",
    "matplotlib;legend",
    "matplotlib;colorbar",
]

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

# Synthetic data
SYNTHETIC_DATA = """
# Sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.randn(100) * 0.2
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
PLOT_PATTERNS = [
    r'plt\.(?:plot|scatter|bar|barh|hist|pie|boxplot|violinplot|imshow|contour|heatmap|fill|stem|step|errorbar|stackplot|hexbin|show)\s*\(',
    r'sns\.(?:lineplot|scatterplot|barplot|histplot|heatmap|boxplot|violinplot|pairplot|jointplot|kdeplot|regplot|countplot|catplot|stripplot|swarmplot|pointplot|lmplot|relplot|displot)\s*\(',
    r'ax\.(?:plot|scatter|bar|hist|imshow|pie|boxplot)\s*\(',
    r'\.plot\s*\([^)]*\)',
    r'fig,\s*ax',
]

# Bad patterns
BAD_PATTERNS = [
    r'\.read_csv\s*\(["\'][^"\']+["\']',
    r'\.read_excel\s*\(',
    r'open\s*\(["\']',
    r'requests\.',
    r'urllib\.',
    r'cv2\.',
    r'PIL\.',
    r'Image\.',
    r'tensorflow',
    r'torch\.',
    r'keras\.',
    r'sklearn\.datasets\.load_',
    r'input\s*\(',
]


def has_plot_call(code: str) -> bool:
    """Check if code has a plotting call."""
    return any(re.search(p, code) for p in PLOT_PATTERNS)


def has_bad_pattern(code: str) -> bool:
    """Check for patterns that indicate external dependencies."""
    return any(re.search(p, code, re.IGNORECASE) for p in BAD_PATTERNS)


def extract_code_blocks(html_content: str) -> List[str]:
    """Extract Python code blocks from HTML content."""
    # Decode HTML entities
    content = html.unescape(html_content)
    
    # Find code blocks
    code_blocks = []
    
    # Pattern for <code> tags
    code_pattern = r'<code>(.*?)</code>'
    matches = re.findall(code_pattern, content, re.DOTALL)
    
    for match in matches:
        # Clean up the code
        code = match.strip()
        code = re.sub(r'<[^>]+>', '', code)  # Remove any remaining HTML tags
        
        # Only keep Python-looking code with plot calls
        if has_plot_call(code) and ('import' in code or 'plt.' in code or 'sns.' in code):
            code_blocks.append(code)
    
    # Also try <pre><code> pattern
    pre_pattern = r'<pre[^>]*><code[^>]*>(.*?)</code></pre>'
    matches = re.findall(pre_pattern, content, re.DOTALL)
    
    for match in matches:
        code = html.unescape(match.strip())
        code = re.sub(r'<[^>]+>', '', code)
        
        if has_plot_call(code) and not has_bad_pattern(code):
            if code not in code_blocks:
                code_blocks.append(code)
    
    return code_blocks


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


def build_complete_snippet(code: str) -> str:
    """Build a complete snippet with imports and sample data."""
    
    # Check what's needed
    needs_imports = 'import' not in code
    needs_data = False
    
    data_vars = ['df', 'data', 'x', 'y', 'X', 'Y']
    for var in data_vars:
        pattern = rf'\b{var}\b'
        if re.search(pattern, code):
            # Check if defined in code
            if not re.search(rf'\b{var}\s*=', code):
                needs_data = True
                break
    
    parts = []
    
    # Add imports if needed
    if needs_imports:
        parts.append(STANDARD_IMPORTS)
    else:
        # Still add backend setup
        parts.append("import matplotlib\nmatplotlib.use('Agg')\nimport warnings\nwarnings.filterwarnings('ignore')\n")
    
    # Add sample data if needed
    if needs_data:
        parts.append(SYNTHETIC_DATA)
    
    parts.append(code)
    
    # Ensure plt.show() at the end
    full_code = '\n\n'.join(parts)
    if 'plt.show()' not in full_code:
        full_code += '\nplt.show()'
    
    return full_code


def detect_library(code: str) -> str:
    """Detect visualization library."""
    libs = []
    if 'plt.' in code or 'matplotlib' in code:
        libs.append('matplotlib')
    if 'sns.' in code or 'seaborn' in code:
        libs.append('seaborn')
    return ','.join(libs) if libs else 'matplotlib'


def search_questions(tag: str, page: int = 1, pagesize: int = 100) -> List[Dict]:
    """Search StackOverflow for questions with a tag."""
    params = {
        'order': 'desc',
        'sort': 'votes',
        'tagged': tag,
        'site': SITE,
        'filter': 'withbody',  # Include body
        'page': page,
        'pagesize': pagesize,
    }
    
    try:
        response = requests.get(f"{API_BASE}/questions", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('items', [])
        else:
            logging.warning(f"API error: {response.status_code}")
            return []
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return []


def get_answers(question_id: int) -> List[Dict]:
    """Get answers for a question."""
    params = {
        'order': 'desc',
        'sort': 'votes',
        'site': SITE,
        'filter': 'withbody',
    }
    
    try:
        response = requests.get(f"{API_BASE}/questions/{question_id}/answers", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('items', [])
        else:
            return []
    except:
        return []


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_snippets = []
    seen_hashes = set()
    seen_questions = set()
    
    # Load progress
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, 'r') as f:
            seen_questions = set(json.load(f))
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            all_snippets = json.load(f)
            for s in all_snippets:
                seen_hashes.add(hashlib.md5(s['code'].encode()).hexdigest())
        logging.info(f"Resuming with {len(all_snippets)} existing snippets")
    
    logging.info("Starting StackOverflow extraction...")
    
    for tag in SEARCH_TAGS:
        logging.info(f"\nSearching tag: {tag}")
        
        for page in range(1, 6):  # 5 pages per tag
            questions = search_questions(tag, page=page, pagesize=50)
            
            if not questions:
                logging.info(f"  No more results for {tag}")
                break
            
            for question in tqdm(questions, desc=f"'{tag}' p{page}", leave=False):
                q_id = question.get('question_id')
                
                if q_id in seen_questions:
                    continue
                seen_questions.add(q_id)
                
                # Extract from question body
                q_body = question.get('body', '')
                code_blocks = extract_code_blocks(q_body)
                
                # Get answers and extract from them
                answers = get_answers(q_id)
                for answer in answers:
                    a_body = answer.get('body', '')
                    code_blocks.extend(extract_code_blocks(a_body))
                
                # Process code blocks
                for code in code_blocks:
                    if has_bad_pattern(code):
                        continue
                    
                    # Build complete snippet
                    complete_code = build_complete_snippet(code)
                    
                    # Test it
                    success, error = test_snippet(complete_code)
                    
                    if success:
                        code_hash = hashlib.md5(complete_code.encode()).hexdigest()
                        if code_hash in seen_hashes:
                            continue
                        seen_hashes.add(code_hash)
                        
                        snippet = {
                            'code': complete_code,
                            'caption': question.get('title', ''),
                            'library': detect_library(complete_code),
                            'file_id': f"so_{q_id}_{len(all_snippets)}",
                            'source': f"https://stackoverflow.com/q/{q_id}",
                        }
                        all_snippets.append(snippet)
                
                time.sleep(0.1)  # Be nice to API
            
            # Rate limit: 300 requests per minute
            time.sleep(1)
        
        # Save progress after each tag
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_snippets, f, indent=2, ensure_ascii=False)
        
        with open(SEEN_FILE, 'w') as f:
            json.dump(list(seen_questions), f)
        
        logging.info(f"Progress: {len(all_snippets)} validated snippets from {len(seen_questions)} questions")
        
        # Respect rate limits
        time.sleep(2)
    
    logging.info(f"\n{'='*50}")
    logging.info(f"✅ Extracted {len(all_snippets)} VALIDATED snippets from StackOverflow!")
    logging.info(f"   From {len(seen_questions)} questions")
    logging.info(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
