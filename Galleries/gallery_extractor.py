"""
Official Gallery Extractor

Extracts examples from official matplotlib and seaborn galleries.
These are guaranteed to work since they're from official documentation.
"""

import os
import re
import json
import time
import signal
import logging
import hashlib
from typing import List, Dict, Tuple
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

# Set matplotlib backend before importing
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
OUTPUT_DIR = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/Galleries"
OUTPUT_FILE = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/Galleries/gallery_validated.json"

# Gallery URLs
MATPLOTLIB_GALLERY_BASE = "https://matplotlib.org/stable/gallery"
MATPLOTLIB_RAW_BASE = "https://raw.githubusercontent.com/matplotlib/matplotlib/main/galleries/examples"

SEABORN_EXAMPLES_BASE = "https://seaborn.pydata.org/examples"
SEABORN_RAW_BASE = "https://raw.githubusercontent.com/mwaskom/seaborn/master/examples"

# Standard imports to prepend
STANDARD_IMPORTS = """import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
"""

# Categories in matplotlib gallery
MATPLOTLIB_CATEGORIES = [
    "lines_bars_and_markers",
    "images_contours_and_fields",
    "subplots_axes_and_figures",
    "statistics",
    "pie_and_polar_charts",
    "text_labels_and_annotations",
    "color",
    "shapes_and_collections",
    "style_sheets",
    "axes_grid1",
    "axisartist",
    "showcase",
    "specialty_plots",
    "spines",
    "ticks",
    "scales",
    "mplot3d",
]


def test_snippet(code: str, timeout: float = 10.0) -> Tuple[bool, str]:
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
        return False, str(e)[:100]
    finally:
        signal.alarm(0)


def clean_matplotlib_code(code: str) -> str:
    """Clean matplotlib example code."""
    lines = code.split('\n')
    cleaned = []
    skip_block = False
    
    for line in lines:
        # Skip sphinx gallery directives
        if line.strip().startswith('# %%') or line.strip().startswith('# sphinx'):
            continue
        if '.. ' in line and line.strip().startswith('#'):
            continue
        
        # Skip matplotlib.sphinxext imports
        if 'sphinxext' in line:
            continue
        
        # Skip __doc__ assignments
        if '__doc__' in line:
            continue
            
        cleaned.append(line)
    
    return '\n'.join(cleaned)


def build_complete_snippet(code: str, library: str = "matplotlib") -> str:
    """Build a complete, runnable snippet."""
    # Check if code already has imports
    has_plt_import = 'import matplotlib' in code or 'from matplotlib' in code
    has_np_import = 'import numpy' in code
    
    parts = []
    
    # Add backend setup
    parts.append("import matplotlib\nmatplotlib.use('Agg')\nimport warnings\nwarnings.filterwarnings('ignore')")
    
    if not has_np_import:
        parts.append("import numpy as np")
    
    if not has_plt_import:
        parts.append("import matplotlib.pyplot as plt")
    
    if library == "seaborn" and 'import seaborn' not in code:
        parts.append("import seaborn as sns")
    
    parts.append(code)
    
    full_code = '\n'.join(parts)
    
    if 'plt.show()' not in full_code:
        full_code += '\nplt.show()'
    
    return full_code


def get_matplotlib_example_list(category: str) -> List[str]:
    """Get list of example files from matplotlib gallery category."""
    examples = []
    
    # Try to get the index page
    index_url = f"{MATPLOTLIB_GALLERY_BASE}/{category}/index.html"
    
    try:
        resp = requests.get(index_url, timeout=30)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find all example links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.html') and not href.startswith('http') and 'index' not in href:
                # Extract example name
                example_name = href.replace('.html', '').split('/')[-1]
                examples.append(example_name)
        
    except Exception as e:
        logging.warning(f"Failed to get {category} index: {e}")
    
    return examples


def download_matplotlib_example(category: str, example_name: str) -> Tuple[str, str]:
    """Download matplotlib example source code."""
    # Try raw GitHub URL
    raw_url = f"{MATPLOTLIB_RAW_BASE}/{category}/{example_name}.py"
    
    try:
        resp = requests.get(raw_url, timeout=30)
        if resp.status_code == 200:
            code = resp.text
            
            # Extract docstring as caption
            caption = ""
            if '"""' in code:
                match = re.search(r'"""(.*?)"""', code, re.DOTALL)
                if match:
                    caption = match.group(1).strip()[:200]
            
            return code, caption
    except:
        pass
    
    return "", ""


def get_seaborn_examples() -> List[Dict]:
    """Get list of seaborn examples."""
    examples = []
    
    # Known seaborn examples (from their gallery)
    seaborn_examples = [
        "anscombes_quartet",
        "different_scatter_variables",
        "errorband_lineplots",
        "faceted_lineplot",
        "grouped_barplot",
        "grouped_boxplot",
        "grouped_violinplots",
        "heat_scatter",
        "hexbin_marginals",
        "histogram_stacked",
        "horizontal_boxplot",
        "jitter_stripplot",
        "joint_histogram",
        "joint_kde",
        "kde_ridgeplot",
        "layered_bivariate_plot",
        "logistic_regression",
        "many_facets",
        "many_pairwise_correlations",
        "marginal_ticks",
        "multiple_bivariate_kde",
        "multiple_conditional_kde",
        "multiple_ecdf",
        "multiple_regression",
        "pair_grid_with_kde",
        "pairgrid_dotplot",
        "paired_pointplots",
        "palette_choices",
        "palette_generation",
        "part_whole_bars",
        "pointplot_anova",
        "radial_facets",
        "regression_marginals",
        "residplot",
        "scatter_bubbles",
        "scatterplot_categorical",
        "scatterplot_matrix",
        "scatterplot_sizes",
        "simple_violinplots",
        "smooth_bivariate_kde",
        "spreadsheet_heatmap",
        "strip_regplot",
        "structured_heatmap",
        "three_variable_histogram",
        "timeseries_facets",
        "wide_data_lineplot",
        "wide_form_violinplot",
    ]
    
    for name in seaborn_examples:
        examples.append({'name': name, 'url': f"{SEABORN_RAW_BASE}/{name}.py"})
    
    return examples


def download_seaborn_example(example: Dict) -> Tuple[str, str]:
    """Download seaborn example source code."""
    try:
        resp = requests.get(example['url'], timeout=30)
        if resp.status_code == 200:
            code = resp.text
            
            # Extract docstring as caption
            caption = ""
            if '"""' in code:
                match = re.search(r'"""(.*?)"""', code, re.DOTALL)
                if match:
                    caption = match.group(1).strip()[:200]
            
            return code, caption
    except:
        pass
    
    return "", ""


def detect_library(code: str) -> str:
    """Detect visualization library."""
    libs = []
    if 'plt.' in code or 'matplotlib' in code:
        libs.append('matplotlib')
    if 'sns.' in code or 'seaborn' in code:
        libs.append('seaborn')
    return ','.join(libs) if libs else 'matplotlib'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_snippets = []
    seen_hashes = set()
    
    # Load existing
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            all_snippets = json.load(f)
            for s in all_snippets:
                seen_hashes.add(hashlib.md5(s['code'].encode()).hexdigest())
        logging.info(f"Resuming with {len(all_snippets)} existing snippets")
    
    # ========== MATPLOTLIB GALLERY ==========
    logging.info("\n" + "="*50)
    logging.info("Extracting from Matplotlib Gallery...")
    logging.info("="*50)
    
    for category in tqdm(MATPLOTLIB_CATEGORIES, desc="Categories"):
        examples = get_matplotlib_example_list(category)
        
        for example_name in examples:
            code, caption = download_matplotlib_example(category, example_name)
            
            if not code:
                continue
            
            # Clean the code
            code = clean_matplotlib_code(code)
            
            # Build complete snippet
            complete_code = build_complete_snippet(code, "matplotlib")
            
            # Test it
            success, error = test_snippet(complete_code)
            
            if success:
                code_hash = hashlib.md5(complete_code.encode()).hexdigest()
                if code_hash in seen_hashes:
                    continue
                seen_hashes.add(code_hash)
                
                snippet = {
                    'code': complete_code,
                    'caption': caption,
                    'library': 'matplotlib',
                    'file_id': f"mpl_{category}_{example_name}",
                    'source': f"https://matplotlib.org/stable/gallery/{category}/{example_name}.html",
                }
                all_snippets.append(snippet)
            
            time.sleep(0.1)
    
    logging.info(f"Matplotlib: {len([s for s in all_snippets if s['library'] == 'matplotlib'])} snippets")
    
    # ========== SEABORN GALLERY ==========
    logging.info("\n" + "="*50)
    logging.info("Extracting from Seaborn Gallery...")
    logging.info("="*50)
    
    seaborn_examples = get_seaborn_examples()
    
    for example in tqdm(seaborn_examples, desc="Seaborn"):
        code, caption = download_seaborn_example(example)
        
        if not code:
            continue
        
        # Build complete snippet
        complete_code = build_complete_snippet(code, "seaborn")
        
        # Test it
        success, error = test_snippet(complete_code)
        
        if success:
            code_hash = hashlib.md5(complete_code.encode()).hexdigest()
            if code_hash in seen_hashes:
                continue
            seen_hashes.add(code_hash)
            
            snippet = {
                'code': complete_code,
                'caption': caption,
                'library': detect_library(complete_code),
                'file_id': f"sns_{example['name']}",
                'source': f"https://seaborn.pydata.org/examples/{example['name']}.html",
            }
            all_snippets.append(snippet)
        
        time.sleep(0.1)
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_snippets, f, indent=2, ensure_ascii=False)
    
    mpl_count = len([s for s in all_snippets if 'mpl_' in s['file_id']])
    sns_count = len([s for s in all_snippets if 'sns_' in s['file_id']])
    
    logging.info(f"\n{'='*50}")
    logging.info(f"✅ Extracted {len(all_snippets)} VALIDATED snippets!")
    logging.info(f"   Matplotlib Gallery: {mpl_count}")
    logging.info(f"   Seaborn Gallery: {sns_count}")
    logging.info(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
