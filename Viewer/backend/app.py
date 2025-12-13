import os
import io
import json
import base64
import hashlib
import traceback
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)


@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# Configuration
DATA_FILE = "/Users/saeedadeeb/Documents/projects/NLGAD/PyViz/all_validated_snippets.json"
CACHE_DIR = Path("render_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Load data
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

DATA = None

def get_data():
    global DATA
    if DATA is None:
        DATA = load_data()
    return DATA


@app.route("/api/count")
def get_count():
    """Get total number of samples."""
    return jsonify({"count": len(get_data())})


@app.route("/api/samples")
def get_samples():
    """Get paginated samples (lazy loading)."""
    data = get_data()
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 10, type=int)
    
    samples = []
    for i, item in enumerate(data[offset:offset + limit], start=offset):
        samples.append({
            "id": i,
            "code": item.get("code", ""),
            "caption": item.get("caption", ""),
            "library": item.get("library", ""),
            "file_id": item.get("file_id", f"sample_{i}")
        })
    
    return jsonify({
        "samples": samples,
        "total": len(data),
        "offset": offset,
        "limit": limit
    })


@app.route("/api/render/<int:sample_id>")
def render_sample(sample_id):
    """Render a specific sample and return the image."""
    data = get_data()
    
    if sample_id < 0 or sample_id >= len(data):
        return jsonify({"error": "Sample not found"}), 404
    
    item = data[sample_id]
    code = item.get("code", "")
    
    # Check cache first
    code_hash = hashlib.md5(code.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{code_hash}.png"
    
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        return jsonify({"image": img_base64, "cached": True})
    
    # Try to render
    try:
        img_base64 = render_code(code)
        if img_base64:
            # Save to cache
            with open(cache_path, "wb") as f:
                f.write(base64.b64decode(img_base64))
            return jsonify({"image": img_base64, "cached": False})
        else:
            return jsonify({"error": "Failed to render", "details": "No output generated"}), 500
    except Exception as e:
        return jsonify({"error": "Render failed", "details": str(e)}), 500


def render_code(code):
    """Execute Python code and capture the plot as base64 image."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Clear any existing figures
    plt.close('all')
    
    # Create a restricted execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'np': None,
        'pd': None,
        'plt': None,
        'sns': None,
        'px': None,
        'go': None,
    }
    
    # Import libraries
    try:
        import numpy as np
        exec_globals['np'] = np
    except ImportError:
        pass
    
    try:
        import pandas as pd
        exec_globals['pd'] = pd
    except ImportError:
        pass
    
    try:
        import matplotlib.pyplot as plt
        exec_globals['plt'] = plt
    except ImportError:
        pass
    
    try:
        import seaborn as sns
        exec_globals['sns'] = sns
    except ImportError:
        pass
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        exec_globals['px'] = px
        exec_globals['go'] = go
    except ImportError:
        pass
    
    # Modify code to not show or save
    modified_code = code.replace('plt.show()', '').replace('fig.show()', '')
    
    # Remove file operations for safety
    lines = []
    for line in modified_code.split('\n'):
        if 'savefig' in line or 'to_file' in line or 'write_image' in line:
            continue
        if 'open(' in line and ('w' in line or 'write' in line):
            continue
        lines.append(line)
    modified_code = '\n'.join(lines)
    
    try:
        exec(modified_code, exec_globals)
        
        # Check if there's a matplotlib figure
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close('all')
            return img_base64
        
        # Check for plotly figure
        if 'fig' in exec_globals and exec_globals['fig'] is not None:
            try:
                fig = exec_globals['fig']
                if hasattr(fig, 'to_image'):
                    img_bytes = fig.to_image(format='png')
                    return base64.b64encode(img_bytes).decode()
            except:
                pass
        
        return None
        
    except Exception as e:
        print(f"Render error: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print(f"Loading data from {DATA_FILE}...")
    print(f"Total samples: {len(get_data())}")
    app.run(debug=True, port=5001)
