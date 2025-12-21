
import json
import os
import re

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def clean_imports(code, is_main=False):
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove local project imports
        if line.strip().startswith('from .') or line.strip().startswith('from src.'):
            continue
        # Remove relative imports in models
        if line.strip().startswith('from .lnn import') or line.strip().startswith('from .attention import'):
            continue
        # Modify Config path in main/config logic if needed (handled by manually fixing Config class)
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# Paths
base_path = "d:/M.Tech_BITS/Semester 4/Code/Lann/LA-NN-Project"
files = {
    "config": os.path.join(base_path, "src", "config.py"),
    "lnn": os.path.join(base_path, "src", "models", "lnn.py"),
    "attention": os.path.join(base_path, "src", "models", "attention.py"),
    "la_nn": os.path.join(base_path, "src", "models", "la_nn.py"),
    "data": os.path.join(base_path, "src", "data", "preprocessing.py"),
    "trainer": os.path.join(base_path, "src", "training", "trainer.py"),
    "metrics": os.path.join(base_path, "src", "evaluation", "metrics.py"),
    "main": os.path.join(base_path, "main.py")
}

# Read content
content = {}
for key, path in files.items():
    content[key] = read_file(path)

# Custom Fixes
# 1. Config: Fix BASE_DIR for Colab
content["config"] = content["config"].replace(
    "os.path.dirname(os.path.abspath(__file__))", 
    "os.getcwd()"
).replace("from ..", "from .") # unlikely to match but safety

# 2. Main: Remove sys.path modifications and imports
# We will construct a custom Main cell
main_content = content["main"]

# Notebook Structure
cells = []

# Cell 1: Install Dependencies
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install required libraries\n",
        "!pip install wfdb torch numpy scikit-learn tqdm"
    ]
})

# Cell 2: Imports (Global)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import sys\n",
        "import math\n",
        "import time\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "import wfdb\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import classification_report, confusion_matrix\n",
        "from tqdm.notebook import tqdm  # Use notebook version of tqdm\n"
    ]
})

# Cell 3: Configuration
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Configuration"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["config"])]
})

# Cell 4: Models
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## LA-NN Models"]
})

# LNN
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["lnn"])]
})

# Attention
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["attention"])]
})

# LA-NN Assembly
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["la_nn"])]
})

# Cell 5: Data Processing
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Data Pipeline"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["data"]).replace("from tqdm import tqdm", "")]
})

# Cell 6: Trainer
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Trainer"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["trainer"]).replace("from tqdm import tqdm", "")]
})

# Cell 7: Evaluation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Evaluation Metrics"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [clean_imports(content["metrics"])]
})

# Cell 8: Main Execution
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Main Execution"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def main():\n",
        "    print(\"Initializing LA-NN Project on Colab...\")\n",
        "    \n",
        "    # 1. Setup\n",
        "    Config.ensure_dirs()\n",
        "    print(f\"Device: {Config.DEVICE}\")\n",
        "    \n",
        "    # 2. Data Loading\n",
        "    # Force re-download check since Colab is ephemeral\n",
        "    train_loader, val_loader, test_loader = load_data(Config)\n",
        "    \n",
        "    # 3. Model Initialization\n",
        "    model = LANN(Config)\n",
        "    print(\"Model Architecture:\")\n",
        "    # print(model) # Optional: comment out to save space\n",
        "    \n",
        "    # Count parameters\n",
        "    total_params = sum(p.numel() for p in model.parameters())\n",
        "    print(f\"Total Trainable Parameters: {total_params}\")\n",
        "    \n",
        "    # 4. Training\n",
        "    trainer = Trainer(model, Config, train_loader, val_loader)\n",
        "    trainer.train()\n",
        "    \n",
        "    # 5. Evaluation (Load best model)\n",
        "    print(\"\\nLoading best model for evaluation...\")\n",
        "    if os.path.exists(f\"{Config.MODELS_DIR}/la_nn_best.pth\"):\n",
        "        model.load_state_dict(torch.load(f\"{Config.MODELS_DIR}/la_nn_best.pth\", map_location=Config.DEVICE))\n",
        "    else:\n",
        "        print(\"Warning: Model file not found (maybe training didn't finish?), using current weights.\")\n",
        "        \n",
        "    evaluate_model(model, test_loader, Config)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
    ]
})

# Create Notebook Dictionary
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write to file
with open("d:/M.Tech_BITS/Semester 4/Code/Lann/LA-NN-Project/LA_NN_Colab.ipynb", "w") as f:
    json.dump(notebook_json, f, indent=2)

print("Notebook generated successfully.")
