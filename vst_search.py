"""
VST Search Helper — called by the PresetPlayer VST as a subprocess.
Usage: python vst_search.py "dark icy drone" [k]
Outputs: JSON array of results to stdout.
"""

import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

PYTHON_PATH   = "C:/AI/audio_ml/venv/Scripts/python.exe"
LIBRARY_FILE  = "C:/AI/audio_ml/outputs/player_library.json"
MODEL_ID      = "laion/clap-htsat-unfused"

def main():
    if len(sys.argv) < 2:
        print("[]")
        return

    query = sys.argv[1]
    k     = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Load library
    with open(LIBRARY_FILE) as f:
        lib = json.load(f)

    lib_vectors    = np.array(lib["vectors"],    dtype=np.float32)
    lib_filenames  = lib["filenames"]
    lib_categories = lib["categories"]

    # Load CLAP
    import warnings
    warnings.filterwarnings("ignore")
    model     = ClapModel.from_pretrained(MODEL_ID)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Embed query
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {kk: v.to(device) for kk, v in inputs.items()}
    with torch.no_grad():
        out   = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        q = F.normalize(model.text_projection(out.pooler_output), dim=-1)
        q = q.cpu().numpy()[0].astype(np.float32)

    # Search
    scores  = lib_vectors @ q
    top_idx = np.argsort(scores)[::-1][:k]

    results = [
        {
            "category": lib_categories[i],
            "path"    : lib_filenames[i].replace("\\", "/"),
            "score"   : float(scores[i]),
        }
        for i in top_idx
    ]

    print(json.dumps(results))


if __name__ == "__main__":
    main()
