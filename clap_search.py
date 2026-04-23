"""
CLAP Text-to-Audio Search over ESC-50
Type any description → get back the most matching real audio clips.

First run builds embeddings for all 2000 WAVs (~5-10 min, cached after).
Subsequent runs load from cache and are instant.

Usage:
    python clap_search.py "dark icy texture"
    python clap_search.py "warm crackling fire" --k 3
    python clap_search.py "dog barking outside" --k 5
    python clap_search.py --reindex   (rebuild audio embeddings cache)
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import ClapModel, ClapProcessor
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
MODEL_ID    = "laion/clap-htsat-unfused"
ESC50_DIR   = Path("data/ESC-50-master")
AUDIO_DIR   = ESC50_DIR / "audio"
META_CSV    = ESC50_DIR / "meta" / "esc50.csv"
CACHE_DIR   = Path("outputs/clap_cache")
OUTPUT_DIR  = Path("outputs/clap_search")
CLAP_SR     = 48_000   # CLAP expects 48 kHz


# ── Audio loading ─────────────────────────────────────────
def load_audio_48k(path: str) -> np.ndarray:
    """Load a WAV and resample to 48 kHz mono float32."""
    import torchaudio
    data, sr = sf.read(path, always_2d=True)          # (samples, channels)
    waveform = torch.from_numpy(data.T).float()        # (channels, samples)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != CLAP_SR:
        waveform = torchaudio.functional.resample(waveform, sr, CLAP_SR)
    return waveform.squeeze(0).numpy()   # (samples,)


# ── Load metadata ─────────────────────────────────────────
def load_meta() -> dict[str, dict]:
    """Returns {filename: {category, target}} from esc50.csv"""
    meta = {}
    with open(META_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["filename"]] = {
                "category": row["category"],
                "target"  : int(row["target"]),
            }
    return meta


# ── Build / load embedding cache ─────────────────────────
def build_cache(model: ClapModel, processor: ClapProcessor, device: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"\nBuilding CLAP audio embeddings for {len(wav_paths)} files...")
    print("This takes ~5-10 min on CPU, ~1-2 min on GPU. Cached after first run.\n")

    all_embeddings = []
    filenames      = []
    batch_size     = 16

    for i in tqdm(range(0, len(wav_paths), batch_size), desc="Embedding audio"):
        batch_paths = wav_paths[i : i + batch_size]
        audios      = [load_audio_48k(str(p)) for p in batch_paths]

        inputs = processor(
            audio        = audios,
            sampling_rate= CLAP_SR,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model.audio_model(
                input_features=inputs["input_features"],
                is_longer=inputs["is_longer"],
            )
            emb = model.audio_projection(out.pooler_output)
            emb = torch.nn.functional.normalize(emb, dim=-1)

        all_embeddings.append(emb.cpu().numpy())
        filenames.extend([p.name for p in batch_paths])

    vectors = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.save(str(CACHE_DIR / "audio_vectors.npy"), vectors)

    import json
    with open(CACHE_DIR / "filenames.json", "w") as f:
        json.dump(filenames, f)

    print(f"[clap] Cache saved → {CACHE_DIR}/  shape={vectors.shape}")
    return vectors, filenames


def load_cache():
    import json
    vectors   = np.load(str(CACHE_DIR / "audio_vectors.npy"))
    with open(CACHE_DIR / "filenames.json") as f:
        filenames = json.load(f)
    print(f"[clap] Loaded cache: {len(filenames)} files, dim={vectors.shape[1]}")
    return vectors, filenames


# ── Search ────────────────────────────────────────────────
def search(query: str, vectors: np.ndarray, filenames: list[str],
           meta: dict, model: ClapModel, processor: ClapProcessor,
           device: str, k: int = 5) -> list[dict]:

    t_inputs = processor(text=[query], return_tensors="pt", padding=True)
    t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
    with torch.no_grad():
        out   = model.text_model(
            input_ids=t_inputs["input_ids"],
            attention_mask=t_inputs["attention_mask"],
        )
        q_emb = model.text_projection(out.pooler_output)
        q_emb = torch.nn.functional.normalize(q_emb, dim=-1)

    q_vec  = q_emb.cpu().numpy()[0].astype(np.float32)
    scores = vectors @ q_vec
    top_idx= np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        fname = filenames[idx]
        results.append({
            "rank"    : rank,
            "score"   : float(scores[idx]),
            "filename": fname,
            "category": meta.get(fname, {}).get("category", "?"),
            "path"    : str(AUDIO_DIR / fname),
        })
    return results


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query",    nargs="?", default=None)
    parser.add_argument("--k",      type=int, default=5)
    parser.add_argument("--reindex",action="store_true", help="Rebuild audio embedding cache")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clap] Device: {device}")
    print(f"[clap] Loading CLAP model ({MODEL_ID})...")
    model     = ClapModel.from_pretrained(MODEL_ID).to(device)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # Build or load cache
    cache_exists = (CACHE_DIR / "audio_vectors.npy").exists()
    if args.reindex or not cache_exists:
        vectors, filenames = build_cache(model, processor, device)
    else:
        vectors, filenames = load_cache()

    if args.query is None:
        print("\nUsage: python clap_search.py \"your description here\" --k 5")
        return

    meta    = load_meta()
    results = search(args.query, vectors, filenames, meta, model, processor, device, k=args.k)

    # Copy results to output dir so you can listen
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear previous results
    for f in OUTPUT_DIR.glob("*.wav"):
        f.unlink()

    print(f"\nTop {args.k} results for: \"{args.query}\"\n")
    for r in results:
        print(f"  #{r['rank']}  [{r['score']:.3f}]  {r['category']:<20}  {r['filename']}")
        dest = OUTPUT_DIR / f"{r['rank']:02d}_{r['category']}_{r['filename']}"
        shutil.copy(r["path"], dest)

    print(f"\nFiles copied to {OUTPUT_DIR}/  — open them to listen.")


if __name__ == "__main__":
    main()
