"""
Preset Player — CLAP Audio Search
Run from CLI:
    python preset_player.py

On first run: CLAP selects the top pad-like sounds from ESC-50 as the library.
Then: type a prompt, press Search (or Enter), top match plays automatically.
"""

import json
import threading
import tkinter as tk
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

# ── Config ────────────────────────────────────────────────
ESC50_AUDIO  = Path("data/ESC-50-master/audio")
CACHE_DIR    = Path("outputs/clap_cache")
LIBRARY_FILE = Path("outputs/player_library.json")
LIBRARY_SIZE = 100       # top N pad-like sounds to keep as library
LIBRARY_QUERY = "sustained ambient pad texture drone"

MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SR  = 48_000


# ── CLAP engine (decoupled — easy to port to VST later) ───
class ClapEngine:
    def __init__(self):
        print("Loading CLAP model...")
        self.model     = ClapModel.from_pretrained(MODEL_ID)
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()
        self.device = device
        print(f"CLAP ready on {device}")

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out  = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            emb = self.model.text_projection(out.pooler_output)
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().numpy()[0].astype(np.float32)

    def search(self, query: str, vectors: np.ndarray, filenames: list, categories: list, k: int = 5):
        q = self.embed_text(query)
        scores = vectors @ q
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {"rank": i + 1, "score": float(scores[idx]), "path": filenames[idx], "category": categories[idx]}
            for i, idx in enumerate(top_idx)
        ]


# ── Library builder ───────────────────────────────────────
def build_library(engine: ClapEngine):
    """Select top LIBRARY_SIZE pad-like sounds from ESC-50 using CLAP."""
    vectors   = np.load(str(CACHE_DIR / "audio_vectors.npy"))
    with open(CACHE_DIR / "filenames.json") as f:
        all_filenames = json.load(f)

    print(f"Selecting top {LIBRARY_SIZE} pad-like sounds...")
    q = engine.embed_text(LIBRARY_QUERY)
    scores    = vectors @ q
    top_idx   = np.argsort(scores)[::-1][:LIBRARY_SIZE]

    # Load category metadata
    import csv
    meta = {}
    with open("data/ESC-50-master/meta/esc50.csv", newline="") as f:
        for row in csv.DictReader(f):
            meta[row["filename"]] = row["category"]

    library = {
        "filenames"  : [str(ESC50_AUDIO / all_filenames[i]) for i in top_idx],
        "categories" : [meta.get(all_filenames[i], "unknown") for i in top_idx],
        "vectors"    : vectors[top_idx].tolist(),
        "scores"     : [float(scores[i]) for i in top_idx],
    }
    LIBRARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LIBRARY_FILE, "w") as f:
        json.dump(library, f)
    print(f"Library saved: {len(library['filenames'])} presets")
    return (
        np.array(library["vectors"], dtype=np.float32),
        library["filenames"],
        library["categories"],
    )


def load_library():
    with open(LIBRARY_FILE) as f:
        lib = json.load(f)
    return (
        np.array(lib["vectors"], dtype=np.float32),
        lib["filenames"],
        lib.get("categories", [Path(f).stem for f in lib["filenames"]]),
    )


# ── Audio playback ────────────────────────────────────────
class Player:
    def __init__(self):
        self._playing = False

    def play(self, path: str):
        self.stop()
        self._playing = True
        def _run():
            data, sr = sf.read(path, always_2d=True)
            sd.play(data, sr)
            sd.wait()
            self._playing = False
        threading.Thread(target=_run, daemon=True).start()

    def stop(self):
        sd.stop()
        self._playing = False

    @property
    def playing(self):
        return self._playing


# ── GUI ───────────────────────────────────────────────────
class PresetPlayerApp:
    def __init__(self, root, engine, vectors, filenames, categories):
        self.root       = root
        self.engine     = engine
        self.vectors    = vectors
        self.filenames  = filenames
        self.categories = categories
        self.player     = Player()
        self.results    = []

        root.title("Preset Player")
        root.resizable(False, False)
        root.configure(bg="#1e1e1e")

        # ── Prompt row ────────────────────────────────────
        frame_top = tk.Frame(root, bg="#1e1e1e")
        frame_top.pack(padx=12, pady=(12, 4), fill="x")

        self.entry = tk.Entry(
            frame_top, width=36, font=("Segoe UI", 11),
            bg="#2d2d2d", fg="#f0f0f0", insertbackground="white",
            relief="flat", bd=4,
        )
        self.entry.pack(side="left", ipady=4)
        self.entry.bind("<Return>", lambda e: self._search())
        self.entry.focus()

        self.btn_search = tk.Button(
            frame_top, text="Search", command=self._search,
            bg="#0078d4", fg="white", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=10, cursor="hand2",
        )
        self.btn_search.pack(side="left", padx=(6, 0))

        # ── Results list ──────────────────────────────────
        self.listbox = tk.Listbox(
            root, width=52, height=5, font=("Segoe UI", 10),
            bg="#2d2d2d", fg="#d0d0d0", selectbackground="#0078d4",
            selectforeground="white", relief="flat", bd=0,
            activestyle="none",
        )
        self.listbox.pack(padx=12, pady=4, fill="x")
        self.listbox.bind("<Double-Button-1>", lambda e: self._play_selected())

        # ── Play / Stop row ───────────────────────────────
        frame_bot = tk.Frame(root, bg="#1e1e1e")
        frame_bot.pack(padx=12, pady=(4, 12), fill="x")

        self.btn_play = tk.Button(
            frame_bot, text="▶  Play", command=self._play_selected,
            bg="#107c10", fg="white", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, cursor="hand2", state="disabled",
        )
        self.btn_play.pack(side="left")

        self.btn_stop = tk.Button(
            frame_bot, text="■  Stop", command=self._stop,
            bg="#c42b1c", fg="white", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, cursor="hand2",
        )
        self.btn_stop.pack(side="left", padx=(6, 0))

        self.status = tk.Label(
            frame_bot, text="Ready", bg="#1e1e1e", fg="#888",
            font=("Segoe UI", 9),
        )
        self.status.pack(side="left", padx=(12, 0))

    def _search(self):
        query = self.entry.get().strip()
        if not query:
            return
        self.status.config(text="Searching...")
        self.root.update()

        self.results = self.engine.search(query, self.vectors, self.filenames, self.categories, k=5)

        self.listbox.delete(0, "end")
        for r in self.results:
            self.listbox.insert("end", f"  [{r['score']:.3f}]  {r['category']}")

        if self.results:
            self.listbox.selection_set(0)
            self.btn_play.config(state="normal")
            self.status.config(text=f"Found {len(self.results)} results")
            self._play_selected()

    def _play_selected(self):
        sel = self.listbox.curselection()
        if not sel or not self.results:
            return
        idx  = sel[0]
        r    = self.results[idx]
        self.player.play(r["path"])
        self.status.config(text=f"Playing: {r['category']}  [{r['score']:.3f}]")

    def _stop(self):
        self.player.stop()
        self.status.config(text="Stopped")


# ── Main ──────────────────────────────────────────────────
def main():
    engine = ClapEngine()

    if LIBRARY_FILE.exists():
        print("Loading library from cache...")
        vectors, filenames, categories = load_library()
    else:
        if not (CACHE_DIR / "audio_vectors.npy").exists():
            print("ERROR: Run clap_search.py first to build the ESC-50 embedding cache.")
            return
        vectors, filenames, categories = build_library(engine)

    print(f"Library ready: {len(filenames)} presets")
    print("Opening player window...")

    root = tk.Tk()
    app  = PresetPlayerApp(root, engine, vectors, filenames, categories)
    root.mainloop()


if __name__ == "__main__":
    main()
