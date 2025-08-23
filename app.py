# app.py
# -----------------------------
# Hybrid RAG (Structured + Unstructured)
# LlamaIndex (0.13.x) + FAISS + MiniLM + Ollama
# + Manifest-based incremental ingest + Coverage verification
# -----------------------------

from pathlib import Path
from typing import List, Optional, Callable, Dict
import json, hashlib, time

# LlamaIndex - Core
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

# LLM via Ollama
from llama_index.llms.ollama import Ollama

# Embeddings via HuggingFace (MiniLM)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FAISS
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# Structured data
import pandas as pd


# =============================
# Paths
# =============================
BASE_DIR = Path().resolve()        # In Jupyter/VS Code: current directory
DATA_DIR = BASE_DIR / "data"
STRUCTURED_DIR = DATA_DIR / "structured"      # Put CSV/XLSX here
UNSTRUCTURED_DIR = DATA_DIR / "unstructured"  # Put PDF/TXT/DOCX/MD here
STORAGE_DIR = BASE_DIR / "storage"            # Storage will be saved here

STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
UNSTRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Use a single, consistent index id to avoid multiple indices in the same storage
INDEX_ID = "main"

# Manifest path
MANIFEST = STORAGE_DIR / "ingest_manifest.json"


# =============================
# Settings (LLM + Embeddings + Chunking)
# =============================
Settings.llm = Ollama(
    model="deepseek-r1:7b",  # Use the smaller base model
    request_timeout=120.0,
    temperature=0.1,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",  # Run on CPU to save resources
)

# Text splitter (can be tuned later)
# Use a larger chunk size to avoid metadata dominating small chunks
splitter = SentenceSplitter(chunk_size=800, chunk_overlap=100)


# =============================
# Manifest helpers (fingerprint + read/write)
# =============================

def file_fingerprint(p: Path) -> str:
    """Stable fingerprint based on absolute path, size, and mtime."""
    h = hashlib.sha256()
    st = p.stat()
    h.update(str(p.resolve()).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def load_manifest() -> Dict[str, dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: Dict[str, dict]) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================
# Structured (CSV/XLSX) helpers
# =============================

def _df_to_text(df: pd.DataFrame, max_rows: int = 2000) -> str:
    """Convert DataFrame to display text. Prefer Markdown; fall back to CSV.
    Truncates to max_rows for text ONLY (metadata keeps full sizes).
    """
    truncated = False
    if len(df) > max_rows:
        df = df.head(max_rows).copy()
        truncated = True

    try:
        text_md = df.to_markdown(index=False)  # may need 'tabulate'
    except Exception:
        text_md = None

    if text_md is None:
        text = df.to_csv(index=False, na_rep="")
    else:
        text = text_md

    if truncated:
        text += f"\n\n... [truncated to first {max_rows} rows for display]"
    return text


def _table_row_dump(df: pd.DataFrame, max_rows: int = 400) -> str:
    """Convert rows into 'col: value' style lines, up to max_rows."""
    rows = []
    for _, row in df.head(max_rows).iterrows():
        parts = []
        for c, v in row.items():
            if v is None or str(v).strip() == "":
                continue
            parts.append(f"{c}: {v}")
        if parts:
            rows.append(" ; ".join(parts))
    return "\n".join(rows)


def _read_csv_robust(file_path: Path) -> pd.DataFrame:
    """Read messy CSV: skip leading empty rows, detect one/two header rows, and build meaningful column names."""
    raw = pd.read_csv(file_path, header=None, sep=None, engine="python", dtype=str)
    raw = raw.replace({pd.NA: None})

    def row_nonempty_count(r):
        return sum(1 for x in r if x is not None and str(x).strip() != "")

    # Find first non-empty row
    idx_first_data = 0
    while idx_first_data < len(raw) and row_nonempty_count(raw.iloc[idx_first_data]) == 0:
        idx_first_data += 1

    def is_header_like(r):
        cnt = row_nonempty_count(r)
        return cnt >= 3

    h1 = idx_first_data
    h2 = h1 + 1 if h1 + 1 < len(raw) else None
    use_two_headers = False
    if is_header_like(raw.iloc[h1]) and h2 is not None and is_header_like(raw.iloc[h2]):
        use_two_headers = True
    elif not is_header_like(raw.iloc[h1]) and h2 is not None and is_header_like(raw.iloc[h2]):
        h1, h2 = h2, (h2 + 1 if h2 + 1 < len(raw) else None)
        use_two_headers = False

    # Build column names
    if use_two_headers:
        top = raw.iloc[h1].fillna("").astype(str).str.strip().tolist()
        bot = raw.iloc[h2].fillna("").astype(str).str.strip().tolist()
        cols = []
        for a, b in zip(top, bot):
            name = " - ".join([x for x in (a, b) if x])
            cols.append(name if name else "col")
        data_start = h2 + 1
    else:
        hdr = raw.iloc[h1].fillna("").astype(str).str.strip().tolist()
        cols = [c if c else "col" for c in hdr]
        data_start = h1 + 1

    df = raw.iloc[data_start:].copy()
    df.columns = cols[: df.shape[1]]
    df = df.dropna(how="all")
    # Pandas deprecation: applymap -> column-wise map
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    return df.reset_index(drop=True)


def _open_excel_robust(file_path: Path) -> pd.ExcelFile:
    """Open Excel file using appropriate engine with graceful fallback."""
    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        try:
            return pd.ExcelFile(file_path, engine="openpyxl")
        except Exception:
            return pd.ExcelFile(file_path)
    elif suffix == ".xls":
        try:
            return pd.ExcelFile(file_path, engine="xlrd")
        except Exception:
            return pd.ExcelFile(file_path)
    return pd.ExcelFile(file_path)


def load_structured_documents(file_path: Path) -> List[Document]:
    """Load CSV or XLS/XLSX file(s) into Documents with rich metadata."""
    docs: List[Document] = []
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = _read_csv_robust(file_path)
        text_table = _df_to_text(df)
        if str(text_table).strip():
            cols = list(map(str, df.columns))
            dtypes = {str(k): str(v) for k, v in df.dtypes.items()}
            header_lines = [
                f"[CSV] file={file_path.name} | rows={df.shape[0]} | cols={df.shape[1]}",
                "Columns: " + (", ".join(cols) if cols else "(none)"),
                "",
            ]
            row_dump = _table_row_dump(df)
            full_text = "\n".join(header_lines) + text_table + "\n\n-- ROW DUMP --\n" + row_dump

            metadata = {
                "type": "structured_csv",
                "source": file_path.name,
                "source_path": str(file_path),
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "columns": cols,
                "dtypes": dtypes,
                "delimiter_auto": True,
            }
            docs.append(Document(text=full_text, metadata=metadata))

    elif suffix in (".xlsx", ".xls"):
        xls = _open_excel_robust(file_path)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name=sheet_name)
            if df.shape[0] == 0 and df.shape[1] == 0:
                header_lines = [
                    f"[EXCEL] file={file_path.name} | sheet={sheet_name} | rows=0 | cols=0",
                    "Columns: (none)",
                    "",
                    "(empty sheet)",
                ]
                docs.append(
                    Document(
                        text="\n".join(header_lines),
                        metadata={
                            "type": "structured_excel_sheet",
                            "source": file_path.name,
                            "source_path": str(file_path),
                            "sheet_name": sheet_name,
                            "n_rows": 0,
                            "n_cols": 0,
                            "columns": [],
                            "dtypes": {},
                        },
                    )
                )
                continue

            text_table = _df_to_text(df)
            if not str(text_table).strip():
                continue

            cols = list(map(str, df.columns))
            dtypes = {str(k): str(v) for k, v in df.dtypes.items()}
            header_lines = [
                f"[EXCEL] file={file_path.name} | sheet={sheet_name} | rows={df.shape[0]} | cols={df.shape[1]}",
                "Columns: " + (", ".join(cols) if cols else "(none)"),
                "",
            ]
            full_text = "\n".join(header_lines) + text_table
            metadata = {
                "type": "structured_excel_sheet",
                "source": file_path.name,
                "source_path": str(file_path),
                "sheet_name": sheet_name,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "columns": cols,
                "dtypes": dtypes,
            }
            docs.append(Document(text=full_text, metadata=metadata))

    return docs


def load_all_structured_docs() -> List[Document]:
    """Load all structured files (CSV + XLS/XLSX) from STRUCTURED_DIR."""
    all_docs: List[Document] = []
    for file in STRUCTURED_DIR.glob("*.*"):
        if file.suffix.lower() in (".csv", ".xlsx", ".xls"):
            all_docs.extend(load_structured_documents(file))
    return all_docs


# =============================
# Unstructured loaders
# =============================

def load_unstructured_docs() -> List[Document]:
    """Load PDF/TXT/DOCX/MD files as unstructured docs.
    Robust to uppercase extensions (e.g., .PDF) by pre-building the file list.
    """
    if not UNSTRUCTURED_DIR.exists():
        return []

    allowed = {".pdf", ".txt", ".md", ".docx"}
    files = [p for p in UNSTRUCTURED_DIR.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    if not files:
        return []

    # Pass explicit file list to avoid case-sensitive extension filtering
    reader = SimpleDirectoryReader(
        input_files=[str(p) for p in files],
        filename_as_id=True,
    )
    docs = reader.load_data()

    out: List[Document] = []
    for d in docs:
        md = dict(d.metadata) if d.metadata else {}
        md.setdefault("type", "unstructured_doc")
        # Prefer full path if provided by the reader
        file_path = md.get("file_path")
        if file_path:
            md["source_path"] = str(file_path)
            md["source"] = Path(file_path).name
        else:
            src = md.get("source") or md.get("file_name") or md.get("filename") or md.get("id")
            if src:
                md["source"] = str(src)
        out.append(Document(text=d.text, metadata=md))
    return out


# =============================
# Build / Persist index (incremental via manifest)
# =============================

def _resolve_source_to_path(md: dict) -> Optional[Path]:
    """Resolve a file path from document metadata. Prefer 'source_path'; fallback to searching by name."""
    sp = md.get("source_path")
    if sp and Path(sp).exists():
        return Path(sp)
    name = md.get("source")
    if not name:
        return None
    # Search by name under both structured and unstructured dirs
    candidates = list(STRUCTURED_DIR.rglob(name)) + list(UNSTRUCTURED_DIR.rglob(name))
    return candidates[0] if candidates else None


def build_or_update_index(persist_dir: Path = STORAGE_DIR) -> VectorStoreIndex:
    """Build/Update a FAISS index from (Structured CSV/XLSX + Unstructured) incrementally using a manifest.
    - Fresh storage => full reindex. Existing storage => only new/changed docs via fingerprint manifest.
    """
    # Clear storage if it exists
    if persist_dir.exists():
        for item in persist_dir.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
    # 1) Collect documents
    u_docs = load_unstructured_docs()
    s_docs = load_all_structured_docs()
    all_docs = u_docs + s_docs
    if not all_docs:
        raise RuntimeError(
            "No documents available for indexing.\n"
            f"- Put CSV/XLSX files in: {STRUCTURED_DIR}\n"
            f"- Or PDF/TXT/DOCX/MD files in: {UNSTRUCTURED_DIR}\n"
        )

    # 2) Prepare vector store + storage context
    print("Creating fresh storage...")
    # Detect embedding dimension for a fresh FAISS index
    probe_vec = Settings.embed_model.get_query_embedding("dimension probe")
    embed_dim = len(probe_vec)
    print(f"Using embedding dimension: {embed_dim}")
    faiss_idx = faiss.IndexFlatL2(embed_dim)
    vs = FaissVectorStore(faiss_index=faiss_idx)
    storage_context = StorageContext.from_defaults(vector_store=vs)
    print("Storage initialized with empty FAISS index.")
    print("⚠️ Creating new index...")

    # Create the index from all documents
    index = VectorStoreIndex.from_documents(
        documents=all_docs,
        storage_context=storage_context,
        show_progress=True,
        use_async=False  # Ensure synchronous operation for stability
    )

    # 3) Manifest-driven selection of changed/new files
    manifest = load_manifest()
    to_index: List[Document] = []

    print(f"\nProcessing {len(all_docs)} documents...")

    # Update manifest for the indexed docs
    for doc in all_docs:
        md = getattr(doc, "metadata", {}) or {}
        src_path = _resolve_source_to_path(md)
        if src_path and src_path.exists():
            key = str(src_path.resolve())
            manifest[key] = {
                "fingerprint": file_fingerprint(src_path),
                "last_indexed": int(time.time()),
            }

    # Save the manifest with updated fingerprints
    save_manifest(manifest)

    # Persist the index
    storage_context.persist(persist_dir=str(persist_dir))
    print("✓ Index built and persisted.")
    return index


# =============================
# Load from storage + Query
# =============================

def _load_index(persist_dir: Path = STORAGE_DIR) -> VectorStoreIndex:
    """Load index from storage using the fixed INDEX_ID."""
    if not (persist_dir.exists() and any(persist_dir.iterdir())):
        raise RuntimeError("No existing storage found. Run build_or_update_index() first.")

    # Load Vector Store from disk
    vs = FaissVectorStore.from_persist_dir(str(persist_dir))

    # StorageContext points to same folder
    storage_context = StorageContext.from_defaults(
        vector_store=vs,
        persist_dir=str(persist_dir),
    )

    # Load existing index
    return load_index_from_storage(storage_context)


def ensure_index(persist_dir: Path = STORAGE_DIR) -> VectorStoreIndex:
    """If no storage exists, build first; then return loaded index."""
    if not (persist_dir.exists() and any(persist_dir.iterdir())):
        build_or_update_index(persist_dir)
    return _load_index(persist_dir)


# =============================
# Query helpers
# =============================

def ask_one(
    query: str,
    top_k: int = 6,
    include_sources: bool = True,
    restrict: Optional[str] = None,           # None | "excel" | "csv" | "unstructured" | "all"
    file: Optional[str] = None,               # filter by file name (e.g., "plant_data.xlsx" or "report.pdf")
    sheet: Optional[str] = None               # Excel only
) -> str:
    """
    Unified query across all sources.
    - restrict: restrict to a type ("excel" / "csv" / "unstructured" / "all"). Default None = no restriction.
    - file: filter by a specific file name.
    - sheet: filter by a specific sheet (Excel only).
    - include_sources: print/append sources along with the answer.
    """
    idx = ensure_index(STORAGE_DIR)

    will_filter = any([restrict, file, sheet])
    fetch_k = top_k * 3 if will_filter else top_k

    qe = idx.as_query_engine(similarity_top_k=fetch_k)
    resp = qe.query(query)

    if hasattr(resp, "source_nodes") and resp.source_nodes:
        type_map = {
            "excel": {"structured_excel_sheet"},
            "csv": {"structured_csv"},
            "unstructured": {"unstructured_doc"},
            "all": {"structured_excel_sheet", "structured_csv", "unstructured_doc"},
            None: {"structured_excel_sheet", "structured_csv", "unstructured_doc"},
        }
        allowed_types = type_map.get(restrict, type_map[None])

        filtered = []
        for n in resp.source_nodes:
            md = getattr(n.node, "metadata", {}) or {}
            ntype = md.get("type")
            if ntype not in allowed_types:
                continue
            if file:
                src = (md.get("source") or "").casefold()
                if file.casefold() not in src:
                    continue
            if sheet and md.get("sheet_name") != sheet:
                continue
            filtered.append(n)

        resp.source_nodes = (filtered if will_filter else resp.source_nodes)[:top_k]

    answer_text = str(resp)

    if include_sources and hasattr(resp, "source_nodes"):
        lines = ["\n=== Sources ==="]
        for i, n in enumerate(resp.source_nodes[:top_k], 1):
            md = getattr(n.node, "metadata", {}) or {}
            src = md.get("source") or md.get("file_name") or md.get("source_path")
            lines.append(
                f"{i}. type={md.get('type')} | source={src} | sheet={md.get('sheet_name')} | score={getattr(n, 'score', None)}"
            )
        return answer_text + "\n" + "\n".join(lines)

    return answer_text


# =============================
# Coverage verification utilities
# =============================

def _faiss_ntotal_from_storage_dir(storage_dir: Path) -> Optional[int]:
    """Best-effort retrieval of FAISS ntotal from a persisted LlamaIndex FAISS vector store.
    Tries direct attributes first; if missing, parses the vector_store.json to find the .index file and loads it with faiss.
    """
    try:
        vs = FaissVectorStore.from_persist_dir(str(storage_dir))
        for attr in ("faiss_index", "index", "_faiss_index"):
            obj = getattr(vs, attr, None)
            if obj is not None and hasattr(obj, "ntotal"):
                return int(obj.ntotal)
    except Exception:
        pass

    # Fallback: look for a vector_store json and load the underlying .index file manually
    try:
        import json as _json
        candidates = list(storage_dir.glob("*vector_store*.json"))
        for js in candidates:
            data = _json.loads(js.read_text(encoding="utf-8"))
            # recursively scan for any path ending with .index
            def _walk(o):
                if isinstance(o, dict):
                    for v in o.values():
                        p = _walk(v)
                        if p:
                            return p
                elif isinstance(o, list):
                    for v in o:
                        p = _walk(v)
                        if p:
                            return p
                elif isinstance(o, str):
                    if o.lower().endswith((".index", ".faiss")):
                        return o
                return None
            rel = _walk(data)
            if rel:
                p = Path(rel)
                idx_path = p if p.is_absolute() else (storage_dir / rel)
                if idx_path.exists():
                    ix = faiss.read_index(str(idx_path))
                    return int(ix.ntotal)
    except Exception:
        pass
    return None


def check_storage() -> None:
    """Print a quick summary of FAISS vectors inside the persisted storage."""
    if not STORAGE_DIR.exists():
        print("WARNING: Storage folder does not exist.")
        return

    ntotal = _faiss_ntotal_from_storage_dir(STORAGE_DIR)
    if ntotal is None:
        print("WARNING: Could not read FAISS ntotal (index may be empty or attribute not exposed).")
    else:
        print("OK: Storage folder found.")
        print(f"Total vectors in FAISS index: {ntotal}")


def _iter_all_data_files() -> List[Path]:
    exts = {".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".xls"}
    files = []
    for root in (STRUCTURED_DIR, UNSTRUCTURED_DIR):
        if root.exists():
            files.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return files


def verify_coverage(verbose: bool = True) -> dict:
    """Return a report showing missing/stale files against the manifest and FAISS vector count."""
    report = {
        "total_files_in_data": 0,
        "indexed_files": 0,
        "missing_in_storage": [],
        "stale_in_storage": [],
        "faiss_vectors": None,
    }

    files = _iter_all_data_files()
    report["total_files_in_data"] = len(files)

    manifest = load_manifest()

    # read FAISS ntotal using robust helper
    ntotal = _faiss_ntotal_from_storage_dir(STORAGE_DIR)
    if ntotal is not None:
        report["faiss_vectors"] = ntotal

    indexed = 0
    for f in files:
        key = str(f.resolve())
        fp_now = file_fingerprint(f)
        entry = manifest.get(key)
        if not entry:
            report["missing_in_storage"].append(key)
        else:
            indexed += 1
            if entry.get("fingerprint") != fp_now:
                report["stale_in_storage"].append(key)

    report["indexed_files"] = indexed

    if verbose:
        print(f"Files in data: {report['total_files_in_data']}")
        print(f"Files recorded in storage (manifest hits): {report['indexed_files']}")
        print(f"FAISS vectors: {report['faiss_vectors']}")
        if report["missing_in_storage"]:
            print("\nMissing (not indexed yet):")
            for x in report["missing_in_storage"]:
                print(" -", x)
        if report["stale_in_storage"]:
            print("\nStale (changed on disk; needs re-index):")
            for x in report["stale_in_storage"]:
                print(" -", x)

    return report


if __name__ == "__main__":
    # 1) Build/update index incrementally
    build_or_update_index(STORAGE_DIR)
    # 2) Quick checks
    check_storage()
    verify_coverage(verbose=True)
    
    # Example query
    print("\nExample query:")
    print(ask_one("what is the generator rotor dimensions?"))