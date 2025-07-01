#!/usr/bin/env python3
import os, re, random, shutil
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

# ─── Configuration ─────────────────────────────────────────────────────────────
SRC_ROOT     = "archwiki_text"                  # raw wikiextractor output
DOCS_ROOT    = "filtered_docs"                  # where individual docs will go
SPLIT_ROOT   = "split"                          # where train/val splits will be stored
TRAIN_RATIO  = 0.9                              # fraction of docs for training
NUM_WORKERS  = 8                                # number of parallel copy jobs
SEED         = 42                               # for reproducibility
MIN_DOC_SIZE = 1024                             # Minimum document size (bytes) to keep
# ─── Patterns ─────────────────────────────────────────────────────────────────
doc_pattern = re.compile(
        r'<doc[^>]*id="(?P<id>\d+)"[^>]*title="(?P<title>[^"]+)"[^>]*>(?P<body>.*?)</doc>',
    re.S
)

def wrap_code_blocks(body_text: str) -> str:
    """Wrap consecutive lines beginning with a space into <code>...</code>."""
    out_lines, in_code = [], False
    for line in body_text.splitlines(True):
        if line.startswith(" "):
            if not in_code:
                out_lines.append("<code>\n")
                in_code = True
            out_lines.append(line[1:])  # strip one leading space
        else:
            if in_code:
                out_lines.append("</code>\n")
                in_code = False
            out_lines.append(line)
    if in_code:
        out_lines.append("</code>\n")
    return "".join(out_lines)

def extract_docs():
    """Extract docs, filter non-English, drop small files, wrap code, write per-doc files."""
    os.makedirs(DOCS_ROOT, exist_ok=True)
    count = 0
    for sub in os.listdir(SRC_ROOT):
        subpath = os.path.join(SRC_ROOT, sub)
        if not os.path.isdir(subpath):
            continue
        for fname in os.listdir(subpath):
            text = open(os.path.join(subpath, fname), encoding="utf-8", errors="ignore").read()
            for m in doc_pattern.finditer(text):
                doc_id = m.group("id")
                title  = m.group("title")
                body   = m.group("body")
                # skip non-English (parentheses in title)
                if "(" in title and ")" in title:
                    continue
                wrapped = wrap_code_blocks(body)
                # Skip very small docs
                if len(wrapped.encode('utf-8')) < MIN_DOC_SIZE:
                    continue
                out = os.path.join(DOCS_ROOT, f"{doc_id}.txt")
                with open(out, "w", encoding="utf-8") as o:
                    o.write(wrapped)
                count += 1
            print(f"Finished processing file {sub}:{fname}")
        print(f"Finished processing files {sub}")
    print(f"[Extract] {count} English docs written to {DOCS_ROOT}")

def split_and_copy():
    """Shuffle DOCS_ROOT files, split into train/val, copy in parallel."""
    docs = sorted(os.listdir(DOCS_ROOT))
    random.seed(SEED)
    random.shuffle(docs)
    cutoff = int(len(docs) * TRAIN_RATIO)
    train_set = set(docs[:cutoff])
    val_set   = set(docs[cutoff:])

    train_dir = os.path.join(SPLIT_ROOT, "train")
    val_dir   = os.path.join(SPLIT_ROOT, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    def copy_one(doc):
        src = os.path.join(DOCS_ROOT, doc)
        dst_dir = train_dir if doc in train_set else val_dir
        shutil.copy(src, os.path.join(dst_dir, doc))

    print(f"[Split] Copying {len(docs)} docs → train ({len(train_set)}) / val ({len(val_set)}) with {NUM_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(copy_one, docs)
    print("[Done] Train/val split complete.")

if __name__ == "__main__":
    extract_docs()
    split_and_copy()
