"""PubTator splitting and export operations."""
import hashlib
import json
from collections import defaultdict

def deterministic_split(pmid: str, train_ratio=0.8, dev_ratio=0.1) -> str:
    """Hash split; test gets the remaining 1 - train_ratio - dev_ratio."""
    h = hashlib.md5(pmid.encode("utf8")).hexdigest()
    val = int(h[:8], 16) / float(0xFFFFFFFF)
    if val < train_ratio:
        return "train"
    elif val < train_ratio + dev_ratio:
        return "dev"
    else:
        return "test"

def parse_pubtator(path: str) -> dict:
    pmid_blocks = defaultdict(list)
    current_pmid = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "|t|" in line or "|a|" in line:
                pmid = line.split("|", 1)[0]
                current_pmid = pmid
                pmid_blocks[pmid].append(line)
            elif "\t" in line:
                pmid = line.split("\t")[0]
                current_pmid = pmid
                pmid_blocks[pmid].append(line)
            elif current_pmid:
                pmid_blocks[current_pmid].append(line)
    return pmid_blocks

def write_pubtator_file(pmid_list, pmid_blocks, out_path):
    with open(out_path, "w", encoding="utf-8") as fw:
        for pmid in pmid_list:
            if pmid in pmid_blocks:
                for line in pmid_blocks[pmid]:
                    fw.write(line + "\n")
                fw.write("\n")

def emit_jsonl_for_split(pmid_list, pmid_blocks, out_jsonl_path):
    out = []
    for pmid in pmid_list:
        lines = pmid_blocks.get(pmid, [])
        title = ""
        abstract = ""
        for l in lines:
            if "|t|" in l:
                parts = l.split("|", 2)
                if len(parts) == 3:
                    title = parts[2]
            elif "|a|" in l:
                parts = l.split("|", 2)
                if len(parts) == 3:
                    abstract = parts[2]
        context = " ".join(x for x in [title.strip(), abstract.strip()] if x)
        for l in lines:
            if "\t" in l:
                parts = l.split("\t")
                if len(parts) >= 6:
                    pmid_p, start, end, mention, sem_types, cui = parts[:6]
                    entry = {
                        "pmid": pmid_p,
                        "mention": mention,
                        "start": int(start),
                        "end": int(end),
                        "cui": cui,
                        "sem_types": sem_types.split(",") if sem_types else [],
                        "context": context
                    }
                    out.append(entry)
    with open(out_jsonl_path, "w", encoding="utf-8") as fw:
        for e in out:
            fw.write(json.dumps(e, ensure_ascii=False) + "\n")

def load_pmids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
