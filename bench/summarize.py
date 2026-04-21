#!/usr/bin/env python3
"""Pivot warp_sort_bench results.csv into summary tables.

  python3 summarize.py [results.csv] [--md] [--ptxas ptxas.log]

Without --md: prints a single wide pipe-separated text table.
With --md:    prints three narrower Markdown tables, one per dtype.
With --ptxas: also emits a per-dtype kernel-resource table (regs, smem,
              stack, spill bytes) parsed from `ptxas.log`.

Rows:  (batch, seq)
Cols:  one per (impl, block_size) except std_sort which has no block.
Cells: "mean ± std" ms, or "skip" / "--".
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags_list = [a for a in sys.argv[1:] if a.startswith("--")]
PATH = Path(args[0] if args else "results.csv")
MD = "--md" in flags_list
PTXAS_LOG = None
for a in flags_list:
    if a == "--ptxas":
        PTXAS_LOG = Path("ptxas.log")
    elif a.startswith("--ptxas="):
        PTXAS_LOG = Path(a.split("=", 1)[1])

# (algorithm, block) -> column header
COLS = [
    ("std_sort",        "-",   "std_sort"),
    ("warp_sort",       "32",  "warp_32"),
    ("warp_sort",       "128", "warp_128"),
    ("cub_merge",       "32",  "merge_32"),
    ("cub_merge",       "128", "merge_128"),
    ("cub_radix",       "32",  "radix_32"),
    ("cub_radix",       "128", "radix_128"),
    ("cub_radix_4pass", "32",  "radix4_32"),
    ("cub_radix_4pass", "128", "radix4_128"),
]
COL_KEYS = [(a, b) for a, b, _ in COLS]
COL_HDRS = [h for _, _, h in COLS]

# Read
rows = {}  # (dtype, batch, seq) -> {col_key: "cell"}
with PATH.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = (r["dtype"], int(r["batch"]), int(r["seq"]))
        col = (r["algorithm"], r["block"])
        if r["mean_ms"] == "skip":
            cell = "skip"
        else:
            mean = float(r["mean_ms"])
            std  = float(r["std_ms"])
            # Adaptive precision so small and large values both look right.
            if mean >= 100:   fmt = f"{mean:7.1f} ± {std:5.1f}"
            elif mean >= 10:  fmt = f"{mean:7.2f} ± {std:5.2f}"
            elif mean >= 1:   fmt = f"{mean:7.3f} ± {std:5.3f}"
            else:             fmt = f"{mean:7.4f} ± {std:6.4f}"
            cell = fmt
        rows.setdefault(key, {})[col] = cell

# Order
DTYPES = ["float", "int", "int4"]
BATCHES = [1, 100, 10000]
SEQS = [100, 1000, 10000, 100000]

ordered = []
for batch in BATCHES:
    for seq in SEQS:
        for dt in DTYPES:
            key = (dt, batch, seq)
            if key in rows:
                ordered.append(key)

def cell_for(key, col):
    return rows[key].get(col, "--")

# ----------------------------------------------------------------------
# Plain text (wide) rendering
# ----------------------------------------------------------------------
def render_text():
    key_hdrs = ["dtype", "batch", "seq"]
    all_hdrs = key_hdrs + COL_HDRS
    widths = [len(h) for h in all_hdrs]
    for key in ordered:
        for i, h in enumerate(key_hdrs):
            val = {"dtype": key[0], "batch": str(key[1]), "seq": str(key[2])}[h]
            widths[i] = max(widths[i], len(val))
        for j, col in enumerate(COL_KEYS, start=len(key_hdrs)):
            widths[j] = max(widths[j], len(cell_for(key, col)))

    def fmt(vals):
        return " | ".join(v.ljust(w) for v, w in zip(vals, widths))

    print(fmt(all_hdrs))
    print("-+-".join("-" * w for w in widths))
    last_batch = None
    for key in ordered:
        dt, batch, seq = key
        if last_batch is not None and batch != last_batch:
            print()
        vals = [dt, str(batch), str(seq)] + [cell_for(key, c) for c in COL_KEYS]
        print(fmt(vals))
        last_batch = batch

# ----------------------------------------------------------------------
# ptxas.log parser -- pairs "Compiling entry function" with "Used N regs"
# and classifies each kernel by (impl, dtype, block, ipt).
# ----------------------------------------------------------------------
def load_ptxas(path: Path):
    text = path.read_text()
    records = []
    cur = None
    for line in text.splitlines():
        m = re.match(r"ptxas info\s*:\s*Compiling entry function '([^']+)'",
                     line)
        if m:
            if cur:
                records.append(cur)
            cur = {"mangled": m.group(1)}
            continue
        m = re.match(
            r"\s*(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads", line)
        if m and cur:
            cur["stack"]       = int(m.group(1))
            cur["spill_store"] = int(m.group(2))
            cur["spill_load"]  = int(m.group(3))
            continue
        m = re.match(r"ptxas info\s*:\s*Used (\d+) registers", line)
        if m and cur:
            cur["regs"] = int(m.group(1))
            m2 = re.search(r"(\d+) bytes smem", line)
            cur["smem"] = int(m2.group(1)) if m2 else 0
            continue
    if cur:
        records.append(cur)

    names = [r["mangled"] for r in records]
    try:
        demangled = subprocess.run(
            ["/usr/local/cuda/bin/cu++filt", *names],
            capture_output=True, text=True, check=True).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        demangled = subprocess.run(
            ["c++filt", *names],
            capture_output=True, text=True, check=True).stdout.splitlines()
    for r, d in zip(records, demangled):
        r["demangled"] = d

    # keyed by (impl, dtype, block, ipt-or-None)
    out = {}
    for r in records:
        d = r["demangled"]
        m = re.match(r"void our_sort_b(\d+)_kernel<([^,]+),", d)
        if m:
            out[("warp_sort", m.group(2).strip(), int(m.group(1)), None)] = r
            continue
        m = re.match(
            r"void cub_merge_kernel<([^,]+),\s*\(int\)(\d+),\s*\(int\)(\d+),",
            d)
        if m:
            out[("cub_merge", m.group(1).strip(),
                 int(m.group(2)), int(m.group(3)))] = r
            continue
        m = re.match(
            r"void cub_radix_kernel<([^,]+),\s*\(int\)(\d+),\s*\(int\)(\d+)>",
            d)
        if m:
            out[("cub_radix", m.group(1).strip(),
                 int(m.group(2)), int(m.group(3)))] = r
            continue
        m = re.match(
            r"void cub_radix4_int4_kernel<\(int\)(\d+),\s*\(int\)(\d+)>", d)
        if m:
            out[("cub_radix_4pass", "int4",
                 int(m.group(1)), int(m.group(2)))] = r
    return out

def ptxas_lookup(ptx, impl, dtype, block, seq):
    """Resolve the ptxas record for a given (impl, dtype, block, seq)."""
    if impl == "warp_sort":
        return ptx.get((impl, dtype, block, None))
    # CUB: IPT is encoded by (block, seq) in the dispatch tables.
    # The simplest lookup is to scan any record with matching impl/dtype/block
    # and the largest IPT not exceeding ceil(seq/block).  But since we only
    # instantiated specific (block, seq) pairs, iterate and match by seq bucket.
    # Use the same table the C++ dispatch uses:
    CUB_IPT = {
        (32, 100): 4, (32, 1000): 32,
        (128, 100): 1, (128, 1000): 8, (128, 10000): 79,
    }
    ipt = CUB_IPT.get((block, seq))
    if ipt is None:
        return None
    return ptx.get((impl, dtype, block, ipt))

# ----------------------------------------------------------------------
# Markdown rendering -- one table per dtype, dropping columns that are
# entirely "skip" for that dtype to keep width manageable.
# ----------------------------------------------------------------------
def render_markdown():
    ptx = load_ptxas(PTXAS_LOG) if PTXAS_LOG else None

    print("# warp_sort_bench results\n")
    print("All times in milliseconds, reported as `mean ± std`.")
    print("CUDA: 3 warm-up + 10 measured iters; host: 1 warm-up + 3 measured iters.")
    print("`skip` = combo beyond IPT / register budget (CUB only).\n")

    for dt in DTYPES:
        dt_keys = [k for k in ordered if k[0] == dt]
        if not dt_keys:
            continue

        # Drop columns that are all-skip for this dtype to narrow the table.
        active_cols = []
        for col, hdr in zip(COL_KEYS, COL_HDRS):
            if any(cell_for(k, col) != "skip" for k in dt_keys):
                active_cols.append((col, hdr))

        print(f"## dtype = `{dt}`\n")
        hdr_row = ["batch", "seq"] + [h for _, h in active_cols]
        print("| " + " | ".join(hdr_row) + " |")
        print("|" + "|".join("---:" if i >= 2 else "---"
                             for i in range(len(hdr_row))) + "|")
        last_batch = None
        for key in dt_keys:
            _, batch, seq = key
            cells = [cell_for(key, col) for col, _ in active_cols]
            row = [str(batch), str(seq)] + cells
            if last_batch is not None and batch != last_batch:
                print("|" + "|".join(" " for _ in range(len(hdr_row))) + "|")
            print("| " + " | ".join(row) + " |")
            last_batch = batch
        print()

        if ptx:
            # Kernel-resource table: one row per (impl, block, seq-for-CUB).
            # Columns: kernel, block, seq, IPT, regs, smem B, stack B, spill B.
            print(f"### `{dt}` kernel resources (ptxas -v)\n")
            print("| kernel | block | seq | IPT | regs | smem B | "
                  "stack B | spill st/ld B |")
            print("|---|---:|---:|---:|---:|---:|---:|---:|")
            resource_rows = []
            impls_for_dtype = {"float": ["warp_sort", "cub_merge", "cub_radix"],
                               "int":   ["warp_sort", "cub_merge", "cub_radix"],
                               "int4":  ["warp_sort", "cub_merge",
                                         "cub_radix_4pass"]}[dt]
            for impl in impls_for_dtype:
                for block in (32, 128):
                    if impl == "warp_sort":
                        rec = ptx.get((impl, dt, block, None))
                        if rec:
                            resource_rows.append((impl, block, "-", "-", rec))
                    else:
                        for seq in (100, 1000, 10000):
                            rec = ptxas_lookup(ptx, impl, dt, block, seq)
                            if rec:
                                CUB_IPT = {
                                    (32, 100): 4, (32, 1000): 32,
                                    (128, 100): 1, (128, 1000): 8,
                                    (128, 10000): 79,
                                }
                                ipt = CUB_IPT[(block, seq)]
                                resource_rows.append(
                                    (impl, block, seq, ipt, rec))
            for impl, block, seq, ipt, rec in resource_rows:
                spill = rec.get("spill_store", 0) + rec.get("spill_load", 0)
                print(f"| {impl} | {block} | {seq} | {ipt} | "
                      f"{rec.get('regs', '?')} | {rec.get('smem', 0)} | "
                      f"{rec.get('stack', 0)} | {spill} |")
            print()

if MD:
    render_markdown()
else:
    render_text()
