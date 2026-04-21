#!/usr/bin/env python3
"""Parse ptxas -v output into a readable per-kernel table.

  nvcc ... -Xptxas=-v warp_sort_bench.cu 2> ptxas.log
  python3 parse_ptxas.py [ptxas.log]

Reports: kernel short-name, dtype, block, ITEMS_PER_THREAD (if applicable),
         registers/thread, smem bytes, stack bytes, spill store/load bytes.
"""
import re
import subprocess
import sys
from pathlib import Path

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "ptxas.log")
text = PATH.read_text()

# Pair up the three ptxas lines per kernel.
records = []
cur = None
for line in text.splitlines():
    m = re.match(r"ptxas info\s*:\s*Compiling entry function '([^']+)'", line)
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

# Demangle en masse.
names = [r["mangled"] for r in records]
demangled = subprocess.run(
    ["/usr/local/cuda/bin/cu++filt", *names],
    capture_output=True, text=True, check=True).stdout.splitlines()
for r, d in zip(records, demangled):
    r["demangled"] = d

# Classify each kernel.
def classify(d):
    """Return (short, dtype, block, ipt)."""
    # our_sort_b32_kernel<T, Cmp>(...)
    m = re.match(r"void our_sort_b(\d+)_kernel<([^,]+),\s*([^>]+)>", d)
    if m:
        return (f"our_sort_b{m.group(1)}", m.group(2).strip(),
                int(m.group(1)), None)
    # cub_merge_kernel<T, BS, IPT, Cmp>
    m = re.match(
        r"void cub_merge_kernel<([^,]+),\s*\(int\)(\d+),\s*\(int\)(\d+),",
        d)
    if m:
        return ("cub_merge", m.group(1).strip(),
                int(m.group(2)), int(m.group(3)))
    # cub_radix_kernel<T, BS, IPT>
    m = re.match(
        r"void cub_radix_kernel<([^,]+),\s*\(int\)(\d+),\s*\(int\)(\d+)>",
        d)
    if m:
        return ("cub_radix", m.group(1).strip(),
                int(m.group(2)), int(m.group(3)))
    # cub_radix4_int4_kernel<BS, IPT>
    m = re.match(
        r"void cub_radix4_int4_kernel<\(int\)(\d+),\s*\(int\)(\d+)>", d)
    if m:
        return ("cub_radix_4pass", "int4", int(m.group(1)), int(m.group(2)))
    return (None, None, None, None)

rows = []
for r in records:
    short, dtype, block, ipt = classify(r["demangled"])
    if short is None:
        continue
    rows.append({
        "short": short, "dtype": dtype, "block": block, "ipt": ipt,
        "regs": r.get("regs", 0), "smem": r.get("smem", 0),
        "stack": r.get("stack", 0),
        "spill_store": r.get("spill_store", 0),
        "spill_load": r.get("spill_load", 0),
    })

# Sort: kernel family, dtype, block, ipt.
family_order = {"our_sort_b32": 0, "our_sort_b128": 1,
                "cub_merge": 2, "cub_radix": 3, "cub_radix_4pass": 4}
dtype_order = {"float": 0, "int": 1, "int4": 2}
rows.sort(key=lambda r: (family_order.get(r["short"], 99),
                         dtype_order.get(r["dtype"], 99),
                         r["block"] or 0, r["ipt"] or 0))

# Render.
hdr = ["kernel", "dtype", "block", "IPT", "regs", "smem B",
       "stack B", "spill st", "spill ld"]
widths = [len(h) for h in hdr]
table = []
for r in rows:
    row = [
        r["short"], r["dtype"],
        str(r["block"]),
        "-" if r["ipt"] is None else str(r["ipt"]),
        str(r["regs"]), str(r["smem"]),
        str(r["stack"]), str(r["spill_store"]), str(r["spill_load"]),
    ]
    table.append(row)
    for i, v in enumerate(row):
        widths[i] = max(widths[i], len(v))

def fmt(row):
    return " | ".join(v.ljust(w) for v, w in zip(row, widths))

print(fmt(hdr))
print("-+-".join("-" * w for w in widths))
last_family = None
for r, raw in zip(rows, table):
    if last_family is not None and r["short"] != last_family:
        print()
    print(fmt(raw))
    last_family = r["short"]
