import xlrd, math, re, csv, sys, os
from collections import defaultdict
W3 = sys.argv[1]; OUT = sys.argv[2]
PX = 10000 / 40.3
ISO = {"101":"crack","102":"crack","105":"crater crack","201":"gas pore","2013":"clustered porosity",
       "2014":"linear porosity","2015":"elongated cavity","2016":"wormhole","301":"slag inclusion",
       "304":"metallic inclusion","401":"lack of fusion","402":"lack of penetration","501":"undercut",
       "504":"excess penetration","515":"root concavity"}
def name(t):
    t = t.replace("*","").split(".")[0]
    for k in sorted(ISO, key=len, reverse=True):
        if t.startswith(k): return ISO[k]
    return "other"
txt = open(os.path.join(W3, "W0003_readme.txt")).read()
roc2img = {}
for k, v in re.findall(r"(W0003_\d+\.png) is (\S+)", txt):
    g = re.match(r"RRT-(\d+)R", v)
    if g: roc2img[int(g.group(1))] = k
s = xlrd.open_workbook(os.path.join(W3, "real-values.xls")).sheet_by_index(0)
flaws, ncell = defaultdict(list), {}
for r in range(2, s.nrows):
    roc = int(s.cell_value(r, 0)); flaws[roc]
    m = s.cell_value(r, 9)
    if m != "" and roc not in ncell: ncell[roc] = int(m) if float(m) < 100 else 20
    c = s.cell_value(r, 2)
    if isinstance(c, float):
        flaws[roc].append((int(c), float(s.cell_value(r, 4)), str(s.cell_value(r, 3))))
w0001 = {12:"W0001_0002",13:"W0001_0007",22:"W0001_0003",28:"W0001_0004",31:"W0001_0008",
         39:"W0001_0005",40:"W0001_0006",106:"W0001_0009",107:"W0001_0010"}
rows = []
for roc in sorted(flaws):
    cell = defaultdict(set)
    for c, l, t in flaws[roc]:
        for k in range(c, min(ncell[roc], c + max(0, math.ceil(l / 10) - 1)) + 1):
            cell[k].add(name(t))
    n = ncell[roc]
    for k in range(1, n + 1):
        lab = int(k in cell)
        safe = int(lab == 1 or ((k - 1) not in cell and (k + 1) not in cell))
        rows.append(dict(image=roc2img[roc], roc=roc, cell=k, x0=round((k - 1) * PX), x1=round(k * PX),
                         label=lab, clean_margin_ok=safe, defect_types=";".join(sorted(cell.get(k, set()))),
                         also_in_W0001=w0001.get(roc, "")))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "cells |", sum(r["label"] for r in rows), "defective |", len({r["image"] for r in rows}), "images")
