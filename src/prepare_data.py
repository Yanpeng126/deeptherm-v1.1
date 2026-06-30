"""
Convert Supplementary Data 1 (xlsx) to a chemprop-ready CSV.

The xlsx ships with a non-standard OOXML namespace (purl.oclc.org), so
openpyxl/pandas can't open it. Falling back to raw XML parsing.
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

NS = "{http://purl.oclc.org/ooxml/spreadsheetml/main}"


def read_xlsx_rows(path: Path):
    with zipfile.ZipFile(path) as z:
        with z.open("xl/sharedStrings.xml") as f:
            ss = ET.parse(f).getroot()
        shared = [(si.find(f"{NS}t").text or "") for si in ss.findall(f"{NS}si")]

        with z.open("xl/worksheets/sheet1.xml") as f:
            sheet = ET.parse(f).getroot()

    data = sheet.find(f"{NS}sheetData")
    rows = []
    for row in data.findall(f"{NS}row"):
        cells = []
        for c in row.findall(f"{NS}c"):
            v = c.find(f"{NS}v")
            if v is None:
                cells.append("")
            elif c.get("t") == "s":
                cells.append(shared[int(v.text)])
            else:
                cells.append(v.text)
        rows.append(cells)
    return rows


def canonical_smiles(smi: str) -> str | None:
    # Radicals like [CH2]C(C)(OO)... need sanitize=True to keep the radical
    # electron count; RDKit handles them correctly by default.
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True,
                    help="Path to Supplementary Data 1 xlsx")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV path")
    args = ap.parse_args()

    rows = read_xlsx_rows(args.xlsx)
    header, body = rows[0], rows[1:]

    # Sanity: SMILES must be col 0, 9 numeric targets follow
    assert header[0].lower().startswith("smiles"), header
    assert len(header) == 10, f"expected 10 columns, got {len(header)}"

    # Shorter target names for chemprop CLI -- the unit suffixes in the
    # original header break some YAML/CLI tooling.
    target_names = ["Hf_298", "S_298",
                    "Cp_300", "Cp_400", "Cp_500", "Cp_600",
                    "Cp_800", "Cp_1000", "Cp_1500"]
    new_header = ["smiles"] + target_names

    n_invalid = 0
    seen_canonical: dict[str, int] = {}
    duplicates: list[tuple[str, str]] = []
    cleaned: list[list[str]] = []

    for i, row in enumerate(body):
        smi_raw = row[0].strip()
        can = canonical_smiles(smi_raw)
        if can is None:
            n_invalid += 1
            print(f"[warn] invalid SMILES at row {i+2}: {smi_raw}", file=sys.stderr)
            continue
        if can in seen_canonical:
            duplicates.append((smi_raw, can))
            continue
        seen_canonical[can] = i
        targets = row[1:]
        # The xlsx stores some values with float artifacts (e.g.
        # 9.3000000000000007). Round to 4 decimals to match what chemkin
        # NASA polynomials would use anyway.
        targets = [f"{float(x):.4f}" for x in targets]
        cleaned.append([can] + targets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(new_header)
        w.writerows(cleaned)

    print(f"wrote {len(cleaned)} rows to {args.out}")
    print(f"  invalid SMILES skipped: {n_invalid}")
    print(f"  duplicates dropped:     {len(duplicates)}")
    if duplicates:
        print("  (first 5 duplicates:)")
        for raw, can in duplicates[:5]:
            print(f"    {raw} -> {can}")


if __name__ == "__main__":
    main()
