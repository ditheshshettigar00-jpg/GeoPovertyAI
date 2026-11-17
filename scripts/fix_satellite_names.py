"""
fix_satellite_names.py
Copies satellite images whose file names use an old/different state name
(e.g. orissa_*, andhrapradesh_adilabad) to the name the app expects
(e.g. odisha_*, telangana_adilabad). Originals are kept. Dry run by default.

Run from the project root (venv active):
    python scripts/fix_satellite_names.py          # preview only
    python scripts/fix_satellite_names.py --apply  # copy files
"""
import csv, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.name_keys import clean_key

SAT_DIR = os.path.join("data", "satellite")
TABLE = os.path.join("data", "processed", "final_training_table.csv")

# expected state key -> old state keys the images may have been saved under
ALIASES = {
    "odisha": ["orissa"],
    "telangana": ["andhrapradesh"],
    "andamannicobarislands": ["andamanandnicobar"],
    "dadranagarhavelidamandiu": ["dadraandnagarhaveli", "damananddiu"],
}

def main(apply):
    files = {f for f in os.listdir(SAT_DIR) if f.endswith("_rgb.png")}
    with open(TABLE, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    todo, unresolved = [], []
    for r in rows:
        sk, dk = clean_key(r["state_name"]), clean_key(r["area_name"])
        want = f"{sk}_{dk}_rgb.png"
        if want in files:
            continue
        src = None
        for old in ALIASES.get(sk, []):
            cand = f"{old}_{dk}_rgb.png"
            if cand in files:
                src = cand
                break
        if src:
            todo.append((src, want))
        else:
            unresolved.append((r["state_name"], r["area_name"]))
    for src, dst in todo:
        print(f"{'COPY' if apply else 'would copy'}: {src} -> {dst}")
        if apply:
            shutil.copyfile(os.path.join(SAT_DIR, src), os.path.join(SAT_DIR, dst))
    print(f"\n{len(todo)} fixable, {len(unresolved)} still missing")
    for s, d in unresolved:
        print("  no image:", s, "/", d)

if __name__ == "__main__":
    main("--apply" in sys.argv)
