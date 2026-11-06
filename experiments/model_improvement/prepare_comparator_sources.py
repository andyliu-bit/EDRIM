"""Cache pinned, licensed author sources without executing notebooks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "ddprism": {
        "repo": "GIST-CSBL/DD-PRiSM",
        "commit": "8d9c5d5172ccfa76a389cd21d96a86e6dbc4ad11",
        "license": "CC-BY-NC-SA-4.0",
        "files": ["00_CombinationtherapyUtils.ipynb", "00_MonotherapyUtils.ipynb",
                  "00_Utils.ipynb", "01_Preprocessing.ipynb",
                  "04_CombinationTherapyModel_Training.ipynb", "CC-BY-NC-SA-4.0", "README.md"],
    },
    "combokr2": {
        "repo": "aalto-ics-kepaco/comboKR2.0",
        "commit": "0e26676de5fe996c34c2e641d465d5c04a6263ac",
        "license": "MIT",
        "files": ["comboKR2/__init__.py", "comboKR2/braid_surface_model_for_fit.py",
                  "comboKR2/helpers.py", "comboKR2/scalable_comboKR.py", "demo.py",
                  "requirements.txt", "environment.yml", "pyproject.toml", "LICENSE", "README.md"],
    },
    "rlscore": {
        "repo": "aatapa/RLScore",
        "commit": "e2ffc503c83d8a9d2187518d02043017d7f0d820",
        "license": "MIT",
        "files": ["rlscore/learner/cg_kron_rls.py", "LICENCE.TXT"],
    },
}


def main():
    manifest_path = ROOT / "data/raw_manifests/model_comparators.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    result = {"schema_version": 1, "sources": {}}
    for key, spec in SOURCES.items():
        entry = {k: v for k, v in spec.items() if k != "files"}
        entry["files"] = []
        for name in spec["files"]:
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{name}"
            path = ROOT / "data/raw/model_comparators" / key / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if prior:
                old = next(f for f in prior["sources"][key]["files"] if f["file"] == name)
                if old["sha256"] != digest or old["url"] != url:
                    raise RuntimeError(f"Pinned source drift: {key}/{name}")
            entry["files"].append({"file": name, "url": url, "sha256": digest,
                                   "bytes": path.stat().st_size})
        result["sources"][key] = entry
        print(f"Verified {key}: {spec['commit']}", flush=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
