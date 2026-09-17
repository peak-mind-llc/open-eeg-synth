"""`python -m open_eeg_synth make-case --seed N --out DIR`"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_eeg_synth.case import CaseSpec, make_case
from open_eeg_synth.casefile.truth import case_truth
from open_eeg_synth.casefile.writer import write_case
from open_eeg_synth.recipes import resting_case


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="open_eeg_synth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make-case", help="write a synthetic resting case (EDF + sealed truth)")
    mk.add_argument("--seed", type=int, required=True)
    mk.add_argument("--out", required=True)
    mk.add_argument("--duration", type=float, default=240.0)
    mk.add_argument("--spec", help="JSON file with a full CaseSpec (overrides --duration)")
    mk.add_argument("--embed-layers", action="store_true")
    mk.add_argument("--print-truth", action="store_true")
    args = ap.parse_args(argv)

    if args.spec:
        spec = CaseSpec.from_dict(json.loads(Path(args.spec).read_text()))
        spec = CaseSpec.from_dict({**spec.to_dict(), "seed": args.seed})
    else:
        spec = resting_case(args.seed, duration_s=args.duration)
    case = make_case(spec)
    paths = write_case(args.out, case, embed_layers=args.embed_layers)
    if args.print_truth:
        print(json.dumps(case_truth(case, {k: v.name for k, v in paths.recordings.items()})))
    else:
        print(paths.truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
