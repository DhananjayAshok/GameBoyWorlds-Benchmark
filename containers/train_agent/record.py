import argparse
import json
import os
import sys

import cyclelib as lib


def main():
    parser = argparse.ArgumentParser(description="Hand in what a step of the cycle asks for. Write the record as "
                                                 "JSON, then run this on it: it says what is wrong, or accepts "
                                                 "it. Only an accepted record counts.")
    parser.add_argument("kind", choices=sorted(lib.VALIDATORS))
    parser.add_argument("path", nargs="?", help="the JSON file you wrote (default: /work/cycle/drafts/<kind>.json)")
    parser.add_argument("--template", action="store_true", help="print the shape of this record and stop")
    args = parser.parse_args()
    if args.template:
        print(json.dumps(lib.TEMPLATES[args.kind], indent=1))
        return
    context = lib.read_json(lib.CONTEXT, {})
    if args.kind not in context.get("accepts", []):
        sys.exit(f"this step does not take a {args.kind} record; it takes: {context.get('accepts', [])}")
    path = args.path or os.path.join(lib.DRAFTS, f"{args.kind}.json")
    try:
        with open(path) as f:
            draft = json.load(f)
    except OSError as error:
        sys.exit(f"cannot read {path}: {error}")
    except ValueError as error:
        sys.exit(f"{path} is not valid JSON: {error}")
    errors = lib.VALIDATORS[args.kind](draft, context)
    if errors:
        print(f"NOT ACCEPTED: {len(errors)} problem(s) in {path}:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    lib.write_json(os.path.join(lib.STAGED, f"{args.kind}.json"), draft)
    print(f"ACCEPTED: the {args.kind} record is handed in. Running this again replaces it.")


if __name__ == "__main__":
    main()
