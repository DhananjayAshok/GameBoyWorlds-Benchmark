import argparse
import sys
import time

import cyclelib as lib


def main():
    parser = argparse.ArgumentParser(description="One stretch of a run in reading order: what the server measured, "
                                                 "what the harness intended, planned, tried and concluded, and "
                                                 "frames of what the screen showed.")
    parser.add_argument("--run", required=True, help="a playthrough or short test id")
    parser.add_argument("--minutes", type=float, default=None, help="the last N minutes (default: the whole run)")
    parser.add_argument("--since", type=float, default=None, help="unix time the stretch starts")
    parser.add_argument("--until", type=float, default=None, help="unix time the stretch ends (default: now)")
    parser.add_argument("--no_describe", action="store_true", help="make the frame sheets but do not describe them")
    args = parser.parse_args()
    if lib.run_dir(args.run) is None:
        sys.exit(f"there is no run {args.run!r}")
    until = args.until or time.time()
    since = args.since if args.since is not None else (until - args.minutes * 60 if args.minutes else 0)
    print(lib.digest(args.run, since, until, f"digest_{int(until)}", describe=not args.no_describe))


if __name__ == "__main__":
    main()
