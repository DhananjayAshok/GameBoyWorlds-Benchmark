import argparse
import os
import sys
import time
import uuid

import cyclelib as lib

VERDICTS = ("not_yet", "fixed", "failed", "stop_point_reached")


def ask(kind, **payload):
    request_id = uuid.uuid4().hex[:12]
    lib.write_json(os.path.join(lib.REQUESTS, f"{request_id}.json"), {"id": request_id, "kind": kind, **payload})
    reply_path = os.path.join(lib.REPLIES, f"{request_id}.json")
    deadline = time.time() + 120
    while time.time() < deadline:
        reply = lib.read_json(reply_path)
        if reply is not None:
            os.remove(reply_path)
            return reply
        time.sleep(1)
    sys.exit("the controller did not answer; short tests can only be run during an issue step")


def current():
    tests = lib.read_json(lib.CONTEXT, {}).get("tests", [])
    if not tests:
        sys.exit("no short test has been started in this step; start one with: short_test.py start ...")
    tid = tests[-1]
    directory = os.path.join(lib.TESTS, tid)
    return tid, directory, lib.read_json(os.path.join(directory, "test.json"))


def start(args):
    reply = ask("start_test", start=args.start_from, game=args.game, test=args.test, stop=args.stop,
                minutes=args.minutes)
    if not reply.get("ok"):
        sys.exit(f"NOT STARTED: {reply.get('error')}")
    print(f"Started short test {reply['tid']} on harness commit {reply['commit'][:10]}, for at most "
          f"{reply['minutes']} minutes. Now run `short_test.py wait`: it returns after a minute with what "
          "happened, and you then record a verdict with `short_test.py verdict`. Repeat until you can say "
          "fixed, failed or stop_point_reached.")


def wait(args):
    tid, directory, info = current()
    waits = lib.read_jsonl(os.path.join(directory, "waits.jsonl"))
    verdicts = lib.read_jsonl(os.path.join(directory, "checks.jsonl"))
    if len(waits) > len(verdicts):
        sys.exit("record a verdict for the last minute first: short_test.py verdict --verdict ... --evidence ...")
    if verdicts and verdicts[-1]["verdict"] != "not_yet":
        sys.exit(f"short test {tid} is over: its verdict is {verdicts[-1]['verdict']}")
    since = waits[-1]["t"] if waits else info["started"]
    while time.time() < since + lib.SHORT_CHECK_SECONDS:
        if (lib.read_json(os.path.join(directory, "test.json")) or {}).get("status") != "running":
            break
        time.sleep(2)
    now = time.time()
    info = lib.read_json(os.path.join(directory, "test.json"))
    lib.append_jsonl(os.path.join(directory, "waits.jsonl"), {"t": now})
    print(f"Short test {tid}, minute {len(waits) + 1}: {(now - info['started']) / 60:.1f} of {info['minutes']} "
          f"minutes used.")
    print(f"The test: {info['test']}")
    print(f"The stopping point: {info['stop']}")
    print()
    print(lib.digest(tid, since, now, f"minute_{len(waits) + 1}"))
    if info.get("status") != "running":
        print(f"THE TEST HAS ENDED ({info.get('end_reason')}). Record your final verdict now: fixed, failed or "
              "stop_point_reached.")
    else:
        print("Record a verdict now (not_yet keeps it running), then wait again.")


def verdict(args):
    tid, directory, info = current()
    waits = lib.read_jsonl(os.path.join(directory, "waits.jsonl"))
    verdicts = lib.read_jsonl(os.path.join(directory, "checks.jsonl"))
    if len(verdicts) >= len(waits):
        sys.exit("there is nothing new to judge: run `short_test.py wait` first")
    if len(args.evidence.strip()) < lib.MIN_TEXT:
        sys.exit(f"--evidence must say what you saw, in at least {lib.MIN_TEXT} characters")
    if args.verdict == "stop_point_reached" and args.passed not in ("yes", "no"):
        sys.exit("stop_point_reached needs --passed yes|no: did the test pass by the time it got there?")
    if args.verdict == "not_yet" and info.get("status") != "running":
        sys.exit("the test has ended, so the verdict is fixed, failed or stop_point_reached")
    lib.append_jsonl(os.path.join(directory, "checks.jsonl"),
                     {"t": time.time(), "minute": len(waits), "verdict": args.verdict, "passed": args.passed,
                      "evidence": args.evidence})
    if args.verdict == "not_yet":
        print("Recorded. The test keeps running; run `short_test.py wait` again.")
        return
    ask("stop_test", tid=tid)
    print(f"Recorded, and short test {tid} is stopped. Result: {lib.test_result(tid)}.")


def main():
    parser = argparse.ArgumentParser(description="A short playthrough that tests one fix: at most "
                                                 f"{lib.SHORT_TEST_MINUTES} minutes, from a checkpoint, with a "
                                                 "declared test and stopping point, judged every minute.")
    commands = parser.add_subparsers(dest="command", required=True)
    s = commands.add_parser("start", help="start a short test; commit your fix first")
    s.add_argument("--from", dest="start_from", required=True,
                   help=f"\"{lib.BEGINNING}\" or a checkpoint as <run>:<id>")
    s.add_argument("--game", default=None, help=f"needed with --from {lib.BEGINNING}")
    s.add_argument("--test", required=True, help="the one thing that shows the fix works")
    s.add_argument("--stop", required=True, help="the point at which the test is over")
    s.add_argument("--minutes", type=float, default=lib.SHORT_TEST_MINUTES)
    s.set_defaults(run=start)
    w = commands.add_parser("wait", help="wait a minute, then print what happened in it")
    w.set_defaults(run=wait)
    v = commands.add_parser("verdict", help="judge the minute `wait` just showed")
    v.add_argument("--verdict", required=True, choices=VERDICTS)
    v.add_argument("--evidence", required=True)
    v.add_argument("--passed", default=None, choices=("yes", "no"))
    v.set_defaults(run=verdict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
