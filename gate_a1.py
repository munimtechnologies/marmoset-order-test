# Copyright 2026 MV2, Munim, Inc. Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working in MV2's name.
"""Job gate for amendment 1: compute nothing until the amendment and its code on GitHub main equal
the job's own copies (CRLF normalised), so the position test is public before it is computed.
Waits at most 60 minutes."""
import hashlib
import sys
import time
import urllib.request

RAW = "https://raw.githubusercontent.com/munimtechnologies/marmoset-order-test/main/"
FILES = ["AMENDMENT-1-POSITION.md", "marmoset_position.py", "test_marmoset_position.py"]


def md5(blob):
    return hashlib.md5(blob.replace(b"\r\n", b"\n")).hexdigest()


def check():
    ok, lines = True, []
    for name in FILES:
        with open(name, "rb") as fh:
            lm = md5(fh.read())
        try:
            with urllib.request.urlopen(RAW + name + "?t=%d" % int(time.time()), timeout=60) as r:
                rm = md5(r.read())
        except Exception as exc:  # noqa: BLE001
            lines.append("%s local %s remote fetch failed: %s" % (name, lm, exc))
            ok = False
            continue
        lines.append("%s local %s remote %s %s" % (name, lm, rm, "same" if lm == rm else "DIFFERENT"))
        ok = ok and lm == rm
    return ok, lines


def main():
    deadline = time.time() + 60 * 60
    while True:
        ok, lines = check()
        print("\n".join(lines), flush=True)
        if ok:
            print("amendment and code are public and identical", flush=True)
            return 0
        if time.time() > deadline:
            print("gate failed: public copies missing or different after 60 min; nothing computed",
                  flush=True)
            return 1
        time.sleep(300)


if __name__ == "__main__":
    sys.exit(main())
