"""Check a pull request for terms that must not appear in this repository.

The terms are not stored here. They come from the PRIVATE_TERMS secret, one regular expression per
line, matched without regard to case. PRIVATE_ALLOW holds text that is removed before the check
(for example the copyright line). The check reads the pull request title and description, every
commit message in the pull request, and every added line of the diff.

The log never prints the text that matched, because the log of a public repository is public too.
It prints where the match is and the number of the term in the list.

A pull request from a fork cannot read secrets, so there the check prints a notice and passes.
"""
import os
import re
import subprocess
import sys


def patterns(name):
    return [re.compile(line.strip(), re.I) for line in os.environ.get(name, "").splitlines()
            if line.strip()]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def texts(base, head):
    """(where, text) for everything this pull request adds."""
    yield "the pull request title", os.environ.get("TITLE", "")
    yield "the pull request description", os.environ.get("BODY", "")
    for sha in git("rev-list", f"{base}..{head}").split():
        yield f"the message of commit {sha[:7]}", git("log", "-1", "--format=%B", sha)
    path, line = None, 0
    for row in git("diff", "--unified=0", "--no-color", f"{base}...{head}").splitlines():
        if row.startswith("+++ "):
            path = row[6:] if row.startswith("+++ b/") else None
        elif row.startswith("@@"):
            m = re.search(r"\+(\d+)", row)
            line = int(m.group(1)) if m else 0
        elif row.startswith("+") and path:
            yield f"{path}:{line}", row[1:]
            line += 1


def main():
    terms = patterns("TERMS")
    if not terms:
        print("No term list is available (a pull request from a fork cannot read secrets). Skipped.")
        return 0
    allow = patterns("ALLOW")
    hits = 0
    for where, text in texts(os.environ["BASE"], os.environ["HEAD"]):
        for a in allow:
            text = a.sub("", text)
        for number, term in enumerate(terms, 1):
            if term.search(text):
                print(f"{where} matches term {number}")
                hits += 1
    if hits:
        print(f"{hits} match(es). Rewrite these places for a reader outside the project and push again.")
        return 1
    print(f"Checked against {len(terms)} terms. Nothing matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
