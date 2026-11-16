"""Bump the No-Nonsense Photo Sorter version in one step.

  python bump_version.py patch|minor|major|X.Y.Z ["one-line note"]

APP_VERSION in photo_helper.py stays the only place the number lives. This edits it, files the
CHANGELOG "Unreleased" entries under the new version and date, and leaves a fresh Unreleased heading.
The .exe's Properties > Details version is generated from APP_VERSION by build.py."""
import datetime
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, LOG = HERE / "photo_helper.py", HERE / "CHANGELOG.md"
PAT = re.compile(r'^(APP_VERSION\s*=\s*")(\d+)\.(\d+)\.(\d+)(")', re.M)


def current():
    m = PAT.search(SRC.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("Could not find APP_VERSION in photo_helper.py")
    return tuple(int(m.group(i)) for i in (2, 3, 4))


def main(argv):
    cur = current()
    if not argv:
        print("Current version: %d.%d.%d\nUsage: bump_version.py patch|minor|major|X.Y.Z [\"note\"]" % cur)
        return 0
    kind, note = argv[0].lower(), (argv[1].strip() if len(argv) > 1 else "")
    if kind == "patch":
        new = (cur[0], cur[1], cur[2] + 1)
    elif kind == "minor":
        new = (cur[0], cur[1] + 1, 0)
    elif kind == "major":
        new = (cur[0] + 1, 0, 0)
    else:
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", kind)
        if not m:
            raise SystemExit("Use patch, minor, major or a version like 2.3.0")
        new = tuple(int(x) for x in m.groups())
    if new <= cur:
        raise SystemExit("New version %d.%d.%d must be higher than %d.%d.%d" % (new + cur))
    text = LOG.read_text(encoding="utf-8") if LOG.exists() else "# Changelog\n\n## Unreleased\n"
    if "## Unreleased" not in text:
        text = text.rstrip() + "\n\n## Unreleased\n"
    if note:
        text = text.replace("## Unreleased\n", "## Unreleased\n- %s\n" % note, 1)
    label = "%d.%d.%d" % new
    text = text.replace("## Unreleased", "## Unreleased\n\n## %s - %s" % (label, datetime.date.today().isoformat()), 1)
    LOG.write_text(text, encoding="utf-8")
    code = SRC.read_text(encoding="utf-8")
    SRC.write_text(PAT.sub(lambda m: m.group(1) + label + m.group(5), code, count=1), encoding="utf-8", newline="")
    print("Version %d.%d.%d -> %s. CHANGELOG updated. Rebuild to put it in the .exe." % (cur + (label,)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
