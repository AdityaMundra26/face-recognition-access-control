"""Manual smoke test for the access-attempt audit log.

    python demo_audit_log.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.app import AuditLog


def main() -> int:
    tmp_dir = Path(tempfile.mkdtemp(prefix="face_audit_log_demo_"))
    log_path = tmp_dir / "access_log.jsonl"
    ok = True

    try:
        log = AuditLog(path=log_path)

        print(f"empty log -> recent(): {log.recent()}")
        if log.recent() != []:
            print("[FAIL] expected an empty list before any records")
            ok = False

        log.record(name="Alice", similarity=0.91, granted=True)
        log.record(name=None, similarity=0.12, granted=False)
        log.record(name="Bob", similarity=0.55, granted=True)

        print("\nrecent(limit=2), newest first:")
        events = log.recent(limit=2)
        for event in events:
            print(f"   {event}")
        if [e.name for e in events] != ["Bob", None]:
            print("[FAIL] expected [Bob, None] newest-first")
            ok = False
        else:
            print("[ OK ]")

        print(f"\nlog file exists at {log_path}: {log_path.is_file()}")
        if not log_path.is_file():
            print("[FAIL] expected the log file to have been created")
            ok = False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
