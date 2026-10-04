import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CheckpointVerifierTests(unittest.TestCase):
    def test_checkpoint_verifier_accepts_exact_identity_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pth"
            payload = b"gdkvm-checkpoint-fixture"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            command = [
                sys.executable,
                str(ROOT / "scripts" / "verify_checkpoint.py"),
                "--path",
                str(path),
                "--sha256",
                digest,
                "--bytes",
                str(len(payload)),
            ]
            passed = subprocess.run(command, stdout=subprocess.PIPE, text=True)
            self.assertEqual(passed.returncode, 0)
            mismatch = list(command)
            mismatch[mismatch.index(digest)] = "0" * 64
            failed = subprocess.run(mismatch, stdout=subprocess.PIPE, text=True)
            self.assertEqual(failed.returncode, 2)


if __name__ == "__main__":
    unittest.main()
