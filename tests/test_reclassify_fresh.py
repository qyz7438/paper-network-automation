import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.paper_pipeline.reclassify_fresh import load_completed


class FreshReclassificationTests(unittest.TestCase):
    def test_cached_stage_is_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as folder, sqlite3.connect(":memory:") as db:
            root = Path(folder)
            path = root / "evidence.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump({"value": {"paper_id": "paper:test"}}, handle)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            db.execute("CREATE TABLE jobs(paper_id TEXT,stage TEXT,state TEXT,result TEXT,result_hash TEXT)")
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", ("paper:test", "fresh_parse", "done", "evidence.json", digest))
            self.assertEqual(load_completed(db, root, "paper:test", "fresh_parse"), {"paper_id": "paper:test"})
            with path.open("w", encoding="utf-8") as handle:
                json.dump({"value": {"paper_id": "paper:other"}}, handle)
            with self.assertRaises(ValueError):
                load_completed(db, root, "paper:test", "fresh_parse")

    def test_multiple_completed_stages_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as folder, sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE jobs(paper_id TEXT,stage TEXT,state TEXT,result TEXT,result_hash TEXT)")
            db.executemany("INSERT INTO jobs VALUES(?,?,?,?,?)", [("p", "fresh_parse", "done", "a", "x"),
                                                               ("p", "fresh_parse", "done", "b", "y")])
            with self.assertRaises(ValueError):
                load_completed(db, Path(folder), "p", "fresh_parse")


if __name__ == "__main__":
    unittest.main()
