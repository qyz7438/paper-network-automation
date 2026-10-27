import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.paper_pipeline.reclassify_fulltext import load_direct


class FulltextReclassificationTests(unittest.TestCase):
    def test_verified_model_evidence_is_reused_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, sqlite3.connect(":memory:") as db:
            root = Path(folder)
            path = root / "evidence.json"
            path.write_text(json.dumps({"paper_id": "paper:test"}), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            db.execute("CREATE TABLE jobs(paper_id TEXT,stage TEXT,state TEXT,result TEXT,result_hash TEXT)")
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", ("paper:test", "fulltext_entities", "done", "evidence.json", digest))
            self.assertEqual(load_direct(db, root, "paper:test", "fulltext_entities"), {"paper_id": "paper:test"})
            path.write_text(json.dumps({"paper_id": "paper:changed"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_direct(db, root, "paper:test", "fulltext_entities")

    def test_ambiguous_completed_stages_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder, sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE jobs(paper_id TEXT,stage TEXT,state TEXT,result TEXT,result_hash TEXT)")
            db.executemany("INSERT INTO jobs VALUES(?,?,?,?,?)", [("p", "fulltext_entities", "done", "a", "x"),
                                                               ("p", "fulltext_entities", "done", "b", "y")])
            with self.assertRaises(ValueError):
                load_direct(db, Path(folder), "p", "fulltext_entities")


if __name__ == "__main__":
    unittest.main()
