"""Immutable local inputs, atomic artifacts and a resumable SQLite queue."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
import uuid
from typing import Any

PARSER_VERSION = "pymupdf-blocks-v1"
WORK_PREFIX = Path("_working/paper-network-automation-v1")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def checked_output(path: Path, boundary: Path) -> Path:
    root = boundary.resolve()
    target = path.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("artifact path escapes output root")
    current = path.absolute()
    while current != root:
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("symlink/junction output rejected")
        if current.parent == current:
            raise ValueError("output parent not contained")
        current = current.parent
    return path


def write_json(path: Path, value: Any, boundary: Path | None = None) -> bool:
    """Replace only valid JSON; preserve bytes and mtime for identical content."""
    if boundary is not None:
        checked_output(path, boundary)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    json.loads(data)
    if path.exists() and path.read_bytes() == data:
        read_json(path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pipeline-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        read_json(Path(temporary))
        os.replace(temporary, path)
        read_json(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def contained(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("absolute/traversal source path rejected")
    result = (root / rel).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("source path escapes vault")
    return result


def output_root(vault: Path, output: Path) -> Path:
    result = output.resolve()
    prefix = (vault / WORK_PREFIX).resolve()
    if not prefix.is_relative_to(vault.resolve()) or result == prefix or not result.is_relative_to(prefix):
        raise ValueError("output must be a dedicated child of _working/paper-network-automation-v1")
    return result


def freeze(vault: Path, output: Path, identifiers: list[str]) -> dict[str, Any]:
    output = output_root(vault, output)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("nonempty unique paper IDs required")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if [row["id"] for row in manifest["papers"]] != identifiers:
            raise ValueError("frozen cohort differs; use another output directory")
        check_inputs(vault, manifest)
        return manifest
    graph = read_json(vault / "_wiki/network/graph.json")
    nodes = {node["id"]: node for node in graph["nodes"]}
    protected = {}
    for relative in ["_wiki/network/graph.json", "_wiki/network/paper-profiles.json"]:
        path = contained(vault, relative)
        if path.is_file():
            protected[relative] = sha(path)
    papers = []
    seen_sources = set()
    for identifier in identifiers:
        node = nodes.get(identifier)
        if not node or node["type"] != "paper":
            raise ValueError("paper ID not present in current graph: " + identifier)
        choices = node.get("metadata", {}).get("pdf_en") or []
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("exactly one explicit English source required: " + identifier)
        relative = choices[0]
        if not isinstance(relative, str) or any(token in relative.casefold() for token in ["\u955c\u50cf", "mirror"]):
            raise ValueError("translated or ambiguous PDF rejected")
        path = contained(vault, relative)
        if not path.is_file():
            raise FileNotFoundError("explicit original source missing: " + relative)
        digest = sha(path)
        if digest in seen_sources:
            raise ValueError("same physical publication selected twice")
        seen_sources.add(digest)
        protected[relative] = digest
        card = node.get("card")
        if not isinstance(card, str) or not contained(vault, card).is_file():
            raise ValueError("paper card absent")
        protected[card] = sha(contained(vault, card))
        papers.append({"id": identifier, "title": node["label"], "card": card,
                       "pdf": relative, "pdf_sha256": digest,
                       "existing_domains": node.get("domains", []),
                       "recorded_authors": node.get("metadata", {}).get("authors", []),
                       "author_state": "retained_graph_record_not_new_source_verification"})
    reusable = [{"id": n["id"], "type": n["type"], "label": n["label"], "card": n.get("card")}
                for n in graph["nodes"] if n["type"] in {"question", "method", "task", "dataset"}]
    manifest = {"schema": "paper-pipeline-manifest-v1", "papers": papers,
                "protected": protected, "existing_entities": reusable,
                "scope": "working_only_no_formal_publication"}
    write_json(manifest_path, manifest, output)
    return manifest


def check_inputs(vault: Path, manifest: dict[str, Any]) -> None:
    for relative, expected in manifest["protected"].items():
        path = contained(vault, relative)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen input changed: " + relative)


class Jobs:
    def __init__(self, path: Path, boundary: Path | None = None):
        boundary = boundary or path.parent
        for suffix in ["", "-wal", "-shm", "-journal"]:
            checked_output(Path(str(path) + suffix), boundary)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, stage TEXT NOT NULL,
            input_hash TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            lease_until REAL, result TEXT, result_hash TEXT, error TEXT)""")
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(jobs)")}
        if "lease_token" not in columns:
            self.connection.execute("ALTER TABLE jobs ADD COLUMN lease_token TEXT")
        self.tokens: dict[str, str] = {}

    def close(self) -> None:
        self.connection.close()

    def enqueue(self, paper_id: str, stage: str, key: str) -> str:
        identifier = object_hash([paper_id, stage, key])
        self.connection.execute("INSERT OR IGNORE INTO jobs(id,paper_id,stage,input_hash,state) VALUES(?,?,?,?, 'pending')",
                                (identifier, paper_id, stage, key))
        return identifier

    def claim(self, identifier: str, lease_seconds: float = 3600) -> bool:
        now = time.time()
        token = uuid.uuid4().hex
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute("""UPDATE jobs SET state='running', lease_until=?, lease_token=?, attempts=attempts+1,
                error=NULL WHERE id=? AND (state='pending' OR (state='running' AND lease_until<?))""",
                (now + lease_seconds, token, identifier, now))
            self.connection.execute("COMMIT")
            if cursor.rowcount == 1:
                self.tokens[identifier] = token
            return cursor.rowcount == 1
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def finish(self, identifier: str, result: str, result_hash: str) -> None:
        cursor = self.connection.execute("UPDATE jobs SET state='done',result=?,result_hash=?,lease_until=NULL,lease_token=NULL WHERE id=? AND state='running' AND lease_token=?",
                                        (result, result_hash, identifier, self.tokens.get(identifier)))
        if cursor.rowcount != 1:
            raise ValueError("lease ownership lost")
        self.tokens.pop(identifier, None)

    def fail(self, identifier: str, error: str) -> None:
        cursor = self.connection.execute("UPDATE jobs SET state='failed',error=?,lease_until=NULL,lease_token=NULL WHERE id=? AND state='running' AND lease_token=?",
                                        (error, identifier, self.tokens.get(identifier)))
        if cursor.rowcount != 1:
            raise ValueError("lease ownership lost")
        self.tokens.pop(identifier, None)

    def invalidate(self, identifier: str) -> None:
        self.connection.execute("UPDATE jobs SET state='pending',result=NULL WHERE id=? AND state='done'", (identifier,))

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM jobs ORDER BY paper_id,stage")]

    def get(self, identifier: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("job not found")
        return dict(row)


def parse_key(paper: dict[str, Any]) -> str:
    import pymupdf
    return object_hash([paper["pdf_sha256"], PARSER_VERSION, pymupdf.VersionBind, sha(Path(__file__))])


def parse_pdf(path: str, expected_sha: str, title: str, identifier: str) -> dict[str, Any]:
    import pymupdf
    source = Path(path)
    if sha(source) != expected_sha:
        raise ValueError("source changed before parsing")
    pages = []
    with pymupdf.open(source) as document:
        for page_number, page in enumerate(document, 1):
            blocks = []
            text = ""
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0 or not block[4].strip():
                    continue
                start = len(text)
                value = block[4].strip()
                text += value + "\n\n"
                blocks.append({"text": value, "start": start, "end": start + len(value),
                               "bbox": list(block[:4])})
            pages.append({"page": page_number, "text": text, "blocks": blocks})
    if not pages or not any(page["text"].strip() for page in pages):
        raise ValueError("no extractable text; OCR required, not missing source")
    first = " ".join(page["text"] for page in pages[:2]).casefold()
    words = {word for word in re.findall(r"[a-z0-9]+", title.casefold()) if len(word) > 2}
    coverage = sum(word in first for word in words) / max(1, len(words))
    arxiv = identifier.removeprefix("paper:arxiv:") if identifier.startswith("paper:arxiv:") else None
    header_ids = re.findall(r"arxiv\s*:\s*(\d{4}\.\d{4,5})", pages[0]["text"], re.I)
    if header_ids and arxiv and arxiv not in header_ids:
        raise ValueError("PDF arXiv header conflicts with selected identity")
    if coverage < .6 and not (arxiv and arxiv in header_ids):
        raise ValueError("source title/identity unresolved")
    if sha(source) != expected_sha:
        raise ValueError("source changed during parsing")
    return {"parser": PARSER_VERSION, "pdf_sha256": expected_sha, "paper_id": identifier,
            "identity": {"state": "arxiv_header_match" if arxiv and arxiv in header_ids else "title_tokens_candidate",
                         "title_token_coverage": coverage}, "pages": pages}


def parse_manifest(vault: Path, output: Path, manifest: dict[str, Any], workers: int = 2) -> dict[str, Any]:
    output = output_root(vault, output)
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    check_inputs(vault, manifest)
    queue = Jobs(output / "state.sqlite", output)
    summary = {"parsed": 0, "cached": 0, "failed": [], "busy": []}
    started = time.perf_counter()
    futures = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for paper in manifest["papers"]:
                key = parse_key(paper)
                job = queue.enqueue(paper["id"], "parse", key)
                relative = "cache/" + key + ".json"
                path = output / relative
                state = queue.get(job)
                if state["state"] == "done":
                    try:
                        cached = read_json(path)
                        if (sha(path) != state["result_hash"] or cached["parser"] != PARSER_VERSION
                                or cached["pdf_sha256"] != paper["pdf_sha256"] or cached["paper_id"] != paper["id"]):
                            raise ValueError("cache binding mismatch")
                        summary["cached"] += 1
                        continue
                    except (OSError, ValueError, KeyError):
                        queue.invalidate(job)
                if not queue.claim(job):
                    if state["state"] == "failed":
                        summary["failed"].append({"paper_id": paper["id"], "error": state["error"]})
                    else:
                        summary["busy"].append(paper["id"])
                    continue
                future = executor.submit(parse_pdf, str(contained(vault, paper["pdf"])),
                                         paper["pdf_sha256"], paper["title"], paper["id"])
                futures[future] = (paper, job, relative)
            for future in as_completed(futures):
                paper, job, relative = futures[future]
                try:
                    result = future.result()
                    write_json(output / relative, result, output)
                    queue.finish(job, relative, sha(output / relative))
                    summary["parsed"] += 1
                except Exception as error:
                    description = type(error).__name__ + ": " + str(error)
                    queue.fail(job, description)
                    summary["failed"].append({"paper_id": paper["id"], "error": description})
        check_inputs(vault, manifest)
        summary["seconds"] = round(time.perf_counter() - started, 3)
        summary["jobs"] = queue.rows()
        return summary
    finally:
        queue.close()


def cache_path(output: Path, paper: dict[str, Any]) -> Path:
    return checked_output(output / "cache" / (parse_key(paper) + ".json"), output)
