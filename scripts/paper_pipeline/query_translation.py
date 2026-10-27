"""Pinned local query translation, with explicit cache and constraint rejection."""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request
import zipfile

from .core import checked_output, contained, object_hash, read_json, sha, write_json

ROOT = Path(os.environ.get("LLM_WIKI_TRANSLATION_MODEL_DIR") or Path.home() / ".cache" / "llm-wiki" / "query-models").expanduser()
REPO = "Qwen/Qwen3-1.7B-GGUF"
REVISION = "90862c4b9d2787eaed51d12237eafdfe7c5f6077"
MODEL_FILE = "Qwen3-1.7B-Q8_0.gguf"
RUNTIME_TAG = "b11669"
ARCHIVE_SHA = "37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b"
SYSTEM = ("Translate the user's Chinese research search query into concise English. "
          "Preserve every condition, negation, quantity and technical name exactly in meaning. "
          "Do not answer the query, add methods, broaden the task or explain. "
          "Return only the English query, without commentary.")
ENGINE = "qwen17-query-translation-v2-glossary"
SPECS = {
    "qwen17": {"repo": REPO, "revision": REVISION, "file": MODEL_FILE, "manifest": "translator.json", "engine": ENGINE},
    "hymt2": {"repo": "tencent/Hy-MT2-1.8B-GGUF", "revision": "a0c709d9fac510f2c807aa3af52872340dc37a4a",
              "file": "Hy-MT2-1.8B-Q8_0.gguf", "manifest": "translator-hymt2.json", "engine": "hymt2-query-translation-v1"},
}
# Query terminology, not new graph identities. Equilibrium/credit terminology
# reuses the frozen pilot source titles and shared problem definitions.
GLOSSARY = [("深度平衡模型", "deep equilibrium model"), ("固定点", "fixed point"),
            ("信用分配", "credit assignment"), ("图拉普拉斯", "graph Laplacian"),
            ("语义分割", "semantic segmentation"), ("目标检测", "object detection")]


def query_glossary(original: str) -> list[tuple[str, str]]:
    terms = [(cn, en) for cn, en in GLOSSARY if cn in original]
    if ("推理" in original and any(t in original for t in ("微调", "参数", "训练", "辅助", "冻结"))
            and not any(t in original for t in ("多步推理", "链式推理", "逻辑推理", "推理链", "推理能力"))):
        terms.append(("推理", "inference"))
    return terms


def build_messages(original: str, protected: list[str], glossary: list[tuple[str, str]], model_key: str) -> list[dict]:
    if model_key == "hymt2":
        terms = list(glossary) + [(term, term) for term in protected if term not in {en for _, en in glossary}]
        prefix = "参考下面的翻译：\n" + "\n".join(cn + " 翻译成 " + en for cn, en in terms) + "\n\n" if terms else ""
        return [{"role": "user", "content": prefix + "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n" + original}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": original
            + ("\nTechnical glossary: " + "; ".join(cn + " = " + en for cn, en in glossary) if glossary else "")
            + ("\nKeep these strings: " + ", ".join(protected) if protected else "")}]


def download(root: Path = ROOT, model_key: str = "qwen17") -> dict:
    from huggingface_hub import hf_hub_download
    root = root.resolve()
    spec = SPECS[model_key]
    model_dir = checked_output(root / ("model" if model_key == "qwen17" else "model-" + model_key), root)
    checked_output(model_dir / spec["file"], root)
    hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"], local_dir=model_dir, token=False)
    archive = checked_output(root / (RUNTIME_TAG + ".zip"), root)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".download")
        checked_output(partial, root)
        urllib.request.urlretrieve("https://github.com/ggml-org/llama.cpp/releases/download/" + RUNTIME_TAG
                                   + "/llama-" + RUNTIME_TAG + "-bin-win-cpu-x64.zip", partial)
        if sha(partial) != ARCHIVE_SHA:
            raise ValueError("official llama.cpp archive hash mismatch")
        os.replace(partial, archive)
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError("runtime archive changed")
    runtime = checked_output(root / ("llama-" + RUNTIME_TAG), root)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            checked_output(runtime / item.filename, root)
            if ".." in Path(item.filename).parts or Path(item.filename).is_absolute():
                raise ValueError("runtime archive path traversal")
        zipped.extractall(runtime)
    executables = list(runtime.rglob("llama-server.exe"))
    if len(executables) != 1:
        raise ValueError("one pinned llama-server is required")
    value = {"schema": "local-query-translator-v1", "repo": spec["repo"], "revision": spec["revision"],
             "model": str((model_dir / spec["file"]).relative_to(root)), "model_sha256": sha(model_dir / spec["file"]),
             "runtime_tag": RUNTIME_TAG, "archive_sha256": ARCHIVE_SHA,
             "executable": str(executables[0].relative_to(root)),
             "runtime_files": {str(p.relative_to(root)): sha(p) for p in runtime.rglob("*") if p.is_file()}}
    write_json(root / spec["manifest"], value, root)
    return value


def verify(root: Path = ROOT, model_key: str = "qwen17") -> dict:
    spec = SPECS[model_key]
    value = read_json(root / spec["manifest"])
    if (value["schema"] != "local-query-translator-v1" or value["repo"] != spec["repo"] or value["revision"] != spec["revision"]
            or value["runtime_tag"] != RUNTIME_TAG or value["archive_sha256"] != ARCHIVE_SHA):
        raise ValueError("translator identity differs")
    if sha(contained(root, value["model"])) != value["model_sha256"]:
        raise ValueError("translator model changed")
    if value["executable"] not in value["runtime_files"]:
        raise ValueError("runtime executable is not fingerprinted")
    for name, digest in value["runtime_files"].items():
        if sha(contained(root, name)) != digest:
            raise ValueError("translator runtime changed")
    return value


def constraint_errors(original: str, english: str, protected: list[str], required: list[list[str]], finish: str = "stop") -> list[str]:
    errors = []
    if not english.strip() or len(english) > 1800 or re.search(r"[\u4e00-\u9fff]", english):
        errors.append("empty_long_or_untranslated_output")
    if re.search(r"</?think>|^```", english, re.I):
        errors.append("reasoning_or_fenced_output")
    for term in protected:
        pattern = re.escape(term).replace(r"\ ", r"[\s-]+")
        if not re.search(pattern, english, re.I):
            errors.append("protected_term_missing:" + term)
    for alternatives in required:
        if not any(re.search(pattern, english, re.I) for pattern in alternatives):
            errors.append("required_meaning_marker_missing:" + "|".join(alternatives))
    if finish != "stop":
        errors.append("incomplete_generation:" + str(finish))
    return errors


class LocalTranslator(AbstractContextManager):
    def __init__(self, cache: Path, root: Path = ROOT, threads: int = 4, model_key: str = "qwen17"):
        if not 1 <= threads <= 8:
            raise ValueError("threads must be between 1 and 8")
        self.root, self.cache, self.model_key = root.resolve(), cache.resolve(), model_key
        self.identity = {"checkpoint": verify(self.root, model_key), "engine": SPECS[model_key]["engine"], "threads": threads,
                         "context": 2048, "max_tokens": 192, "temperature": 0, "seed": 0,
                         "thinking": False, "system": SYSTEM if model_key == "qwen17" else "official_hymt2_terminology_template",
                         "glossary": GLOSSARY,
                         "implementation_sha256": sha(Path(__file__))}
        self.process = None
        self.calls = self.hits = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path: str, data=None):
        request = urllib.request.Request(self.url + path, data=None if data is None else json.dumps(data).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=180) as response:
            return json.load(response)

    def _start(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError("local translator process exited")
            return
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = "http://127.0.0.1:" + str(port)
        model = self.identity["checkpoint"]
        command = [str(contained(self.root, model["executable"])), "--model", str(contained(self.root, model["model"])),
                   "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "2048", "--parallel", "1",
                   "--threads", str(self.identity["threads"]), "--threads-batch", str(self.identity["threads"]),
                   "--n-gpu-layers", "0", "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}',
                   "--no-webui", "--no-warmup", "--alias", "llm-wiki-query"]
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("local translator startup failed: " + str(self.process.returncode))
            try:
                if self._request("/health").get("status") == "ok":
                    return
            except (OSError, ValueError):
                time.sleep(0.2)
        raise TimeoutError("local translator startup timed out")

    def translate(self, original: str, protected: list[str] | None = None,
                  required: list[list[str]] | None = None) -> dict:
        if not original.strip() or len(original) > 1200:
            raise ValueError("short nonempty query required")
        protected, required = protected or [], required or []
        glossary = query_glossary(original)
        protected = list(dict.fromkeys([*protected, *[en for _, en in glossary]]))
        key = object_hash([self.identity, original, protected, required])
        path = checked_output(self.cache / (key + ".json"), self.cache)
        if path.exists():
            result = read_json(path)
            if (result["cache_key"] != key or result["original"] != original
                    or result["errors"] != constraint_errors(original, result["english"], protected, required, result["finish_reason"])
                    or result["result_sha256"] != object_hash({k: v for k, v in result.items() if k != "result_sha256"})):
                raise ValueError("translation cache validation failed")
            self.hits += 1
            return result
        failure = checked_output(self.cache / "failures" / (key + ".json"), self.cache)
        if failure.exists():
            raise RuntimeError("previous translation failure retained: " + str(failure))
        started = time.perf_counter()
        payload = {"model": "llm-wiki-query", "messages": build_messages(original, protected, glossary, self.model_key),
                   "temperature": 0, "seed": 0, "max_tokens": 192, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        try:
            self._start()
            self.calls += 1
            raw = self._request("/v1/chat/completions", payload)
            choice = raw["choices"][0]
            english = choice["message"].get("content", "").strip()
            finish = choice["finish_reason"]
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as error:
            write_json(failure, {"cache_key": key, "original": original, "protected": protected, "required": required,
                                "engine": self.identity["engine"], "error": type(error).__name__ + ": " + str(error)}, self.cache)
            raise
        errors = constraint_errors(original, english, protected, required, finish)
        result = {"cache_key": key, "original": original, "english": english, "protected": protected,
                  "required": required, "errors": errors, "state": "rejected" if errors else "candidate_translation",
                  "meaning_review": "pending", "glossary": glossary, "finish_reason": finish, "usage": raw.get("usage", {}),
                  "seconds": round(time.perf_counter() - started, 3), "engine": self.identity["engine"]}
        result["result_sha256"] = object_hash(result)
        write_json(path, result, self.cache)
        return result

    def __exit__(self, *args):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify", "translate"])
    parser.add_argument("--models", type=Path, default=ROOT)
    parser.add_argument("--translator", choices=list(SPECS), default="qwen17")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--query")
    args = parser.parse_args()
    if args.command == "download":
        value = download(args.models, args.translator)
    elif args.command == "verify":
        value = verify(args.models, args.translator)
    else:
        if args.cache is None or args.query is None:
            parser.error("translation requires --cache and --query")
        with LocalTranslator(args.cache, args.models, model_key=args.translator) as translator:
            value = translator.translate(args.query)
    print(json.dumps({"state": "ready", "repo": value.get("repo"), "revision": value.get("revision"),
                      "translation": value if args.command == "translate" else None}, ensure_ascii=True))


if __name__ == "__main__":
    main()
