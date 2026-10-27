"""Revision-pinned local CPU models; downloading and inference are separate steps."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

from .core import checked_output, contained, read_json, sha, write_json

DEFAULT_MODELS = Path(os.environ.get("LLM_WIKI_MODELS_DIR") or Path.home() / ".cache" / "llm-wiki" / "models").expanduser()
CHECKPOINTS = {
    "extractor": {"repo": "fastino/gliner2.5-small-v1", "revision": "df5910e44bc4ffdb0d95399a83b0ca4516349aa5",
                  "files": ["config.json", "encoder_config/config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]},
    "embedding": {"repo": "intfloat/multilingual-e5-small", "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                  "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"]},
}
LABELS = {
    "method": "A named algorithm, model, technique, solver or concrete computational mechanism described in the paper.",
    "question": "A research problem, limitation, difficulty or objective that a method aims to address.",
    "task": "A prediction, learning or decision-making task studied in the paper.",
    "dataset": "The name of a dataset, benchmark or experimental environment mentioned in the paper.",
}


def download_models(root: Path) -> dict:
    """Only public allowlisted checkpoint files; never submit source documents."""
    from huggingface_hub import snapshot_download
    root = root.resolve()
    manifest = {"schema": "paper-local-models-v1", "models": {}}
    for name, spec in CHECKPOINTS.items():
        folder = checked_output(root / (name + "-" + spec["revision"]), root)
        for relative in spec["files"]:
            checked_output(folder / relative, root)
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=folder,
                          allow_patterns=spec["files"], token=False, max_workers=3)
        files = {}
        for relative in spec["files"]:
            path = contained(folder, relative)
            if not path.is_file():
                raise FileNotFoundError("checkpoint file missing: " + relative)
            files[relative] = sha(path)
        manifest["models"][name] = {**spec, "directory": folder.relative_to(root).as_posix(), "sha256": files}
    write_json(root / "models.json", manifest, root)
    return manifest


def verify_models(root: Path) -> dict:
    manifest = read_json(root / "models.json")
    if set(manifest["models"]) != set(CHECKPOINTS):
        raise ValueError("unexpected local model manifest")
    for name, expected in CHECKPOINTS.items():
        model = manifest["models"][name]
        if any(model[k] != expected[k] for k in ("repo", "revision", "files")):
            raise ValueError("checkpoint identity differs from frozen model")
        if set(model["sha256"]) != set(expected["files"]):
            raise ValueError("checkpoint file inventory differs")
        folder = contained(root, model["directory"])
        for relative, digest in model["sha256"].items():
            if sha(contained(folder, relative)) != digest:
                raise ValueError("checkpoint file changed: " + name + "/" + relative)
    return manifest


class LocalModels:
    """Lazily loaded encoders. Runtime always uses local files, CPU and eval mode."""
    def __init__(self, root: Path, threads: int = 4):
        if not 1 <= threads <= 8:
            raise ValueError("CPU threads must be between 1 and 8")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.root = root.resolve()
        self.manifest = verify_models(self.root)
        self.identity = {"models": self.manifest["models"], "device": "cpu", "threads": threads,
                         "versions": {name: version(name) for name in ("torch", "transformers", "gliner2", "numpy",
                                                                        "tokenizers", "sentencepiece", "safetensors")}}
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.extractor = self.tokenizer = self.encoder = None
        self.extraction_calls = self.embedding_calls = 0

    def extract(self, text: str, threshold: float) -> dict:
        if self.extractor is None:
            from gliner2 import AutoExtractor
            path = contained(self.root, self.manifest["models"]["extractor"]["directory"])
            self.extractor = AutoExtractor.from_pretrained(str(path), map_location="cpu", local_files_only=True)
            self.extractor.eval()
            if getattr(self.extractor.config, "architecture", None) != "boundary":
                raise ValueError("GLiNER2.5 must load the frozen boundary architecture")
        self.extraction_calls += 1
        with self.torch.inference_mode():
            return self.extractor.extract_entities(text, LABELS, threshold=threshold,
                                                   include_spans=True, include_confidence=True)

    def encode(self, texts: list[str], prefix: str, batch_size: int = 16):
        import numpy as np
        if prefix not in {"query", "passage"}:
            raise ValueError("E5 input prefix must be query or passage")
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if self.encoder is None:
            from transformers import AutoModel, AutoTokenizer
            path = contained(self.root, self.manifest["models"]["embedding"]["directory"])
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self.encoder = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                     use_safetensors=True).to("cpu").eval()
        vectors = []
        for offset in range(0, len(texts), batch_size):
            batch = [prefix + ": " + text for text in texts[offset:offset + batch_size]]
            inputs = self.tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors="pt")
            self.embedding_calls += 1
            with self.torch.inference_mode():
                states = self.encoder(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(states.dtype)
                pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
        result = np.concatenate(vectors)
        if result.shape != (len(texts), 384) or not np.isfinite(result).all():
            raise ValueError("invalid embedding output")
        if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-4):
            raise ValueError("embeddings must be normalized and nonzero")
        return result

    def runtime_evidence(self) -> dict:
        return {name: {"class": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
                       "device": str(next(model.parameters()).device), "training": model.training}
                for name, model in (("extractor", self.extractor), ("embedding", self.encoder)) if model is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify"])
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    manifest = download_models(args.models) if args.command == "download" else verify_models(args.models)
    print(json.dumps({"state": "models_ready", "models": {k: {"repo": v["repo"], "revision": v["revision"]}
                                                           for k, v in manifest["models"].items()}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
