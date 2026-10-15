"""Prepare lossless TP2 safetensors caches using the native inspector's authoritative partition plan."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from dataclasses import dataclass


DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "I16": 2, "I32": 4, "I64": 8, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1}
MODEL_REVISION = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
FORMAT_VERSION = 2


def safe_shard(name):
    return isinstance(name, str) and name.endswith(".safetensors") and not name.startswith(".") and not any(
        c in name for c in "/\\:")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def catalog(source):
    index = json.loads((source / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    names = set(index.values())
    if not names or not all(safe_shard(n) for n in names):
        raise ValueError("unsafe or empty checkpoint index")
    out = {}
    for name in sorted(names):
        path = source / name
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated header: {name}")
            size, = struct.unpack("<Q", raw)
            if size > 64 << 20 or size > path.stat().st_size - 8:
                raise ValueError(f"oversized header: {name}")
            header = json.loads(stream.read(size))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in out or index.get(key) != name:
                raise ValueError(f"duplicate tensor or index mismatch: {key}")
            out[key] = dict(value, file=name, data_start=8 + size)
    if set(out) != set(index):
        raise ValueError("checkpoint shards are incomplete")
    return out


def product(shape):
    result = 1
    for size in shape:
        result *= size
    return result


@dataclass(frozen=True, slots=True)
class Spans:
    offset: int
    stride: int
    length: int
    count: int

    def __iter__(self):
        for row in range(self.count):
            yield self.offset + row * self.stride, self.length

    @property
    def byte_count(self):
        return self.length * self.count


def ranges(shape, itemsize, axis=None, first=0, count=0):
    if axis is None:
        return Spans(0, 0, product(shape) * itemsize, 1), list(shape)
    if not isinstance(axis, int) or axis < 0 or axis >= len(shape) or first < 0 or count <= 0 or first + count > shape[axis]:
        raise ValueError("slice outside tensor")
    inside = product(shape[axis + 1:]) * itemsize
    outside = product(shape[:axis])
    stride = shape[axis] * inside
    selected = Spans(first * inside, stride, count * inside, outside)
    local = list(shape)
    local[axis] = count
    return selected, local


def selection(catalog_, plan, rank):
    chosen = {}

    def add(name, axis=None, first=0, count=0):
        if name in chosen:
            raise ValueError(f"duplicate planned tensor: {name}")
        tensor = catalog_[name]
        dtype = tensor["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported cache dtype: {dtype}: {name}")
        spans, shape = ranges(tensor["shape"], DTYPE_BYTES[dtype], axis, first, count)
        chosen[name] = dict(tensor=tensor, spans=spans, shape=shape)

    if plan.get("version") == 2:
        for item in plan["payloads"]:
            part = item["ranks"][rank]
            if not part["keep"]:
                continue
            name = item["name"]
            tensor = catalog_[name]
            if tensor["dtype"].lower() != item["dtype"] or tensor["shape"] != item["shape"]:
                raise ValueError(f"inspector/source tensor mismatch: {name}")
            add(name, part["axis"], part["first"], part["count"])
        return chosen
    for item in plan["items"]:
        spec, part = item["spec"], item["ranks"][rank]
        if not part["keep"]:
            continue
        key, axis, first, count = spec["key"], part["axis"], part["first"], part["count"]
        if spec["kind"] != "matrix":
            add(key, axis, first, count)
        elif key + ".trellis" in catalog_:
            add(key + ".trellis", axis, first // 16, count // 16)
            for suffix, affected in (("suh", 0), ("svh", 1)):
                name = key + "." + suffix
                packed = catalog_[name]["dtype"] == "I16"
                factor = 16 if packed else 1
                add(name, 0 if axis == affected else None, first // factor, count // factor)
            for codebook in ("mul1", "mcg", "inst3"):
                name = key + "." + codebook
                if name in catalog_:
                    add(name)
        else:
            add(key + ".weight", 1 - axis if axis is not None else None, first, count)
    return chosen


def write_rank(source, output, chosen):
    header, offset = {}, 0
    for name in sorted(chosen):
        entry = chosen[name]
        size = entry["spans"].byte_count
        header[name] = dict(dtype=entry["tensor"]["dtype"], shape=entry["shape"], data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    temporary = output.with_suffix(".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(output)
    digest = hashlib.sha256()
    with ExitStack() as stack:
        inputs = {name: stack.enter_context((source / name).open("rb"))
                  for name in {x["tensor"]["file"] for x in chosen.values()}}
        dst = stack.enter_context(temporary.open("xb"))

        def write(block):
            dst.write(block)
            digest.update(block)

        write(struct.pack("<Q", len(encoded)))
        write(encoded)
        for name in sorted(chosen):
            entry = chosen[name]
            tensor = entry["tensor"]
            src = inputs[tensor["file"]]
            for start, length in entry["spans"]:
                src.seek(tensor["data_start"] + tensor["data_offsets"][0] + start)
                while length:
                    data = src.read(min(length, 8 << 20))
                    if not data:
                        raise ValueError(f"truncated tensor: {name}")
                    write(data)
                    length -= len(data)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, output)
    return dict(file=output.name, bytes=output.stat().st_size, sha256=digest.hexdigest(), tensors=len(chosen))


def prepare(source, output, inspector):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("cache must be outside the immutable source checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use an empty cache directory; existing cache files are never overwritten")
    data = subprocess.run([str(inspector.resolve()), str(source), "--plan"], check=True, capture_output=True)
    plan = json.loads(data.stdout)
    tensors = catalog(source)
    names = sorted({x["file"] for x in tensors.values()})
    metadata = [n for n in ("config.json", "model.safetensors.index.json", "quantization_config.json",
                            "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")
                if (source / n).is_file()]
    if "tokenizer.json" not in metadata or "tokenizer_config.json" not in metadata:
        raise ValueError("checkpoint must include its tokenizer files")
    print("Hashing immutable source checkpoint...", flush=True)
    hashes = {name: sha256(source / name) for name in sorted(names + metadata)}
    identity = dict(storage=plan["storage"], cache_code_sha256=plan["cache_code_sha256"], partition_version=2, format_version=FORMAT_VERSION, model_revision=MODEL_REVISION, tp=2, sources=hashes,
                    plan_sha256=hashlib.sha256(data.stdout).hexdigest())
    identity_text = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity_hash = hashlib.sha256(identity_text.encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    ranks = []
    for rank in (0, 1):
        print(f"Writing rank {rank} lossless slices...", flush=True)
        ranks.append(write_rank(source, output / f"rank{rank}.safetensors", selection(tensors, plan, rank)))
    # Detect a source changed during packing rather than publishing a mixed-revision cache.
    for name, expected in hashes.items():
        if sha256(source / name) != expected:
            raise ValueError(f"source changed while preparing cache: {name}")
    manifest = dict(identity=identity_text, identity_sha256=identity_hash, ranks=ranks,
                    kv_memory=plan["kv_memory"], model_qualified=False)
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inspector", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.inspector)
    print(json.dumps({"identity": result["identity_sha256"], "ranks": result["ranks"]}, indent=2))


if __name__ == "__main__":
    main()
