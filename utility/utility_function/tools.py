"""Configuration reading and hashing (configure/<Model>.txt, key = value format)."""
import ast
import hashlib
import json
from pathlib import Path

_LITERALS = {"none": None, "null": None, "true": True, "false": False}


def parse_value(text):
    """Typed value of one configuration entry: none/null, true/false, numbers and lists; anything else is a string."""
    text = text.strip()
    if text.lower() in _LITERALS:
        return _LITERALS[text.lower()]
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def read_configuration(path):
    """Read `key = value` lines (blank lines and lines starting with # are ignored); duplicate keys are an error."""
    config = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("Configuration format error at {}:{}".format(path, line_number))
            name, value = line.split("=", 1)
            name = name.strip()
            if name in config:
                raise ValueError("Duplicate configuration key {} at {}:{}".format(name, path, line_number))
            config[name] = parse_value(value)
    return config


CODE_DIRS = ("config_path", "configure", "models", "utility", "tools", "tests")
CODE_ROOT_FILES = ("main.py", "Parser.py", "requirements.txt")


def code_hash(root=None):
    """SHA-256 of the code and default configurations of this codebase (relative POSIX path + bytes of every *.py,
    *.txt and *.json file under CODE_DIRS, plus CODE_ROOT_FILES). Third-party environments, data and outputs are out."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    files = [root / f for f in CODE_ROOT_FILES if (root / f).is_file()]
    for d in CODE_DIRS:
        files += [p for p in (root / d).rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json")
                  and "__pycache__" not in p.parts]
    digest = hashlib.sha256()
    for p in sorted(files, key=lambda q: q.relative_to(root).as_posix()):
        digest.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(p.read_bytes() + b"\0")
    return digest.hexdigest()


def config_hash(config):
    """SHA-256 of the canonical JSON of a resolved configuration (frozen with the protocol)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
