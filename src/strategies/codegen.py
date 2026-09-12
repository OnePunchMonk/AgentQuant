"""Constrained strategy code generation and registration."""
import ast
from typing import Iterable, Tuple
from src.strategies.base import Strategy

ALLOWED_IMPORTS = {"pandas", "numpy"}

class StrategyCodeValidator:
    def __init__(self, allowed_imports: Iterable[str] = ALLOWED_IMPORTS):
        self.allowed_imports = set(allowed_imports)

    def validate(self, source: str) -> Tuple[bool, str]:
        try: tree = ast.parse(source)
        except SyntaxError as exc: return False, f"syntax error: {exc}"
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        if len(classes) != 1: return False, "source must define exactly one class"
        cls = classes[0]
        if not any(isinstance(b, ast.Name) and b.id == "Strategy" for b in cls.bases):
            return False, "class must inherit Strategy"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                root = (node.names[0].name if isinstance(node, ast.Import) else node.module or "").split(".")[0]
                if root not in self.allowed_imports: return False, f"import not allowed: {root}"
            if isinstance(node, (ast.Call, ast.Attribute)) and isinstance(getattr(node, "func", None), ast.Name) and node.func.id in {"exec", "eval", "__import__"}:
                return False, "dynamic execution is not allowed"
        if not any(isinstance(n, ast.FunctionDef) and n.name == "generate_signal" for n in cls.body):
            return False, "class must implement generate_signal"
        return True, cls.name

def register_strategy_source(source: str, name: str, registry: dict) -> str:
    ok, detail = StrategyCodeValidator().validate(source)
    if not ok: raise ValueError(detail)
    namespace = {"Strategy": Strategy}
    exec(compile(ast.parse(source), "<generated_strategy>", "exec"), {"__builtins__": {"int": int, "float": float, "len": len}}, namespace)
    candidate = namespace[detail]()
    if not isinstance(candidate, Strategy): raise TypeError("generated class is not a Strategy")
    registry[name] = candidate
    return name
