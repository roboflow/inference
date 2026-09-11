"""Rewrites the 11 uniform LMMInferenceRequest call sites onto `run_lmm`.

Each site is one of two shapes:

  A)  request = LMMInferenceRequest(api_key=..., model_id=..., image=...,
                                    source="workflow-execution", prompt=...)
      prediction = self._model_manager.infer_from_request_sync(
          model_id=..., request=request)

  B)  request_kwargs = dict(api_key=..., model_id=..., image=...,
                            source="workflow-execution", prompt=...,
                            [enable_thinking=...])
      if max_new_tokens is not None:
          request_kwargs["max_new_tokens"] = max_new_tokens
      request = LMMInferenceRequest(**request_kwargs)
      prediction = self._model_manager.infer_from_request_sync(
          model_id=..., request=request)

Both collapse to one `self._model_manager.run_lmm(...)` carrying the same
argument expressions. The script reads the arguments off the AST, edits with a
`\r?\n`-safe regex, re-parses before writing, and refuses a file whose shape it
does not recognise - so a drifted site is reported, never mangled. It does NOT
touch `add_model` (registration stays in the block) or the `prediction.response`
read that follows; both are handled by hand in the next step.

Run:  python scripts/phase11_swap_lmm_call_sites.py <file> [<file> ...]
"""

import ast
import re
import sys
from pathlib import Path

_SOURCE_LITERAL = "workflow-execution"


def _kwargs_of(call: ast.Call, source: str) -> dict:
    return {
        keyword.arg: ast.get_source_segment(source, keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None
    }


def rewrite(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LMMInferenceRequest"
    ]
    if not calls:
        # Already rewritten. An AST check, not a grep: the class name survives
        # in comments and docstrings (`qwen/v1.py:265`, `qwen_vlm/v1.py:203`),
        # so a text test would make the second run abort instead of skip.
        return False
    if len(calls) != 1:
        raise SystemExit(
            f"{path}: expected exactly 1 LMMInferenceRequest, got {len(calls)}"
        )
    kwargs = _kwargs_of(calls[0], source)
    if not kwargs:  # shape B keeps its arguments in a preceding dict(...)
        dict_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dict"
            and any(
                keyword.arg == "source"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == _SOURCE_LITERAL
                for keyword in node.keywords
            )
        ]
        if len(dict_calls) != 1:
            raise SystemExit(f"{path}: expected exactly one request_kwargs dict()")
        kwargs = _kwargs_of(dict_calls[0], source)
    for required in ("api_key", "model_id", "image", "prompt"):
        if required not in kwargs:
            raise SystemExit(f"{path}: missing {required} in the request kwargs")
    arguments = [
        f"model_id={kwargs['model_id']}",
        f"image={kwargs['image']}",
        f"prompt={kwargs['prompt']}",
        f"api_key={kwargs['api_key']}",
    ]
    if "enable_thinking" in kwargs:
        arguments.append(f"enable_thinking={kwargs['enable_thinking']}")
    if "max_new_tokens" in source:
        arguments.append("max_new_tokens=max_new_tokens")
    replacement = (
        "            prediction = self._model_manager.run_lmm(\n"
        + "".join(f"                {argument},\n" for argument in arguments)
        + "            )\n"
    )
    pattern = re.compile(
        r"[ \t]*(?:request_kwargs[^\n]*=\s*dict\(|request\s*=\s*LMMInferenceRequest\()"
        r".*?infer_from_request_sync\(\r?\n.*?\r?\n[ \t]*\)\r?\n",
        re.DOTALL,
    )
    new_source, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise SystemExit(f"{path}: could not locate the call block to replace")
    new_source = re.sub(
        r"from inference\.core\.entities\.requests\.inference import LMMInferenceRequest\r?\n",
        "",
        new_source,
    )
    ast.parse(new_source)  # refuse to write anything that does not parse
    path.write_text(new_source, encoding="utf-8")
    return True


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        print(argument, "rewritten" if rewrite(Path(argument)) else "skipped")
