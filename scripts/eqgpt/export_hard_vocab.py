"""Export the frozen downstream hard AST-BPE vocabulary in model ID order."""
from collections import Counter
from html import escape
from pathlib import Path

from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/eqgpt_fast_reward_20261022/tokenizers"


def display(payload: bytes) -> str:
    return repr(payload.decode("utf-8", errors="backslashreplace"))


def main() -> None:
    for size in (280, 315, 350):
        source = RUN / f"ast_bpe_{size}.model.json"
        output = ROOT / f"reports/hard_ast_bpe_{size}_vocabulary.md"
        model = CompactAlphabet.load(source)
        counts = Counter(model.vocab)
        assert len(model.vocab) == size - 3 and len(counts) == size - 3
        lines = [
            f"# Hard AST-BPE {size}-ID vocabulary",
            "",
            f"Frozen model: `{source.relative_to(ROOT)}`.",
            "IDs 0-2 are control tokens; content token ID = compact vocabulary index + 3.",
            "Escapes use Python string notation; e.g. `\\\\` represents one backslash byte.",
            "",
            "| ID | Token bytes | Kind |",
            "|---:|---|---|",
            "| 0 | `<PAD>` | control |",
            "| 1 | `<EOS>` | control |",
            "| 2 | `<BOS>` | control |",
        ]
        for index, payload in enumerate(model.vocab, start=3):
            kind = "base byte" if len(payload) == 1 else "learned merge"
            shown = escape(display(payload)).replace("|", "&#124;")
            lines.append(f"| {index} | <code>{shown}</code> | {kind} |")
        output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
