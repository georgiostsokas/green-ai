"""The dashboard shows packaged copies of the docs. They must match the repository docs."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "doc,packaged",
    [("docs/METHODOLOGY.md", "methodology.md"), ("docs/AA_INTEGRATION.md", "aa_integration.md")],
)
def test_packaged_docs_match_repository_docs(doc, packaged):
    source = (ROOT / doc).read_text(encoding="utf-8")
    copy = (ROOT / "src/green_ai/data" / packaged).read_text(encoding="utf-8")
    assert copy == source, f"Copy {doc} to src/green_ai/data/{packaged}"


def test_task_example_is_shipped_in_the_package():
    a = (ROOT / "examples/task_profiles.csv").read_text(encoding="utf-8")
    b = (ROOT / "src/green_ai/data/examples/task_profiles.csv").read_text(encoding="utf-8")
    assert a == b
