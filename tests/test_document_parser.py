"""
Offline unit tests for document_parser.py helpers.
No PDF parsing, no network, no ML models - just the pure-logic pieces:
text chunking, table-HTML -> DataFrame -> markdown, and metadata shaping.
"""
from src.document_parser import (
    Chunk,
    _chunk_text,
    _dataframe_to_markdown,
    _flatten_columns,
    _table_html_to_dataframe,
)

MERGED_HEADER_HTML = """
<table>
<tr><th colspan=2>Region</th><th rowspan=2>Total</th></tr>
<tr><th>North</th><th>South</th></tr>
<tr><td>10</td><td>20</td><td>30</td></tr>
</table>
"""


def test_chunk_text_respects_size_and_overlap():
    text = "x" * 2000
    chunks = _chunk_text(text, size=800, overlap=120)
    assert len(chunks) > 1
    # every chunk except possibly the last should be exactly `size` long
    assert all(len(c) == 800 for c in chunks[:-1])


def test_chunk_text_returns_single_chunk_when_short():
    text = "short text"
    assert _chunk_text(text, size=800, overlap=120) == [text]


def test_table_html_with_merged_headers_parses_to_dataframe():
    df = _table_html_to_dataframe(MERGED_HEADER_HTML)
    assert df is not None
    assert df.shape == (1, 3)


def test_flatten_columns_collapses_multiindex_readably():
    df = _table_html_to_dataframe(MERGED_HEADER_HTML)
    flat = _flatten_columns(df)
    assert list(flat.columns) == ["Region - North", "Region - South", "Total"]


def test_dataframe_to_markdown_produces_clean_table():
    df = _table_html_to_dataframe(MERGED_HEADER_HTML)
    md = _dataframe_to_markdown(df)
    assert "Region - North" in md
    assert "|" in md  # markdown table syntax
    assert "(" not in md  # no leftover tuple repr from a raw MultiIndex


def test_table_html_to_dataframe_returns_none_on_garbage_input():
    assert _table_html_to_dataframe("<not a table>") is None


def test_chunk_to_metadata_is_flat_and_chroma_safe():
    chunk = Chunk(
        id="abc123",
        type="table",
        content="| a |\n|---|\n| 1 |",
        page_number=4,
        source="report.pdf",
        html="<table></table>",
    )
    meta = chunk.to_metadata()
    assert meta == {
        "type": "table",
        "page_number": 4,
        "source": "report.pdf",
        "image_path": "",
        "has_html": True,
    }
    # every value must be a scalar (str/int/bool) - Chroma metadata requirement
    assert all(isinstance(v, (str, int, bool)) for v in meta.values())
