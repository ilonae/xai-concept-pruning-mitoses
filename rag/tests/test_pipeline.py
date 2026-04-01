import io
from pathlib import Path
import pytest
import fitz  # pymupdf


def make_fixture_pdf(path: Path, pages: list[str]):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), text, fontsize=12)
    doc.save(str(path))
    doc.close()


class TestLoadPdfs:
    def test_loads_text_from_pdf(self, tmp_path):
        from pipeline import load_pdfs
        pdf = tmp_path / "test.pdf"
        make_fixture_pdf(pdf, ["Concept Relevance Propagation extends LRP."])

        docs = load_pdfs(tmp_path)
        assert len(docs) >= 1
        assert any("Concept Relevance Propagation" in d.text for d in docs)

    def test_each_page_is_separate_document(self, tmp_path):
        from pipeline import load_pdfs

        pdf = tmp_path / "multipage.pdf"
        make_fixture_pdf(pdf, [
            "Page one: CRP extends LRP for concept attribution.",
            "Page two: Pruning removes irrelevant convolutional filters.",
            "Page three: Heatmaps visualise concept relevance spatially.",
        ])

        docs = load_pdfs(tmp_path)
        assert len(docs) == 3

    def test_metadata_contains_filename_and_page(self, tmp_path):
        from pipeline import load_pdfs

        pdf = tmp_path / "report.pdf"
        make_fixture_pdf(pdf, ["Some content about mitosis detection."])

        docs = load_pdfs(tmp_path)
        assert docs[0].metadata["file_name"] == "report.pdf"
        assert "page_label" in docs[0].metadata

    def test_blank_pages_are_skipped(self, tmp_path):
        from pipeline import load_pdfs

        pdf = tmp_path / "withblank.pdf"
        doc = fitz.open()
        # Page 1: content
        p1 = doc.new_page()
        p1.insert_text((50, 50), "This page has real content about CRP.", fontsize=12)
        # Page 2: blank
        doc.new_page()
        doc.save(str(pdf))
        doc.close()

        docs = load_pdfs(tmp_path)
        assert len(docs) == 1  # blank skipped

    def test_multiple_pdfs_are_all_loaded(self, tmp_path):
        from pipeline import load_pdfs

        make_fixture_pdf(tmp_path / "paper_a.pdf", ["Content from paper A about CRP."])
        make_fixture_pdf(tmp_path / "paper_b.pdf", ["Content from paper B about LRP."])

        docs = load_pdfs(tmp_path)
        sources = {d.metadata["file_name"] for d in docs}
        assert "paper_a.pdf" in sources
        assert "paper_b.pdf" in sources

    def test_raises_if_no_pdfs(self, tmp_path):
        from pipeline import load_pdfs

        with pytest.raises(FileNotFoundError, match="No PDFs found"):
            load_pdfs(tmp_path)

    def test_text_is_clean_string(self, tmp_path):
        from pipeline import load_pdfs

        pdf = tmp_path / "clean.pdf"
        make_fixture_pdf(pdf, ["Clean readable text without binary garbage."])
        docs = load_pdfs(tmp_path)

        for doc in docs:
            assert isinstance(doc.text, str)
            # No null bytes
            assert "\x00" not in doc.text

# Chunking 

class TestChunking:

    def test_chunks_respect_size_limit(self, tmp_path):
        from llama_index.core.node_parser import SentenceSplitter
        from pipeline import load_pdfs

        pages = [
            f"Sentence {i}: Concept Relevance Propagation extends Layer-wise Relevance Propagation."
            for i in range(10)
        ]
        pdf = tmp_path / "long.pdf"
        make_fixture_pdf(pdf, pages)

        docs = load_pdfs(tmp_path)
        # Use a very small chunk_size to guarantee splitting across docs
        splitter = SentenceSplitter(chunk_size=20, chunk_overlap=5)
        nodes = splitter.get_nodes_from_documents(docs)

        assert len(nodes) > len(docs)  # more chunks than pages means splitting occurred

    def test_chunks_are_non_empty(self, tmp_path):
        from llama_index.core.node_parser import SentenceSplitter
        from pipeline import load_pdfs, CHUNK_SIZE, CHUNK_OVERLAP

        pdf = tmp_path / "content.pdf"
        make_fixture_pdf(pdf, ["CRP attributes decisions to individual concepts layer by layer."])

        docs = load_pdfs(tmp_path)
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(docs)

        for node in nodes:
            assert node.text.strip() != ""

    def test_chunk_metadata_propagates(self, tmp_path):
        from llama_index.core.node_parser import SentenceSplitter
        from pipeline import load_pdfs, CHUNK_SIZE, CHUNK_OVERLAP

        pdf = tmp_path / "meta.pdf"
        make_fixture_pdf(pdf, ["Text about mitosis detection and CRP attribution."])

        docs = load_pdfs(tmp_path)
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(docs)

        for node in nodes:
            assert "file_name" in node.metadata
            assert node.metadata["file_name"] == "meta.pdf"
