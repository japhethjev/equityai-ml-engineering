from app.rag.document_loader import load_pdf


def test_load_pdf():
    pages = load_pdf("data/dangote_refinery_valuation.pdf")

    assert len(pages) > 0

    assert "document" in pages[0]
    assert "page" in pages[0]
    assert "text" in pages[0]

    assert pages[0]["document"] == "dangote_refinery_valuation.pdf"
    assert pages[0]["page"] == 1