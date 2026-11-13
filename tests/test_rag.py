import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag import (
    DocumentChunk,
    DocumentPage,
    answer_question,
    chunk_pages,
    create_qdrant_client,
    create_session_documents,
    create_session_retriever,
    delete_session_documents,
    load_document_pages,
    source_label,
    source_text,
    split_text,
    summarize_document,
)


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    @staticmethod
    def _embed(text):
        return [
            float(text.lower().count("alpha") + 1),
            float(text.lower().count("beta") + 1),
            float(len(text) + 1),
        ]


class FakeLLM:
    def __init__(self, response="test answer"):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


class DocumentLoadingTests(unittest.TestCase):
    def test_load_text_file_strips_bom_and_keeps_filename(self):
        pages = load_document_pages("notes.txt", b"\xef\xbb\xbfUseful text")

        self.assertEqual(pages, [DocumentPage("Useful text", "notes.txt", 1)])

    def test_load_text_file_rejects_empty_content(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            load_document_pages("empty.txt", b" \n")

    @patch("pypdf.PdfReader")
    def test_pdf_pages_keep_one_based_page_numbers(self, pdf_reader):
        pdf_reader.return_value = SimpleNamespace(
            is_encrypted=False,
            pages=[
                SimpleNamespace(extract_text=lambda: "First page"),
                SimpleNamespace(extract_text=lambda: "Second page"),
            ],
        )

        pages = load_document_pages("guide.pdf", b"pdf bytes")

        self.assertEqual(
            pages,
            [
                DocumentPage("First page", "guide.pdf", 1),
                DocumentPage("Second page", "guide.pdf", 2),
            ],
        )

    def test_scanned_pdf_without_text_is_rejected(self):
        with patch("pypdf.PdfReader") as pdf_reader:
            pdf_reader.return_value = SimpleNamespace(
                is_encrypted=False,
                pages=[SimpleNamespace(extract_text=lambda: "")],
            )
            with self.assertRaisesRegex(ValueError, "Scanned PDFs"):
                load_document_pages("scan.pdf", b"pdf bytes")

    def test_chunking_preserves_page_metadata_and_overlap(self):
        page = DocumentPage("abcdefghijklmnopqrstuv", "guide.pdf", 4)
        chunks = chunk_pages([page], chunk_size=12, overlap=3)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.source == "guide.pdf" for chunk in chunks))
        self.assertTrue(all(chunk.page == 4 for chunk in chunks))
        self.assertIn("jkl", chunks[0].text)
        self.assertIn("jkl", chunks[1].text)

    def test_invalid_chunk_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            split_text("some text", chunk_size=10, overlap=10)


class QdrantIsolationTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name="test_docs",
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        )
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name="test_docs",
            embedding=TinyEmbeddings(),
        )
        self.store.add_documents(
            create_session_documents(
                [
                    DocumentChunk("alpha private document", "a.pdf", 1),
                    DocumentChunk("beta another document", "b.pdf", 2),
                ],
                "session-a",
                "index-a",
            )
            + create_session_documents(
                [DocumentChunk("alpha different user", "other.pdf", 1)],
                "session-b",
                "index-b",
            )
        )

    def tearDown(self):
        self.client.close()

    def test_memory_mode_ignores_stale_cloud_credentials(self):
        client = create_qdrant_client(
            "memory",
            url="https://unreachable.example",
            api_key="stale-key",
        )
        try:
            self.assertFalse(client.collection_exists("pdf_docs"))
        finally:
            client.close()

    def test_cloud_mode_requires_a_url(self):
        with self.assertRaisesRegex(ValueError, "QDRANT_URL is required"):
            create_qdrant_client("cloud")

    def test_retrieval_is_filtered_to_one_session_and_index(self):
        retriever = create_session_retriever(self.store, "session-a", "index-a")

        results = retriever.invoke("alpha")

        self.assertTrue(results)
        self.assertTrue(
            all(document.metadata["session_id"] == "session-a" for document in results)
        )
        self.assertTrue(
            all(document.metadata["index_id"] == "index-a" for document in results)
        )

    def test_delete_only_removes_the_requested_session_index(self):
        delete_session_documents(
            self.client,
            "test_docs",
            "session-a",
            "index-a",
        )

        results = create_session_retriever(
            self.store, "session-b", "index-b"
        ).invoke("alpha")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["session_id"], "session-b")


class GenerationTests(unittest.TestCase):
    def test_answer_prompt_contains_history_and_source_passages(self):
        source = Document(
            page_content="The result is 42.",
            metadata={"source": "guide.pdf", "page": 3},
        )
        retriever = SimpleNamespace(invoke=lambda query: [source])
        llm = FakeLLM()

        answer, sources = answer_question(
            "What is the result?",
            retriever,
            llm,
            [{"role": "user", "content": "Previous question"}],
        )

        self.assertEqual(answer, "test answer\n\nSources: guide.pdf (page 3)")
        self.assertEqual(sources, [source])
        self.assertIn("Previous question", llm.prompts[0])
        self.assertIn("Passage: The result is 42.", llm.prompts[0])
        self.assertNotIn("guide.pdf", llm.prompts[0])
        self.assertIn("Answer the question using the passage.", llm.prompts[0])
        self.assertEqual(source_label(source), "guide.pdf (page 3)")
        self.assertEqual(source_text(source), "The result is 42.")

    def test_answer_cites_each_retrieved_page_once(self):
        sources = [
            Document(
                page_content="First passage.",
                metadata={"source": "guide.pdf", "page": 3},
            ),
            Document(
                page_content="Another passage on the same page.",
                metadata={"source": "guide.pdf", "page": 3},
            ),
            Document(
                page_content="A second cited page.",
                metadata={"source": "guide.pdf", "page": 4},
            ),
        ]
        retriever = SimpleNamespace(invoke=lambda query: sources)
        llm = FakeLLM()

        answer, _ = answer_question("What happened?", retriever, llm)

        self.assertEqual(
            answer,
            "test answer\n\nSources: guide.pdf (page 3), guide.pdf (page 4)",
        )

    def test_summary_covers_all_chunks_and_reduces_large_documents(self):
        chunks = [
            DocumentChunk("A" * 1700, "a.pdf", 1),
            DocumentChunk("B" * 1700, "b.pdf", 2),
        ]
        llm = FakeLLM(response="summary")

        summary, sources = summarize_document(chunks, llm)

        self.assertEqual(summary, "summary")
        self.assertEqual(sources, chunks)
        self.assertGreaterEqual(len(llm.prompts), 3)
        self.assertTrue(any(prompt.startswith("summarize: ") for prompt in llm.prompts))
        self.assertTrue(any("all key facts" in prompt for prompt in llm.prompts))
        self.assertFalse(any("a.pdf (page" in prompt for prompt in llm.prompts))
        self.assertTrue(any("A" * 100 in prompt for prompt in llm.prompts))
        self.assertTrue(any("B" * 100 in prompt for prompt in llm.prompts))


if __name__ == "__main__":
    unittest.main()
