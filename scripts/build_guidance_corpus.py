from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from insurance_copilot.retrieval.chunking import (
    chunk_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "docs" / "raw"
PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "docs"
    / "processed"
)

CHUNKS_PATH = (
    PROCESSED_DIR
    / "guidance_chunks.jsonl"
)

MANIFEST_PATH = (
    PROCESSED_DIR
    / "guidance_manifest.json"
)


GUIDANCE_SOURCES = [
    {
        "document_id": "TDI-AUTO-GUIDE",
        "slug": "auto_insurance_guide",
        "title": "TDI Auto Insurance Guide",
        "url": (
            "https://www.tdi.texas.gov/"
            "pubs/consumer/cb020.html"
        ),
    },
    {
        "document_id": "TDI-AUTO-FAQ",
        "slug": "auto_insurance_faq",
        "title": "TDI Auto Insurance FAQ",
        "url": (
            "https://www.tdi.texas.gov/"
            "consumer/auto-insurance-faq.html"
        ),
    },
    {
        "document_id": "TDI-COMPLAINT-HELP",
        "slug": "insurance_complaint_help",
        "title": "TDI Insurance Complaint Help",
        "url": (
            "https://www.tdi.texas.gov/"
            "consumer/"
            "get-help-with-an-insurance-complaint.html"
        ),
    },
]


def ensure_directories() -> None:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def download_html(
    url: str,
) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "InsuranceSupportInvestigationCopilot/"
                "0.1 educational-project"
            )
        },
    )

    with urlopen(
        request,
        timeout=30,
    ) as response:
        return response.read()


def sha256_bytes(
    content: bytes,
) -> str:
    return hashlib.sha256(
        content
    ).hexdigest()


def clean_text(
    text: str,
) -> str:
    return " ".join(
        text.split()
    ).strip()


def extract_sections(
    html: bytes,
    fallback_title: str,
) -> list[tuple[str, str]]:
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
        ]
    ):
        tag.decompose()

    content = (
        soup.find("main")
        or soup.find(
            id="main"
        )
        or soup.body
    )

    if content is None:
        return []

    sections: list[
        tuple[str, str]
    ] = []

    current_heading = (
        fallback_title
    )

    current_parts: list[str] = []

    def flush_section() -> None:
        nonlocal current_parts

        text = clean_text(
            " ".join(
                current_parts
            )
        )

        if text:
            sections.append(
                (
                    current_heading,
                    text,
                )
            )

        current_parts = []

    for element in content.find_all(
        [
            "h1",
            "h2",
            "h3",
            "p",
            "li",
        ]
    ):
        text = clean_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not text:
            continue

        if element.name in {
            "h1",
            "h2",
            "h3",
        }:
            flush_section()

            current_heading = text
            continue

        if (
            element.name == "li"
            and element.find(
                "p",
                recursive=False,
            )
        ):
            continue

        current_parts.append(text)

    flush_section()

    return sections


def build_corpus() -> None:
    ensure_directories()

    retrieved_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    chunks: list[dict] = []
    manifest_documents = []

    for source in GUIDANCE_SOURCES:
        print(
            f"Downloading: "
            f"{source['title']}"
        )

        html = download_html(
            source["url"]
        )

        raw_path = (
            RAW_DIR
            / f"{source['slug']}.html"
        )

        raw_path.write_bytes(
            html
        )

        sections = extract_sections(
            html=html,
            fallback_title=source[
                "title"
            ],
        )

        document_chunk_count = 0

        for (
            section_index,
            (
                section_title,
                section_text,
            ),
        ) in enumerate(
            sections,
            start=1,
        ):
            section_chunks = (
                chunk_text(
                    section_text,
                    chunk_size=180,
                    overlap=30,
                )
            )

            for (
                chunk_index,
                text,
            ) in enumerate(
                section_chunks,
                start=1,
            ):
                chunk_id = (
                    f"{source['document_id']}"
                    f"-S{section_index:03d}"
                    f"-C{chunk_index:03d}"
                )

                chunks.append(
                    {
                        "chunk_id": (
                            chunk_id
                        ),
                        "document_id": (
                            source[
                                "document_id"
                            ]
                        ),
                        "document_title": (
                            source[
                                "title"
                            ]
                        ),
                        "section_title": (
                            section_title
                        ),
                        "source_url": (
                            source["url"]
                        ),
                        "retrieved_at_utc": (
                            retrieved_at
                        ),
                        "text": text,
                    }
                )

                document_chunk_count += (
                    1
                )

        manifest_documents.append(
            {
                **source,
                "retrieved_at_utc": (
                    retrieved_at
                ),
                "sha256": (
                    sha256_bytes(
                        html
                    )
                ),
                "sections": len(
                    sections
                ),
                "chunks": (
                    document_chunk_count
                ),
            }
        )

    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "retrieved_at_utc": (
            retrieved_at
        ),
        "document_count": len(
            GUIDANCE_SOURCES
        ),
        "chunk_count": len(
            chunks
        ),
        "chunking": {
            "strategy": (
                "HTML heading-aware "
                "word chunking"
            ),
            "chunk_size_words": 180,
            "overlap_words": 30,
        },
        "documents": (
            manifest_documents
        ),
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print(
        "GUIDANCE CORPUS BUILD COMPLETE"
    )
    print("=" * 70)
    print(
        f"Documents: {len(GUIDANCE_SOURCES)}"
    )
    print(
        f"Chunks:    {len(chunks)}"
    )
    print(
        f"Corpus:    {CHUNKS_PATH}"
    )
    print(
        f"Manifest:  {MANIFEST_PATH}"
    )

    print(
        "\nChunks per document:"
    )

    for document in (
        manifest_documents
    ):
        print(
            f"  "
            f"{document['document_id']:<22}"
            f"{document['chunks']:>5}"
        )


if __name__ == "__main__":
    build_corpus()
