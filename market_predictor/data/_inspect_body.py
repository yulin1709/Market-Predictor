import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).with_name("articles.db")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        SELECT id, headline, published_at, LENGTH(COALESCE(body_text, '')) AS body_len, body_text
        FROM articles
        WHERE id GLOB '????????-????-????-????-????????????'
          AND LENGTH(COALESCE(body_text, '')) > 5000
        ORDER BY body_len DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        print("No long-body UUID article found.")
        return

    article_id, headline, published_at, body_len, body_text = row
    print(f"id={article_id}")
    print(f"headline={headline}")
    print(f"published_at={published_at}")
    print(f"body_len={body_len}")
    print("--- BODY START ---")
    sample = (body_text or "")[:6000]
    print(sample.encode("cp1252", errors="replace").decode("cp1252"))
    print("--- BODY END ---")


if __name__ == "__main__":
    main()
