import sqlite3
from pathlib import Path

from collect_prices import extract_prices_from_text, extract_relevant_lines


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
    lines = extract_relevant_lines(body_text or "")
    print(f"relevant_lines={len(lines)}")
    print("--- FIRST 80 RELEVANT LINES ---")
    for line in lines[:80]:
        safe = line.encode("cp1252", errors="replace").decode("cp1252")
        print(safe)
    print("--- PARSED ROWS ---")
    rows = extract_prices_from_text(body_text or "", source_name=headline or "")
    for row in rows:
        print(row)
    if not rows:
        print("No parsed rows.")


if __name__ == "__main__":
    main()
