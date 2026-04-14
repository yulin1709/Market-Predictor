import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "articles.db"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    print("labelled", cur.execute("select count(*) from labelled_articles").fetchone()[0])
    print(
        "entities_table",
        cur.execute(
            "select count(*) from sqlite_master where type='table' and name='entities'"
        ).fetchone()[0],
    )
    print("labels", cur.execute("select label, count(*) from labelled_articles group by label").fetchall())
    print("dates", cur.execute("select min(aligned_date), max(aligned_date) from labelled_articles").fetchone())
    print(
        "body_rows",
        cur.execute(
            "select count(*) from articles where body_text is not null and trim(body_text) <> ''"
        ).fetchone()[0],
    )
    conn.close()


if __name__ == "__main__":
    main()
