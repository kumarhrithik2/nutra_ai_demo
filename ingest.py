"""Load company documents into the persistent knowledge base.  Run:  python ingest.py"""
from config import get_settings
from knowledge import KnowledgeBase


def main() -> None:
    s = get_settings()
    kb = KnowledgeBase(s.db_path)
    report = kb.ingest_folder(s.knowledge_dir)
    for name, count in report["added"].items():
        print(f"  added/updated  {name}  ({count} chunks)")
    for name in report["unchanged"]:
        print(f"  unchanged      {name}")
    for name in report["removed"]:
        print(f"  removed        {name}")
    for name, err in report["errors"].items():
        print(f"  ERROR          {name}: {err}")
    print(f"Knowledge base now holds {kb.stats()} in {s.db_path}")


if __name__ == "__main__":
    main()