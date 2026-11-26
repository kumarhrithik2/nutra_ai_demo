"""Interactive chat in the terminal.  Run:  python chat.py [session-name]

Commands:  /new  start a fresh conversation   /stats  token and cost totals   /quit  exit
"""
import sys

from assistant import Assistant


def main() -> None:
    session = sys.argv[1] if len(sys.argv) > 1 else "demo"
    bot = Assistant()
    totals = {"in": 0, "out": 0, "calls": 0, "usd": 0.0}
    print(f"NutraAI demo | provider={bot.s.provider} | knowledge={bot.kb.stats()} | session='{session}'")
    if bot.kb.stats()["chunks"] == 0:
        print("The knowledge base is empty. Run `python ingest.py` first.")
    print("Type a question. /new, /stats, /quit\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            bot.store.clear(session)
            print("(conversation cleared)\n")
            continue
        if text == "/stats":
            print(f"calls={totals['calls']} input_tokens={totals['in']} output_tokens={totals['out']} "
                  f"estimated_cost=${totals['usd']:.5f}\n")
            continue
        a = bot.ask(session, text)
        totals["in"] += a.input_tokens
        totals["out"] += a.output_tokens
        totals["calls"] += a.llm_calls
        totals["usd"] += bot.cost_usd(a)
        print(f"\nAssistant: {a.text}\n")
        src = ", ".join(sorted({f"{s['source']} > {s['heading']}" for s in a.sources})) or "none"
        print(f"  [status={a.status} | flags={a.flags or '-'} | sources={src}]")
        print(f"  [tokens in={a.input_tokens} out={a.output_tokens} calls={a.llm_calls} "
              f"cost~${bot.cost_usd(a):.5f}]\n")


if __name__ == "__main__":
    main()