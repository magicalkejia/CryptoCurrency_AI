"""Run all crypto tests without pytest. Usage: python tests/run_all.py"""
import sys
import traceback

import tests.test_crypto as T


def main():
    mods = [T]
    try:
        import tests.test_crypto_phase4 as T4
        mods.append(T4)
    except Exception as e:
        print(f"(phase4 tests not loaded: {e})")
    try:
        import tests.test_agents as TA
        mods.append(TA)
    except Exception as e:
        print(f"(agent tests not loaded: {e})")
    try:
        import tests.test_phase5 as T5
        mods.append(T5)
    except Exception as e:
        print(f"(phase5 tests not loaded: {e})")
    fns = []
    for m in mods:
        fns += [getattr(m, n) for n in dir(m) if n.startswith("test_")]
    passed, failed = 0, 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
