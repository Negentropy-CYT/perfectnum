# Legacy Reference Implementations

This directory preserves earlier research implementations for reproducibility
and comparison. It is not part of the current `opn_main.py` execution path and
does not define the current mathematical or persistence contracts.

Contents:

```text
main.py              legacy entry point and configuration
core.py              prime generation and arithmetic helpers
search.py            independent-prime DFS search
io.py                legacy checkpoint helpers
opn_factor_chain.py  early factor-chain prototype
```

The legacy DFS fixes non-Euler components to exponent 2 and is primarily useful
for reproducing Descartes-type spoof searches. It should not be used as a
substitute for the current variable-exponent factor-chain engine.

Run it explicitly from the repository root:

```bash
python legacy/main.py
```

Legacy checkpoints, telemetry, and output formats are not guaranteed to be
compatible with the current engine.
