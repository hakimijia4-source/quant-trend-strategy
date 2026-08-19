# Architecture

```text
Authorized SIP bars ─┐
Context assets ──────┼─> point-in-time features ─> labeled market states
SEC / IR / macro ────┤                              │
Thesis ledger ───────┘                              v
                                             offline trajectories
                                                   │
                         ┌─────────────────────────┴──────────────────────┐
                         v                                                v
             Hierarchical Decision Transformer                Exogenous World Model
             option / action / termination / value            state / reward / done
                         └─────────────────────────┬──────────────────────┘
                                                   v
                                      world-model-reranked signal
                                                   v
                                     walk-forward research report
```

The project intentionally ends at the research-signal boundary. A future execution service
must independently implement account state, position limits, order validation, broker
connectivity, reconciliation, kill switches, and incident logging.

