# Point-in-time and leakage checklist

- Use `first_seen_time`, not only the publisher's timestamp.
- Never activate a thesis before `max(created_at, valid_from)`.
- Keep retrospective narratives out of historical model inputs.
- Build same-minute RVOL baselines from prior sessions only.
- Fit scaler and feature selection on the training window only.
- Keep all rows from one market date in the same split.
- Select confidence thresholds only on validation dates.
- Execute a signal on a later bar; never fill at the close that created it.
- Preserve source, adjustment, condition-code and corporate-action metadata.
- Use ALFRED vintages for revised macro series when available.
- Treat explicit-event failures as high-quality negative samples.
- Do not interpret Friday option expiry as a directional label without licensed options data.

