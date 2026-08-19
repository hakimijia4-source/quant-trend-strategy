# Security and data handling

## Credentials

Keep API credentials in environment variables. Never put real values in TOML,
CSV, notebooks, source files, issue reports, or commits. The supported variable
names are documented in `README.md`; configuration files contain names only.

Before publishing changes, inspect the staged diff and run a secret scanner if
one is available. If a credential is committed, revoke it at the provider first,
then remove it from Git history.

## Trading safety

This repository has no broker order endpoint and must remain research-only.
`OfflinePolicyRuntime` emits signals with `execution_authorized=false`. Any live
execution system should be maintained separately with explicit risk limits,
position limits, kill switches, audit logs, and independent review.

## Vulnerability reports

Do not include credentials, private datasets, account identifiers, or brokerage
details in public reports. Use the repository owner's private contact channel.
