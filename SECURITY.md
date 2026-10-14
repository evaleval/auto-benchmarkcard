# Security

Auto-BenchmarkCard is research code, not a hardened public service. The
pipeline follows candidate URLs from its inputs and fetches papers,
repository pages, dataset pages and other web sources. It does not provide a
complete blocklist for localhost, private networks, cloud metadata endpoints
or other sensitive destinations.

Do not expose the CLI or the webhook to untrusted inputs without adding URL
validation, network egress controls, authentication, request limits and
logging appropriate to the deployment. Run jobs in an isolated environment
with least-privilege credentials. Never place credentials in committed
files; use environment variables or a secret manager.

Report suspected vulnerabilities through GitHub's private vulnerability
reporting for this repository rather than in a public issue, and do not
include live credentials or sensitive source material in the report.
