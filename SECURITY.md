# Security

Auto-BenchmarkCards is research code, not a hardened public service. The
pipeline follows candidate URLs from input records and fetches papers,
repository pages, dataset pages, and other web sources. It does not currently
provide a complete blocklist for localhost, private networks, cloud metadata
endpoints, or other sensitive destinations.

Do not expose the CLI or webhook to untrusted inputs without adding URL
validation, network egress controls, authentication, request limits, and
logging appropriate to the deployment. Run research jobs in an isolated
environment with least-privilege credentials. Never place credentials in
committed files; use environment variables or an approved secret manager.

Report suspected vulnerabilities privately to `aris.hofmann@ibm.com`. Do not
include live credentials, participant data, or sensitive source material in a
public issue.
