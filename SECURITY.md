# Security policy

This is research software intended for local execution. It has no dedicated security response team or guaranteed response time. The supported dependency environment is the pinned Python 3.12 CPU installation documented in the README.

For a security-sensitive report, use GitHub's private vulnerability reporting on this repository if that option is enabled. Otherwise contact a maintainer using the author contact information published with the accompanying manuscript. Avoid publicly posting an exploitable proof of concept, credentials or private datasets before the maintainers can assess the issue.

Include the affected commit, platform, relevant dependency versions and a minimal reproduction that does not expose secrets. Ordinary numerical discrepancies and reproducibility failures belong in public issues; use the reproduction report template and include sanitized run metadata.

Run only trusted source and result files. The local launcher can install project dependencies and execute experiments on the host computer; it is not designed as a public network service. Preserve its local-only deployment. Do not commit virtual environments, access tokens, private data or unsanitized machine-specific logs.
