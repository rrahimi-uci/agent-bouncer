# Security Policy

Agent Bouncer is a defensive safety tool, so we take security seriously.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, email
**145064767+rrahimi-uci@users.noreply.github.com** (or use GitHub's private "Report a vulnerability"
feature). Include steps to reproduce and the potential impact. We aim to
acknowledge reports within 72 hours.

## Scope

Relevant reports include, for example:
- Ways to bypass the guard that generalize across inputs (evasion attacks).
- Prompt-injection / jailbreak patterns the guard should catch but doesn't.
- Vulnerabilities in the serving layer or training pipeline.

Model false negatives on individual crafted inputs are expected and best filed as
regular issues (or added as evaluation cases).
