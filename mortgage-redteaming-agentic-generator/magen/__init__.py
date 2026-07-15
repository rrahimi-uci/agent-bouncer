"""magen — Mortgage AGENtic red-teaming benchmark generator.

An agentic pipeline that grounds realistic mortgage-guardrail evaluation prompts in the
public HMDA 2022 National Loan-Level Snapshot, dual-labels them (general-safety G +
mortgage-policy D) with a rubric-bound LLM judge, decontaminates and family-isolates them,
and packages a reproducible benchmark — following the methodology in the parent repo's
docs/mortgage-benchmark-build-spec.md and -runbook.md.

Runs fully offline (deterministic mock provider + bundled HMDA sample); swap in a real LLM
provider and the real HMDA CSV for a production build.
"""

__version__ = "0.1.0-agentic"
