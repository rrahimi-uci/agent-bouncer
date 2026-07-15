# Paper B plan

This is the authoritative research plan for the near-term Paper B. It selects the composition
axis (a calibrated average of the untuned base and its SFT adapter) and supersedes the older
objective-axis and mortgage joint-stack alternatives. The manuscript is
[`compose_dont_tune.tex`](compose_dont_tune.tex).

This folder is the implementation workspace. Its current phase is **retrospective draft**, not a
separately locked prospective study.

## Development sequence

- [x] Anchor and validate the clean-v2 retrospective composition result.
- [x] Generate manuscript macros and tables from evidence instead of hand-copying values.
- [x] Create an 11pt `article` draft with retrospective/prospective sections kept separate.
- [x] Add a machine-validated prospective protocol template.
- [ ] Select and hash a genuinely uninspected prospective cohort.
- [ ] Justify and lock a represented-source noninferiority margin.
- [ ] Expand the model panel and independently measure development-cohort base competence.
- [ ] Implement the shared-adapter-aware SFT+SFT equal-inference-cost control.
- [ ] Run actual WiSE-FT rescoring.
- [ ] Train and score the matched-compute KL/replay baseline.
- [ ] Lock the Paper B source, environment, operators, estimands, statistics, and cohort.
- [ ] Run the prospective analysis once and report complete systems costs.
- [ ] Publish the claim-bearing artifacts and standalone plain-language edition.

The completion gate is executable:

```sh
make protocol-locked PROTOCOL=config/prospective_protocol.json
```

The checked-in template intentionally fails that command until the missing study decisions and
implementations are complete.
