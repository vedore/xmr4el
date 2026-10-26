# Documentation

For installation, input formats, and quick-start commands, start with the
[project README](../README.md). For training, conversion, and diagnostic recipes,
see the [scripts guide](../scripts/README.md).

## Current implementation and research

| Document | Purpose |
| --- | --- |
| [Pipeline](pipeline.md) | Implemented behavior, validation, and pipeline observations. |
| [Results](results.md) | Dated experiment results and reproduction commands; older sections describe their historical configurations. |
| [Future work](future_work.md) | Open research problems and where to pick up. |
| [Working status](../STATUS.md) | Detailed development state and session notes for this checkout. |

## Interface design

| Document | Status |
| --- | --- |
| [TUI design](design/tui.md) | Interface design; increments 1-2 built (branch `tui`). |
| [TUI architecture](design/tui_architecture.md) | Module layout, data model and build order for the TUI. |

## Audits

Audits describe the code inspected on their stated dates. Findings may have been
fixed; use the pipeline document for current behavior.

- [Tool readiness audit, 2026-10-09](audits/tool_readiness_audit.md)
- [Comprehensive audit, 2026-10-16](audits/comprehensive_audit_2026-10-16.md)
- [Code quality audit, 2026-10-16](audits/code_quality_audit_2026-10-16.md)

## Archive and references

These documents preserve earlier proposals, procedures, and research history.
Their commands and architectural recommendations are not current contracts.

- [Earlier experiment-management proposal](archive/experiment_management.md)
- [Earlier full-vocabulary CUDA comparison procedure](archive/full_cuda_comparison.md)
- [Historical status log](archive/status_log.md)
- [Architecture assessment, 2026-10-23](archive/architecture.md)
- [Thesis PDF](references/thesis.pdf)

Keep current implementation references at the top level, proposed interfaces in
`design/`, dated assessments in `audits/`, historical notes in `archive/`, and
supporting publications in `references/`.
