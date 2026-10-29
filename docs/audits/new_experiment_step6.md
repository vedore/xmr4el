# New Experiment: Step 6 engineering verification

Date: 2026-10-29. Starting revision: `18dfefd`, branch `tui`; working tree clean on entry.
Environment: macOS, Python 3.12.16, Textual 8.2.8. No dependency additions, model downloads, GPU use,
biomedical training/evaluation, research-data changes, or deletion of user outputs.

## A. Assessment

Recommendation: **Ready with limitations**. The supported
hybrid New Experiment workflow has been exercised through actual Textual keyboard/mouse events and real
detached synthetic subprocesses. Confirmed defects were repaired in the existing architecture. The ML
algorithm, metric definitions, evaluation protocol and comparability authority were not changed.

Final regression result: **455 passed, 0 failed, 0 skipped** in the full suite (314.24 s), plus **4 passed**
for the final configuration-shape and dataset-cache boundary checks added while that run was in progress.
The configuration/UI/backend cross-check subset also passed **93 tests**. Ruff and `git diff --check` pass.

This is engineering verification, not evidence of scientific model quality or full-corpus resource capacity.
Linux execution and a real terminal/SSH session remain manual checks. Results semantics have independent
fixture tests; synthetic coordinator tests intentionally produce no numerical evaluation metrics.

## B. Verification matrix

PASS means the stated behavior was exercised, not that every conceivable failure is excluded. PARTIAL marks
an explicit boundary. Test paths below are under `tests/`.

| Subsystem | Status | Evidence and boundary |
|---|---|---|
| Dataset Setup | PASS | `tui/test_dataset_setup.py`: spec/directory discovery, variants, registration, missing files, duplicate splits, train/eval aliases, `.txt` PubTator, bounded sniff, picker cancellation and typed paths, confirmation isolation. Added malformed-document and invalid-spec-type cases. E2E uses a root containing spaces. Full-corpus parsing remains the runner's job. |
| Model Builder | PASS | `tui/test_model_builder.py`, `test_tui_components.py`: eight supported hybrid slots, fixed PIFA, registry/backend defaults and config round trips, three branches, 45/55 and 40/60 geometry, scrolling, resize, all zoom levels and unchanged model JSON. |
| Tree Selection | PASS | Numbered tabs, arrows/Enter, Add stop, removal confirmation and last-tree guard, duplication isolation, ordering, editor→library→grid→tabs→parent hierarchy. Current code has no Tree Actions popup or Ctrl+Tab/F6 panel cycle; those older proposals are NOT APPLICABLE. |
| Component editing | PASS | Draft/Apply/Cancel, invalid values, full read-only proposed JSON, multi-key highlighting, caret/dropdown navigation, help coverage. Added actual editor test for `max_df=1.0` versus `1` dirty protection. |
| Review Batch | PASS | `tui/test_review_batch.py`: stable oldest-created reference, recursive defaults/differences, batch naming, queue reorder/cancel, selected identity, shared validation once, cursor changes avoid revalidation. New tests distinguish count/fraction DF settings. |
| Fix Issue | PASS | Existing dataset/train/eval/name and fallback tests; `tui/test_workflow_e2e.py` adds field focus, JSON path, unapplied isolation, Apply and return to Review for transformer, both SVDs, hierarchy, matcher and search. |
| Launch | PASS | Actual click, Tab/Enter and L launch the coordinator; repeated L starts one batch; invalid configurations block. Fourteen trees at 80×24, 100×30 and 140×40 retain visible, keyboard-reachable Launch. Cautions remain nonblocking. |
| Coordinator | PASS | `tui/test_batch.py`, `tui/test_batch_audit.py`: real subprocess order/non-overlap, one/three trees, identical configs retained, first/middle/last failures continue, startup failures, no-output exits, malformed outputs, concurrent launchers, hash failures, atomic status writes, immutable plan/config copies and restart refusal. |
| Cancellation | PASS | UI Stop confirmation; stop before first/during/between runs; stop arriving during input hashing; repeated requests; SIGTERM-ignoring runner; worker surviving its leader; launch-record and status-write failures; unrelated-process protection in `tui/test_process.py`. |
| Detachment | PASS | `test_workflow_e2e.py::test_dashboard_to_batch_and_reconnect`: Leave Running exits one app and remaining subprocesses finish while a fresh app reconnects. `test_batch_audit.py::test_coordinator_outlives_the_launcher_process_and_its_pipes` additionally verifies actual launcher-process exit and pipe closure while three trees continue. Coordinator uses its own session, DEVNULL stdin, log-file stdout/stderr, same interpreter and explicit project cwd. Actual SSH/terminal hangup is NOT TESTED. |
| Reconnection | PASS | Fresh app discovers same batch and run IDs without a new launch; completion/failure remain discoverable. Coordinator SIGKILL test yields LOST/INTERRUPTED, releases both locks, and never restarts pending work. |
| Batch Monitor | PASS | Actual persisted state, counts, tree/run associations, completed/failed batches, repeated reopen, direct Esc→Dashboard, draft resume and existing log/telemetry panels. Repaired pre-mount log-update crash; dataset name now shown. |
| Dashboard | PASS | Existing Explorer/active-run tests plus batch list, active/completed batches, monitor reopen and stack tests. Three monitor/resume cycles preserve one Dashboard and unapplied drafts. |
| Results | PARTIAL | `tui/test_results.py`, `tui/test_runs.py`: independent valid fixtures cover baseline/challengers, hashes/protocols, missing evidence, resources and incompatibility. Batch-generated paths are ordinary run directories. No real model was trained to produce numerical Results through the new UI. |
| Scientific correctness | PASS (synthetic invariants) | `integration/test_label_mapping.py`, hierarchy/feature/learning tests, `test_eval.py`, `test_run_experiment.py`, `test_tui_components.py`. Source inspection confirms sorted classes, loader ordering guard, leaf label matching, reassignment, matcher routing scores and path multiplication. No corpus-quality claim. |
| Footer / notifications / quit | PASS | `tui/test_footer.py`, `test_notifications.py`, `test_batch_ui.py`: full-width feedback, color/priority, stale expiry isolation, transition deduplication, no replay at first scan, three quit choices with Stay focused, ordinary quit, safe Esc. A flaky 0.3-second navigation assumption was replaced with direct stale-callback verification plus a real expiry check. |

## C. Findings and repairs

Every entry below is fixed. None required replacing the training engine or adding recovery infrastructure.
Regression names refer to `tui/test_batch_audit.py` unless another file is named.

| ID / severity | Location | Reproduction, root cause and consequence | Evidence |
|---|---|---|---|
| F1 / P1 | `coordinator.py:Coordinator.wait/signal_group` | Runner exits on TERM while a forked worker ignores it. Poll reaped the leader before group cleanup; macOS also cannot query the exited leader's pgid. Worker survived Stop Batch. Keep the owned leader unreaped until group cleanup, then wait. | `test_stop_cleans_workers_even_when_the_runner_exits_first`; existing forced-stop test |
| F2 / P1 | `process.py:spawn` | Make the launch-record destination unwritable after Popen. Exception returned without a child handle and leaked the process. Failed handoff now kills/reaps its owned child and closes stdin before re-raising. | `test_launch_record_failure_reaps_the_spawned_child` |
| F3 / P1 | `coordinator.py:Coordinator.save/abort` | Fail both status writes while a runner is active. `SystemExit` bypassed abort, releasing locks with the runner alive. Persistence errors now enter cleanup before batch failure is recorded. | `test_status_write_failure_stops_owned_child_before_releasing_lock` |
| F4 / P1 | `coordinator.py:Coordinator.run` | Stop arrives during potentially long input hashing. The loop had checked Stop only before hashing and still called spawn. It now rechecks before starting the tree. | `test_stop_arriving_during_hashing_prevents_the_next_spawn` asserts zero spawn calls |
| F5 / P1 | `coordinator.py:main` | Invoke a completed plan's coordinator again after the timestamp changes. It reran the same approved trees and replaced their batch associations. Only a fresh `preparing` status can execute; interrupted/final batches are not resumed. | `test_completed_batch_cannot_be_executed_again`; unchanged status and order log |
| F6 / P2 | `batch.py:start` | Delay the launcher's first read until a fast runner has completed. Terminal-state check incorrectly raised “did not start: completed.” A recorded start by this child now acknowledges launch even if it has finished. | `test_fast_completed_batch_is_a_successful_launch` |
| F7 / P1 | `batch.py:read_status/info/scan/launch` | Persist `trees=[null]`, missing keys/unknown states, or replace a lock file with a directory. Scan/monitor/launch exceptions could crash the TUI. Validate persisted shapes and report unreadable state or actionable BatchError. The monitor displays the problem. | `test_malformed_batch_status_is_unreadable_not_a_scan_crash` (four cases), both `test_unusable_*` tests |
| F8 / P1 | `batch.py:start` | Inherit `XMR4EL_TEST_RUNNER` in a normal launch. An unintended script replaced the production runner. Normal launch now removes that environment key; explicit internal test injection still works. | `test_production_launch_does_not_inherit_test_runner` |
| F9 / P1 | `coordinator.py:Coordinator.classify` | Runner exits zero with only `status=completed`. Missing outputs were accepted as success. Completion now requires readable spec/config/provenance/tree/evaluation JSON and the referenced saved-tree file. | `test_completed_status_without_outputs_is_not_success`, `test_no_artifacts_is_not_success` |
| F10 / P1 | `coordinator.py:Coordinator.classify` | Input changes between coordinator hashing and the runner's own initial hash. Runner-local consistency alone could accept a different dataset than the approved snapshot. Completed runner provenance must now match the batch hashes; mismatch blocks the remaining batch while preserving artifacts. | `test_change_between_coordinator_check_and_runner_read_blocks_batch`; existing between-tree mutation test |
| F11 / P1 | `coordinator.py:Coordinator.console_tail` | Failed runner emits invalid UTF-8. Whole-file text decode raised and aborted later trees. Reuse bounded, replacement-decoding `LogTail`; ordinary tree failure remains local. | `test_non_utf8_runner_output_does_not_abort_the_queue`; existing `tui/test_logs.py` bounded-tail checks |
| F12 / P1 | `widgets/log_panel.py:set_run`, `screens/batch_monitor.py:fill_detail` | Repository scan updates a newly attached LogPanel before composition; overlapping removal/mount could also remove its children before Mount ran. Both raised `NoMatches`, including during monitor Esc. Defer pre-mount updates and await log-panel removal/mount in the screen's message handlers. | Initial batch-UI failures; `test_monitor.py::test_log_update_before_mount_uses_latest_run`; full E2E exposed the second race; `test_batch_ui.py::test_monitor_replaces_logs_during_selection_and_scans` |
| F13 / P2 | `components.py:same`, `model_builder.py:dirty`, `review.py:diff` | Edit TF-IDF `max_df` from `1.0` to `1`. Python numeric equality concealed a semantic change (fraction versus count): no dirty warning and no Review difference. Draft equality now preserves nested types; Review specifically preserves DF types while retaining ordinary numeric equivalence elsewhere. | `test_model_builder.py::test_document_frequency_type_change_requires_apply_or_discard`; four `test_review_batch.py` DF cases |
| F14 / P2 | `datasets.py:sniff/discover` | Invalid document IDs/offsets or annotation-only-without-label files passed Ready; a numeric spec path crashed discovery. Bounded sniff now checks sampled document association/ranges and usable annotations; discovery ignores non-string suggestions. | Five malformed `.txt` cases and `test_discovery_ignores_invalid_path_types` |
| F15 / P2–P3 | `components.py` hierarchy help; `batch.py:prepare`; `batch_monitor.py:overview_text` | Minimum-leaf help implied automatic early leaf creation, although an unsplittable internal node fails at fixed depth. Corrected description. Monitor omitted the dataset; added its confirmed name/variant to persisted status and overview. | Backend source cross-check, metadata suite, snapshot assertion and E2E overview assertion |
| F16 / P2 | `components.py:installed/slot_problems` | Import a supported component whose `kwargs` is null, a list or a string. Default lookup hid the malformed container and Review reported no errors. Reject its shape with an explicit JSON-object error; never silently replace it. | `test_review_batch.py::test_malformed_component_kwargs_block_preflight`, all seven component slots |
| F17 / P2 | `datasets.py:inspect` | Inspect a valid file, then remove read permission or replace it with invalid same-size content preserving mtime. Cache keyed only by mtime/size returned stale Ready. Include device, inode and ctime in the key. | `test_dataset_setup.py::test_replaced_input_with_same_size_and_mtime_is_rechecked`; temporary-file chmod reproduction failed before, passed after repair |

No P0 defect was demonstrated. This does not certify arbitrary filesystem corruption or model quality.

## D. Changed files

- Execution and persistence: `xmr4el_tui/{batch,coordinator,process}.py`.
- Dataset and scientific-configuration UI: `xmr4el_tui/{datasets,components,review}.py`,
  `screens/model_builder.py`, `screens/batch_monitor.py`, `widgets/log_panel.py`.
- New tests: `tests/tui/test_batch_audit.py`, `tests/tui/test_workflow_e2e.py`.
- Extended tests: batch fixtures/UI, dataset setup, builder, review, monitor and notifications. Fake runner
  outputs are structural fixtures in pytest temporary directories: empty metric objects and an explicitly
  non-loadable synthetic tree marker, never research results. UI tests clean up their synthetic batches.
- Documentation: this report, `STATUS.md`, README TUI write behavior and architecture introduction/state ownership.

Verified persistence layout:

```text
outputs/batches/.active.lock
outputs/batches/<batch_id>/
  batch.json                 approved dataset, hashes, provenance, ordered tree IDs/names/specs
  configs/<tree_id>.json      complete applied configuration; hash recorded in the plan
  status.json                atomic mutable lifecycle and unambiguous run associations
  coordinator.lock           kernel-held liveness; not a PID-only ownership claim
  coordinator.log
  stop                       idempotent cancellation request, created when requested
outputs/runs/<run_id>/        ordinary runner artifacts, discovered by existing Results/Monitor code
```

Locks are non-inheritable; runner children hold neither coordinator lock. No heartbeat, database, recovery daemon
or dataset copying was added. Process signals target owned, unreaped runner groups; external Run Monitor stop
still uses `procinfo` runtime/start/UID/group identity and the launch record.

## E. Test execution

Primary final command (offline, no cache writes):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -p no:cacheprovider tests/ -q --tb=short
uvx --offline ruff check xmr4el xmr4el_tui scripts tests
git diff --check
```

The process tests require ordinary macOS process inspection. The restricted tool sandbox denies `ps`, so
those checks were rerun with approved access; no software installation was needed.

| Run | Actual outcome |
|---|---|
| Initial full suite in sandbox | 393 passed, 16 failed, 1 warning. Eleven failures required real process identity inspection; four exposed the log-panel mount race; one exposed a timing-sensitive footer test. |
| Initial new process reproductions | 6 passed, 8 failed; failures demonstrated fast-start reporting, replay, worker cleanup, malformed status, environment injection and handoff leaks. |
| Dataset reproductions | 20 passed, 6 failed before repairs. |
| DF type reproductions | 5 failed before repairs. |
| Focused UI regression run | 89 passed after mount/dataset/footer fixes. |
| Builder + Review + integrated workflow run | 81 passed after DF fixes (before the stricter runner output contract). |
| Process regression run with output contract | 52 passed at that checkpoint. Later binary-output and lock-error regressions were added. |
| Intermediate full suite | 445 passed, 2 failed: synthetic E2E fixtures still in the running pytest process used the previous output contract while newly spawned coordinators used the repair. Final suite was restarted against one consistent code/fixture version. |
| Subsequent full suite | 451 passed, 1 failed: E2E exposed overlapping log-panel removal/mount. Repaired and covered with rapid selection/rescan regression. |
| Focused final execution integration | 45 passed after serializing log-panel mount/removal. |
| Final full suite | **455 passed, 0 failed, 0 skipped**, 1 warning, 314.24 s. Includes concurrent status readers, actual launcher-process exit, and all synthetic scenarios. |
| Final configuration subset | **93 passed**, 1 warning; builder, Review and backend registry cross-checks. |
| Last boundary regressions | **4 passed**, 66 deselected: malformed kwargs (three types, all component slots) and same-stat file replacement. These four were added after full-suite collection; 459 cases now exist, 50 more than at entry. |
| Static analysis | `uvx --offline ruff check xmr4el xmr4el_tui scripts tests` and `git diff --check`: **PASS**. |

The warning is scikit-learn reporting that `stop_words` is ignored for a non-word analyzer in an existing
parameter-enumeration test. No test was skipped or weakened to accept a scientific error.

Focused commands also executed (all pytest commands used `-p no:cacheprovider -q --tb=short`):

```sh
.venv/bin/python -m pytest tests/tui/test_batch.py tests/tui/test_batch_audit.py tests/tui/test_process.py tests/tui/test_procinfo.py
.venv/bin/python -m pytest tests/tui/test_batch_ui.py tests/tui/test_monitor.py tests/tui/test_dataset_setup.py tests/tui/test_footer.py tests/tui/test_notifications.py
.venv/bin/python -m pytest tests/tui/test_workflow_e2e.py
.venv/bin/python -m pytest tests/tui/test_review_batch.py tests/tui/test_model_builder.py tests/test_tui_components.py
.venv/bin/python -m pytest tests/tui/test_batch_ui.py tests/tui/test_workflow_e2e.py tests/tui/test_batch_audit.py
```

Focused reproduction selectors included `-k document_frequency`, `-k completed_status_without`,
`-k non_utf8`, `-k unusable` and `-k malformed_component_kwargs`. A standalone temporary-file assert check
also verified paths with spaces, symlink train/eval leakage rejection and unreadable-file reporting.

Synthetic scenarios:

- A: three trees, applied matcher/search edits, Overview/Differences, reorder, friendly rename, mouse launch,
  monitor, Esc to Dashboard, reopen, three unique run associations.
- B: middle runner fails; both surrounding trees succeed; finished-with-failures counts survive reconnection.
- C: actual monitor Stop plus backend active/between-tree/forced-stop tests; no following launch, worker cleanup.
- D: Leave Running closes the first app; a fresh app discovers the same active batch; no duplicate runner.
- E: invalid parameters block Launch; Fix Issue focuses the exact field and Apply clears the error.

Results beyond missing-metric handling are verified with the existing independent Results fixtures, not with
made-up numerical results from the fake runner. No synthetic outcome was added to `docs/results.md`.

## F. Remaining limits and manual verification

1. **Real execution environment — NOT TESTED.** No biomedical training, HF download, GPU path, SSH hangup,
   or Linux kernel process behavior was exercised. Run the same offline pytest command on the Linux research
   environment before relying on platform-specific process control there. A real training smoke run remains
   the user's separate action; it is not implied by this report.
2. **Dataset preflight — intentionally bounded.** Only the first 64 KiB are sniffed. Invalid later documents,
   vocabulary sizes, SVD rank and hierarchy feasibility can still fail inside the runner. Input hashing is
   streaming but reads entire files at launch and before each tree, off the UI thread. Directory discovery and
   cached small-file checks still have synchronous UI paths; huge directory/NFS latency was not measured.
3. **Crash policy — explicit V1 limitation.** SIGKILL of the coordinator can leave its runner alive; the batch
   becomes LOST and work is never resumed or blindly terminated. Inspect the surviving run through its verified
   Run Monitor identity before starting another resource-heavy batch. Locks guarantee one coordinator, not
   automatic cleanup after an uncatchable kill or protection against shell-launched training.
4. **Result completeness.** Coordinator checks output structure/presence and launch hashes; it does not unpickle
   models or recompute metrics. Real runner tests own scientific output generation. Review-generated evaluations
   currently omit optional eval `train_path`, so seen/unseen dictionary statistics and `strings_sha256` are absent;
   Results conservatively withholds those comparisons. No compatibility shortcut or metric/protocol change was made.
5. **Provenance and file assumptions.** Git commit/dirty state is recorded, not an executable code snapshot.
   Do not edit training code during a batch if a single-code-version experiment is intended. Hash checks detect
   ordinary mutations, not adversarial write-and-restore attacks. Atomic JSON uses rename; sudden power-loss/fsync
   durability and hostile symlink replacement by another writer were not established.
6. **Draft persistence.** Confirmed mappings, trees and unapplied editor resume data persist within one TUI
   session; restarting restores batch monitoring, not an unsaved editing session. Field caret/scroll restoration
   is not promised by the current resume design. Choosing a different comparison reference is not supported;
   the oldest surviving tree remains the reference, independently of queue order.
7. **Terminal rendering — PARTIAL.** Headless geometry and Pilot mouse/key tests pass. Actual font rendering,
   terminal-specific key encoding and OSC 52 clipboard behavior require a terminal check.

Exact manual UI check without training: run `.venv/bin/python -m xmr4el_tui`, resize to 80×24, then 140×40;
press `n`, choose and confirm a local dataset, press `m`; traverse all three branches and eight slots; open an
editor, modify a value, use Preview, Esc and discard; exercise compact/normal/expanded/fit zoom and resize.
In Tree Selection add/duplicate/reorder/remove a tree, open Review with its button, Tab to Launch, and inspect
Overview/Differences. **Do not activate Launch for this rendering check.** Confirm no clipping and readable
focus/footer colors. Test Ctrl+C→Esc and the safe default in any active-batch quit dialog separately.

For reproducible isolated execution checks, use
`.venv/bin/python -m pytest tests/tui/test_workflow_e2e.py tests/tui/test_batch_audit.py -q`.
These tests provide their own fake runners and temporary roots; the production UI exposes no runner override.

## G. Recommendation

**Ready with limitations** for the verified local workflow after these repairs. No unresolved demonstrated
P0/P1 defect is intentionally deferred. Keep the explicit
crash policy, bounded preflight and untested Linux/real-terminal/real-training boundaries in view.
