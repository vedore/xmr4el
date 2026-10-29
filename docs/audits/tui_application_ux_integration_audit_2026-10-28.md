# XMR4EL TUI — Application UX, Navigation, Notifications and Integration Audit

Date: 2026-10-28  
Scope: application-level UI and integration, Textual 8.2.8  
Basis: the current working tree, including pre-existing uncommitted modifications

The audit changed no application code, tests or configuration. This file records the report subsequently requested by the user. References describe the implementation at audit time.

## A. Executive summary

The shared UI architecture is coherent and can be repaired with targeted changes. Ordinary Dashboard, Results, Inspection and Run Monitor navigation works well. Notification reliability has several confirmed integration defects.

- **Navigation:** normal back navigation preserves underlying screen state. Opening a notification destroys the current Results session.
- **Notifications:** genuinely session-only, with bounded visible history. Terminal-state reconciliation, cancellation attribution and reused run identities need correction.
- **Footer:** feedback replacement, expiration and confirmation priority work.
- **Quit:** accurately leaves detached experiments running. Oversized toasts can obscure its confirmation.
- **Readiness:** suitable for supervised browsing; fix the high-priority lifecycle and destination issues before relying on notifications during regular work.

No Critical finding or automatic experiment cancellation was identified.

## B. Architecture assessment

| Component | Actual ownership and relationships |
|---|---|
| `XMR4ELApp` | Owns the repository, current snapshot, notification manager, footer feedback, quit confirmation and launch handles. |
| Navigation | Textual’s screen stack is authoritative. Screens push/dismiss destinations; `App.go_to()` handles breadcrumb returns. |
| Shared navbar | Every main screen uses `TopBar`. Breadcrumbs derive from the stack and each screen’s `crumbs`; Inspection adds a local breadcrumb. |
| Notification manager | `Notifications` owns in-memory history, unseen count, previous run states and deduplication keys. Notices contain destination identifiers, not screen references. |
| Toast presenter | `App.toast()` converts newly created notices into Textual notifications. Textual maintains a separate transient presentation collection. |
| Notification center | A `ModalScreen` containing a compact overlay. It reads shared history, marks it seen and delegates destination navigation to the App. |
| Contextual footer | Each main screen mounts the same `ContextFooter`. State lives on the App; Textual supplies contextual bindings. |
| Shared Explorer | Dashboard uses `Explorer`; Results subclasses it for comparison styling and selection. Dashboard has no Results compatibility filtering. |
| Run observer | A serialized background scan produces `Snapshot`; the App observes lifecycle changes and posts `RunsChanged` to mounted screens. |
| Process ownership | `Launches` retains subprocess handles. Cancellation separately verifies process identity, runtime, ownership and process-group isolation. |
| Theme | `theme.py` defines semantic colors; `app.THEME`, TCSS and Python rendering consume them. |

There is no competing navigation history or problematic import cycle in the inspected paths. The main coupling is explicit: shared bars call App methods, and notification routing knows the destination screen classes. This remains understandable without a rewrite.

The significant architectural weaknesses are the notification transition model and coordination between domain history and Textual’s transient toasts.

## C. Findings

### F01 — High · Lifecycle reconciliation: temporary states permanently hide the verified outcome

References: [notifications.py:68](../../xmr4el_tui/services/notifications.py#L68), [experiment_repository.py:69](../../xmr4el_tui/services/experiment_repository.py#L69).

- **Evidence:** `observe()` replaces its previous-state map every scan and normally emits only when the immediately previous state was `RUNNING` or `UNVERIFIED`. Headless state replay produced:
  - `RUNNING → LOST → COMPLETED`: interruption notification, no completion notification.
  - `RUNNING → LOST → STOPPED`: interruption notification, no cancellation correction.
  - `RUNNING → UNKNOWN → COMPLETED`: no terminal notification.
- **Impact:** users can retain a false interruption warning or miss completion entirely. The repository reads recorded status before checking liveness, allowing an exit between those operations to produce a stale `LOST` snapshot.
- **Reproduce:** deliver any sequence above to `Notifications.observe()`.
- **Recommended fix:** reconcile dead processes against freshly read final status before publishing interruption; retain enough observation history to handle temporary unknown states and correct provisional outcomes.
- **Regression test:** exercise these sequences through repository scanning and notification ingestion, including cancellation racing process exit.

### F02 — High · Cancellation correctness: external termination is reported as requested cancellation

References: [notifications.py:53](../../xmr4el_tui/services/notifications.py#L53), [run_experiment.py:280](../../scripts/run_experiment.py#L280).

- **Evidence:** every `STOPPED` state becomes informational “Run cancelled — Stopped on request.” The runner records `stopped` for SIGTERM and KeyboardInterrupt without recording who requested them. External SIGTERM follows this same path.
- **Impact:** an externally terminated experiment can appear to have been intentionally cancelled.
- **Reproduce:** observe a running experiment, terminate it outside the TUI with SIGTERM, then rescan.
- **Recommended fix:** distinguish a verified cancellation request from a generic stopped process. Without intent evidence, use neutral wording that names the signal rather than asserting user intent.
- **Regression test:** compare confirmed TUI cancellation, external SIGTERM, external SIGKILL and cancellation racing successful completion.

### F03 — High · Destination integrity: reused directory identities can open another experiment

References: [notifications.py:38](../../xmr4el_tui/services/notifications.py#L38), [notifications.py:66](../../xmr4el_tui/services/notifications.py#L66), [app.py:154](../../xmr4el_tui/app.py#L154).

- **Evidence:** notification identity uses directory-name run IDs; evaluation targets add only the split. Destination validation checks the latest snapshot for those identifiers. An in-memory replacement with the same ID opened “Unrelated replacement” from the old notification.
- **Impact:** an old notification can open an unrelated replacement run. Deduplication also suppresses a replacement’s matching terminal event.
- **Reproduce:** retain a notice, remove its run, introduce another run under the same directory name, then activate the notice.
- **Recommended fix:** bind notices and deduplication to a run incarnation, using suitable start/process identity evidence, and validate it before navigation.
- **Regression test:** replacement under the same directory and split must reject the old destination and independently notify the new run. Normal TUI launches already reject reused console filenames; preserve that safeguard.

### F04 — Medium · Notification-center selection: new arrivals change the selected event

Reference: [notification_center.py:54](../../xmr4el_tui/screens/notification_center.py#L54).

- **Evidence:** row IDs are positional strings such as `"2"`. Refresh preserves that position after prepending new events. A selected notification with key `5` changed to key `6` without cursor movement.
- **Impact:** Enter can open a different experiment from the one the user selected.
- **Reproduce:** highlight an older row, deliver a new event while the center remains open, then press Enter.
- **Recommended fix:** preserve selection by notification identity, and preserve the corresponding scroll position.
- **Regression test:** prepend one and several events while a middle row is selected; verify that Enter still opens the original destination.

### F05 — Medium · Cross-screen navigation: notification activation discards comparison work

Reference: [app.py:169](../../xmr4el_tui/app.py#L169).

- **Evidence:** `open_notice()` pops everything above Dashboard before pushing its destination. A headless reproduction confirmed that the previous Results screen was removed from the stack.
- **Impact:** challenger membership, metric group, inspection and scroll position cannot be resumed after following a notification—even a failure notification leading to Run Monitor.
- **Reproduce:** build a comparison, open a run notification, then return. The prior comparison is gone.
- **Recommended fix:** preserve the originating screen; reuse an appropriate existing destination where possible. Opening another evaluation must not silently replace an active comparison.
- **Regression test:** follow both monitor and Results notifications from an active comparison, then return and verify all comparison and Explorer state.

### F06 — Medium · Responsive layout: notification-center controls extend below 80×24

Reference: [app.tcss:350](../../xmr4el_tui/app.tcss#L350).

- **Evidence:** with eight events, the popup measured `52×26` at `y=2`. Its footer started at `y=25`, outside a 24-row terminal. The list’s fixed maximum height ignores the popup’s other content.
- **Impact:** navigation instructions and the popup’s bottom are clipped at a practical terminal size.
- **Reproduce:** open a populated notification center at 80×24.
- **Recommended fix:** constrain the whole popup to available terminal height and allocate remaining height to its scrolling list.
- **Regression test:** verify popup and footer containment at all four requested sizes, including live resizing and full history.

### F07 — Medium · Toast bounds: long messages obscure controls and rapid deliveries exceed the intended limit

References: [app.py:131](../../xmr4el_tui/app.py#L131), [app.tcss:318](../../xmr4el_tui/app.tcss#L318).

- **Evidence:** toast width is constrained, but height and message length are not. A long error created a **36-row toast on an 80×24 terminal**, covering underlying content and part of quit confirmation. Four rapid deliveries created four simultaneous toasts.
- **Impact:** notifications can interrupt work and obscure confirmation instructions despite leaving keyboard focus unchanged.
- **Reproduce:** deliver a long error description, or several separate deliveries within four seconds. Manual rescans mean the five-second automatic polling interval is not a reliable concurrency limit.
- **Recommended fix:** bound toast content and height; enforce one or two visible toasts, consolidating overflow while retaining full history.
- **Regression test:** enable actual toast rendering, deliver long errors and bursts, and assert bounds, visible count and confirmation readability.

### F08 — Medium · Overlay coordination: existing toasts remain over the notification center

Reference: [app.py:148](../../xmr4el_tui/app.py#L148).

- **Evidence:** new toast creation is suppressed while the center is open, but opening it does not suppress existing Textual notifications. With rendering enabled, all four existing toasts appeared over the center.
- **Impact:** duplicated information covers notification rows and popup controls.
- **Reproduce:** trigger a toast and immediately open the center with Ctrl+N.
- **Recommended fix:** suppress active toast presentation while the center is visible, preserving domain history.
- **Regression test:** open the center during toast lifetime, deliver another event, close it, and verify visibility, expiry and unseen behavior.

### F09 — Medium · Startup timing: second-resolution timestamps miss short runs near initialization

References: [notifications.py:70](../../xmr4el_tui/services/notifications.py#L70), [run_experiment.py:225](../../scripts/run_experiment.py#L225).

- **Evidence:** the first scan records a microsecond-resolution cutoff; the runner records start time to whole seconds. A first scan at `12:00:00.500` rejects a newly discovered completed run recorded as starting at `12:00:00`, even if it actually began after that scan.
- **Impact:** a genuinely new, short experiment can complete without notification. Establishing the cutoff after discovery also creates a window during a slow initial scan.
- **Reproduce:** establish the baseline at half a second, then introduce a completed run started later in that second with the runner’s timestamp precision.
- **Recommended fix:** use consistent timestamp precision and establish the observation boundary before discovery; use explicit launch identity where available.
- **Regression test:** cover same-second startup and an experiment starting and ending during delayed initial discovery.

### F10 — Low · Navbar responsiveness: help can hide notifications and move the title

Reference: [top_bar.py:30](../../xmr4el_tui/widgets/top_bar.py#L30).

- **Evidence:** at 80 columns with Help open, the title began at column 34 rather than 37. The notification control disappeared at 80, 100 and 120 columns in the tested help layouts. At 40 columns, changing unseen count from one to twelve removed it.
- **Impact:** opening Help changes shared navigation geometry and removes mouse access to notifications.
- **Reproduce:** open Help at these widths, or use a narrow terminal with a multi-digit count.
- **Recommended fix:** retain full-width navbar space independently of Help and provide a compact notification fallback.
- **Regression test:** assert title position and notification hit area with Help, resizing and different counts.

### F11 — Low · Interaction clarity: the empty Dashboard advertises an unavailable action

References: [dashboard_details.py:244](../../xmr4el_tui/screens/dashboard_details.py#L244), [app.py:238](../../xmr4el_tui/app.py#L238).

- **Evidence:** the empty-state description says “n starts a training run”; `n` only displays “New Experiment: screen not built yet.” The global “new” binding also consumes limited footer space.
- **Impact:** users are directed to functionality that is unavailable.
- **Reproduce:** launch an empty workspace and press `n`.
- **Recommended fix:** describe the feature as planned and avoid presenting it as an enabled primary action.
- **Regression test:** verify that empty-state guidance matches implemented actions.

### R01 — Low · Retention risk: history is bounded, but deduplication storage is not

Reference: [notifications.py:61](../../xmr4el_tui/services/notifications.py#L61).

- **Evidence:** adding 10,000 distinct events retained 50 history items and 10,000 `sent` keys. Clear all retains those keys.
- **Impact:** memory grows with all events ever observed during a session. No practical memory failure was demonstrated.
- **Reproduce:** continuously add distinct events, including runs later removed from discovery.
- **Recommended fix:** define bounded retention for retired run identities while preserving repeated-poll deduplication.
- **Regression test:** verify retention behavior after history eviction, clearing and run disappearance.

## D. Requirements coverage matrix

| Feature | Status | Assessment |
|---|---|---|
| Shared navbar implementation | PASS | One reused widget across main screens. |
| Breadcrumb correctness | PASS | Stack-derived; focus and metric groups do not change route. |
| Breadcrumb returns | PASS | Dashboard return and closing Inspection work. |
| Ordinary title centering | PASS | Correct at all four requested widths and tested counts. |
| Navbar with Help/extreme narrowing | FAIL | F10. |
| Notification keyboard access | PASS | Ctrl+N works; duplicate center instances are prevented. |
| Session-only delivery | PASS | No notification persistence or history restoration. |
| Bounded visible history | PASS | Fifty items. Additional retention risk: R01. |
| Historical startup suppression | PASS | Existing completed/failed runs establish the baseline silently. |
| Startup transition completeness | PARTIAL | F09. |
| Repeated-poll deduplication | PASS | Repeated terminal snapshots do not duplicate events. |
| Final-state reconciliation | FAIL | F01. |
| Cancellation attribution | FAIL | F02. |
| Destination integrity | PARTIAL | Missing targets handled safely; reused identities fail, F03. |
| Multiple completed evaluations | PASS | Routes to Monitor rather than arbitrarily selecting a baseline. |
| Toast focus and expiry | PASS | Focus retained; actual rendered toasts expired. |
| Toast bounds and overlay coordination | FAIL | F07–F08. |
| Center empty state/history/clear all | PASS | Minimal empty state; clear touches notification memory only. |
| Seen behavior | PASS | Opening marks history seen; arrivals while open are immediately seen. |
| Center keyboard/focus restoration | PARTIAL | Basic restoration passes; arrival changes selected identity, F04. |
| Center responsiveness | FAIL | F06. |
| Contextual footer state model | PASS | Confirmation takes precedence over feedback and shortcuts. |
| Feedback timer replacement | PASS | Older expiry cannot clear newer feedback. |
| Quit semantics | PASS | Confirmation/cancellation work; detached runs survive exit. |
| Confirmation visibility | PARTIAL | Oversized toasts can cover it, F07. |
| Shared Explorer/Dashboard integration | PASS | Reuse, hierarchy, search, row handling and preservation covered. |
| Notification navigation preservation | FAIL | F05. |
| Central theme architecture | PASS | Semantic palette feeds Python and TCSS. |
| Real-terminal visual accessibility | NOT VERIFIED | Headless geometry/rendering inspected; no interactive terminal visual review. |

## E. Interaction and UX assessment

- **Navbar and breadcrumbs:** ordinary routes are accurate, including Logs and Inspection. The notification control is textual `Notifications`, with an unseen count. Truncation avoids overlapping the title, but removes breadcrumb hit areas when truncated. Breadcrumbs themselves are mouse-only; keyboard users use back navigation.
- **Toasts:** correctly positioned below the navbar, do not resize content and do not steal focus. Bounds and overlay coordination fail.
- **Notification center:** compact overlay rather than a full-page application destination. Rows are actionable, history scrolls, newest entries appear first, and clear all is isolated from experiment data. Selection identity and short-terminal layout fail.
- **Footer:** contextual bindings update with focus and Inspection. At 80 columns, Results Explorer shortcuts consume enough space to clip later global shortcuts. Panel cycling remains available through Ctrl+Tab and `]`/`[`.
- **Quit:** persistent footer confirmation; Esc cancels without leaving the screen or signalling a process. Quitting from a modal first closes that modal safely.
- **Explorer:** shared two-line tree rendering and full-item hit areas are implemented. Search and expansion preservation have existing coverage. Dashboard remains a general explorer.
- **Cross-screen flows:** normal Dashboard → Results → Inspect → back and Dashboard → Monitor/logs → back work. Notification destinations break return-to-existing-work behavior.

The only application `notify()` call is the lifecycle toast presenter. Routine invalid selection, incompatible comparison, challenger-limit and unavailable-destination messages use footer feedback. Missing logs use a persistent panel message, appropriately keeping the explanation with the unavailable content.

## F. Reliability and safety assessment

- **Startup reconciliation:** avoids historical notification floods, but its timestamp boundary misses some short new runs.
- **Deduplication:** effective for repeated observations of an unchanged run; insufficiently distinguishes replacement runs.
- **Lifecycle correctness:** completed, failed, interrupted and stopped states are supported. Completion depends on the recorded state, not merely file existence. Partial metrics do not turn a failed run into a success notification. Temporary-state reconciliation and cancellation origin remain incorrect.
- **Focus safety:** ordinary modal opening/closing restores focus, Explorer cursor and comparison state. Incoming-event selection drift remains a destination hazard.
- **Timers:** feedback replacement and expiry guards work. Rendered toast expiry also worked. Textual owns timer shutdown; a quit diagnostic with pending feedback completed without an exception.
- **Process safety:** cancellation verifies identity and ownership before signalling an isolated process group. External shell runs are monitor-only. Process tests passed once macOS inspection was available.
- **Actual shutdown:** a detached dummy process remained alive through opening confirmation, cancelling it and actually exiting the App. No UI workers remained registered afterward.
- **Cleanup:** one App scan subscription was found; no duplicate notification subscriptions. Filesystem scans, detail reads and monitoring reads use workers. Real network-filesystem stalls and prolonged shutdown under blocked I/O were not exercised.

## G. Theme and maintainability assessment

Styling is sufficiently centralized for another startup theme without editing every screen.

- Colors reside in `theme.py`; the existing test rejects hex literals elsewhere in TUI Python and TCSS.
- Notification title backgrounds use subdued semantic tints. Rendered success toast styling matched the shared palette.
- Whole notification cards use neutral backgrounds; no dominating status icons or redundant action buttons were found.
- Beam colors are deterministic and retain numeric labels.
- Status titles, caution markers and explanatory text provide information beyond color.

Some Python styles and `app.THEME` are constructed at import time. Theme replacement therefore needs to happen before those consumers initialize; live theme switching is not implemented.

Actual terminal contrast, font rendering and perceived readability remain visually unverified.

## H. Test results

| Command | Result |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/tui -q --tb=short` | 108 passed; 10 process-inspection failures inside sandbox. |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/tui/test_process.py tests/tui/test_procinfo.py -q --tb=short` | With process inspection permitted: **15 passed**. |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_run_experiment.py -q --tb=short` | 8 passed; one identity-read failure inside sandbox. |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_run_experiment.py::test_sigterm_records_stopped -q --tb=short` | With process inspection permitted: **1 passed**. |
| `uvx --offline ruff check --no-cache xmr4el_tui scripts/run_experiment.py tests/tui tests/test_run_experiment.py` | **Passed; no findings.** |

All **118 TUI tests and nine runner tests** passed across the runs and environment retries. No unresolved test failure remained.

Additional in-memory diagnostics exercised actual toast rendering, notification replacement, popup geometry, navbar resizing, same-second startup, destination replacement and App exit.

Important coverage gaps:

- Existing notification tests call `run_test()` without `notifications=True`, so their toast assertions do not exercise rendered toast layout.
- Terminal-state corrections, external termination attribution and reused run identities.
- Notification arrival while a specific destination is selected.
- Returning to an existing comparison after notification navigation.
- Popup containment, long toasts, toast concurrency and confirmation occlusion.
- Help-aware navbar geometry and real-terminal keyboard/mouse behavior.

## I. Prioritized remediation plan

| Priority | Corrections |
|---|---|
| **P0** | F01: reconcile final outcomes; F02: stop asserting unverified cancellation intent; F03: validate run incarnation before routing. |
| **P1** | F04–F05: preserve selected notification identity and originating screen state. F06–F08: constrain overlays and coordinate center/toast visibility. F09: correct startup timing. |
| **P2** | F10: navbar responsiveness. F11: truthful planned-feature guidance. R01: bounded deduplication retention. |

Use focused changes in the existing manager, router and shared widgets. A new navigation framework, notification database or broad screen rewrite is unnecessary.

## J. Final verdict

1. **Shared architecture coherent?** Yes.
2. **Notification system reliable?** Not yet; reconciliation, attribution and identity defects need correction.
3. **Truly session-only?** Yes.
4. **Quit safe and accurate?** Yes in tested process behavior; toast occlusion weakens confirmation visibility.
5. **Keyboard and mouse navigation consistent?** Mostly for ordinary flows; incoming notifications and responsive layouts expose inconsistencies.
6. **Shared components maintainable?** Yes; reuse and theme ownership are sound.
7. **Fix before regular use?** P0 lifecycle/destination issues and P1 notification interaction defects.
8. **Can wait?** Navbar polish, planned-feature wording and long-session retention optimization.

Not checked: real-terminal visual appearance, clipboard transport, GPU workloads or network-filesystem stress.

Remaining risk: notifications can misstate an experiment’s outcome, open a replacement run, or discard active comparison work.
