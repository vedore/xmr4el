# XMR4EL TUI design

Status (2026-10-30): describes the built TUI only: Dashboard, Run Monitor, Results, New Experiment (Dataset Setup,
Model Builder, Review Batch) and detached batch execution. The first plan (Runs / Run / Trees / Compare / Setup /
Review screens, removed 2026-10-27) is in git history (`58cdffd`). Code map: [tui_architecture.md](tui_architecture.md).

## 1. Concept

An experiment workbench over `outputs/runs/` and `outputs/batches/`:

- **Dashboard:** saved trees by dataset with their evaluations, a read-only description of the selected item,
  active runs and batches.
- **Run Monitor:** one run in any state: overview, pipeline stages, resources, live log.
- **Results:** one baseline evaluation against up to three challengers, with deltas only where comparable.
- **New Experiment:** Dataset Setup -> Model Builder -> Review Batch -> launch a batch of trees, watched in the
  Batch Monitor.

The TUI reads run and batch files and launches a detached coordinator that runs `scripts/run_experiment.py`; it
holds no training or evaluation logic. Files are the only durable state (plus the dataset registry and the palette setting), so a run or
batch outlives the TUI that started it and any TUI can show it.

## 2. Where it runs

| Place | Typical use | What works |
| --- | --- | --- |
| Server, in the `xmr4el` container, over SSH | Launch, watch, stop, compare | Everything. |
| Laptop with synced `outputs/runs/` | Browse, compare, read logs | Everything that needs no tree or process; trees (several GB each) are usually not synced. |

### Server over SSH, in Docker

Runs execute inside the image (`xmr4el.dockerfile`, code bind-mounted at `/app`). Keep
one long-lived container per user and attach the TUI to it:

```bash
docker run -d --name xmr4el-tui --init --gpus all --hostname "$(hostname)-xmr4el" \
  --user "$(id -u):$(id -g)" --mount "type=bind,src=$PWD,dst=/app" \
  --mount "type=bind,src=$HOME/.cache/huggingface,dst=/cache/huggingface" \
  -e HF_HOME=/cache/huggingface xmr4el sleep infinity
ssh -t server docker exec -it -e TERM -e COLORTERM xmr4el-tui python3 -m xmr4el_tui
```

- `--hostname` gives the container a readable name; each run records it (§9). Without it the name is the container ID.
- `-e TERM -e COLORTERM` passes the client terminal's color support into the
  container; otherwise Textual sees a plain `xterm` and drops to 16 colors.
- `--init` reaps finished children, so runs started by an exited TUI leave no zombies.

The coordinator and its runners are started in their own sessions (`start_new_session=True`) with stdout and
stderr written to files, never a pipe. An SSH drop or a TUI crash therefore does not signal or block them.
Reattaching is running the `docker exec` line again: the TUI rebuilds everything from the files. tmux is not
required.

### Laptop

Sync only run records, e.g. `rsync -a server:xmr4el/outputs/runs/ outputs/runs/`. Trees are absent: the tree's
Description says it is not on this machine. A run recorded as running on another host shows **UNVERIFIED**
(§9); the laptop never signals anything.

## 3. Workflows

| Task | Keys |
| --- | --- |
| Train and evaluate | Dashboard `n` -> Dataset Setup (confirm `ctrl+s`) -> `m` Model Builder -> `Review Batch` -> `l` launch -> Batch Monitor |
| Several configurations on one dataset | Model Builder: add / duplicate trees (`n`, `d` on the tab strip); one batch runs them in order |
| Resume an unfinished experiment | Dashboard `c` (the New Experiment screens as they were left) |
| Watch a run | Dashboard Active runs `enter`, or Description `right` -> Run Monitor |
| Watch a batch | Batch Monitor (opens at launch), Active runs `enter` on its row, or Dashboard `b` -> Batches |
| Recover from a failure | Run Monitor (failed stage, log) -> `c` / `n` on the Dashboard, fix, launch again |
| Compare | Dashboard `r` on an evaluation (baseline) -> Results `space` on up to three challengers |

There is no training resume, and no eval-only launch from the TUI.

Commands outside the TUI keep their CLI: grouped-TSV training (`train.py`), evaluating a saved tree on another
split (`run_experiment.py -xmodel_path`), dataset conversion and splitting, `diagnose_routing.py` /
`diagnose_unseen.py`, rerankers and external baselines.

## 4. Visual language

**Layout.** One header line, one footer line, content between. Blank lines separate
groups. Panels are bordered boxes with a title; the focused one has the accent border. Help (`?`) replaces the panel and `esc`
closes it. Scrollbars are one cell wide and muted.
`ctrl+c` / `ctrl+q` asks in the footer (`Confirm  Quit XMR4EL?`), a second `ctrl+c` quits, `esc` cancels; no
timer. Runs keep running after a quit (TUI launches are detached: own session, output to files), and the question
says so when runs are active. `ctrl+r` says the run count in the footer.

**Navbar.** Breadcrumbs left (`Dashboard / Results / Inspect`), `XMR4EL`
centred on the app width (never moved by the crumbs, the bell or the help panel, which docks under the navbar and
above the footer), `Notifications` right with the unseen count (`Notif` when that does not fit; text: no emoji anywhere in the TUI); one blank row under the
navbar separates it from the panels. Crumbs come from
the screen stack (each screen's `crumbs`; modal screens add none): no second navigation history. Focus, panels and
metric groups never change them. A click on an earlier crumb goes back to it (screens above close as with `esc`;
`Results` while inspecting closes the inspection); too long, they are cut on the left (`… / Inspect`). No host,
counts or other summaries (they live in the panels).

**Footer confirmations are modal.** While a screen's footer asks (`App.ask`, or the quit question), `App.on_event`
drops every key, paste and mouse press / release / click / scroll except the question's own key, `esc` and `ctrl+c` /
`ctrl+q`; mouse moves pass. Nothing underneath can run or leave the question stale.

**Three message layers.** (1) *Notifications*: experiment lifecycle only, in memory for the session
(`services/notifications.py`, 50 newest). The first scan records each run's state and notifies nothing; a run seen
RUNNING / UNVERIFIED and later seen COMPLETED / FAILED / LOST / STOPPED, or a run started during the session (at or
after the first scan's start, to the second) first seen already ended, makes one notice per (incarnation, end
state). Incarnation = run id + recorded start + process identity: a new run reusing a directory name is a new run,
and an old notice never opens it. LOST is provisional: a process found gone is re-read once (a final status written
meanwhile wins), the run stays watched, and a final status read later replaces the interruption notice; UNKNOWN
(status.json unreadable) is waited out. STOPPED is "Run cancelled · Stopped on request" only when this TUI sent the
stop (Cancel Run); otherwise "Run stopped · Stopped by SIGTERM" (the recorded signal, no intent claimed). A launch
that exits without a final status uses the LOST key (reported once); a new scan problem is a notice without
destination. Keys of runs whose directory is gone are forgotten (memory bounded by outputs/runs/). Destination: one
evaluation with metrics → Results with it as the baseline; otherwise (several evaluations, failure, interruption,
stop) → Run Monitor; pushed over the current screen (`esc` returns to it as it was; an open comparison is never
replaced; the monitor already shown is not opened twice), checked against the latest scan by incarnation (gone or
replaced: footer says so). Toast: top right under the navbar, one at a time (a newer delivery replaces it; several
events = one summary), lines cut to 88 characters, at most 6 rows, 4 s; none while the center is open, and opening
the center removes the shown one; the severity tints only the title chip. Center (`ctrl+n` or a click on
Notifications): a 52-column box under Notifications over the current screen (ModalScreen: focus, cursor, scroll,
comparison of the screen below unchanged); its list gets the rows the terminal leaves (whole box visible at 80x24);
the selected row stays on its notice (and on screen) when newer ones arrive; opening marks all seen; `↑↓` `enter` /
click opens, `c` / `Clear all` empties the history only, `esc` closes. (2) *Footer feedback*: replies to the user's actions (invalid Results target, maximum
challengers, incompatible evaluation, missing run, copy, rescan, dataset confirmed, component applied) for 3.5 s,
over the whole footer in place of every shortcut, on the `feedback-row` tint (accent over surface, never the
confirmation's); a newer one replaces it and an older timer cannot clear it; it belongs to its screen (navigation
drops it); never kept as a notification. (3) *Footer confirmation*: quit, discard, remove; replaces the shortcuts
until answered, on the `confirm-row` tint; feedback timers and notifications never replace it. One state at a time:
confirmation > feedback > shortcuts. Comparison Caution stays
in Results.
The right side is the same on every screen, so state is always in one place.

**Footer.** The main context keys only, then `? help` always last; it must fit 80 columns.
Secondary keys (`/` filter, `y` copy path, `ctrl+p` commands) work everywhere and are listed
in help, not in the footer.

**Wireframe notation** (this document only):

| Notation | Meaning |
| --- | --- |
| `[ value ]` | Text input or select |
| `< Start >` | Button |
| `Overview` with `========` under it | Active tab |
| `>` in the first column | Selected row |
| `*` in the second column | Marked row |

**State words** (always text; color is extra):

| Word | Color | Meaning |
| --- | --- | --- |
| RUNNING | accent | Recorded running; process verified alive on this host. |
| UNVERIFIED | amber | Recorded running, but local process identity cannot be established. |
| LOST | red | Recorded running on this host, process gone, no final status (e.g. OOM kill). |
| COMPLETED | green | All evaluations written. |
| FAILED | red | Runner recorded an error. |
| STOPPED | amber | Runner recorded SIGTERM or Ctrl+C. |
| UNKNOWN | plain | `status.json` missing or unreadable. |

LOST and UNVERIFIED are derived on screen; the TUI never rewrites `status.json`.
Narrow cells (Run Monitor pipeline, Results comparison rows) use short words: RUN, UNVER, LOST, DONE, FAIL,
STOP, `?` (`fmt.STATE_SHORT`, same colours). Headers and details use the full word.

**Formats.**
- Timestamps: `MM-DD HH:MM` in tables, `YYYY-MM-DD HH:MM` in details.
- Durations: `m:ss` under an hour, `h:mm:ss` above. Everywhere, including the header.
- Metrics: 4 decimals; a percent only for coverage (`96.3%`).
- Counts: thousands separators. Latency: `2.99 ms/query`.
- Tree IDs: first 8 hex characters; full ID in Description.

**Stage markers.** Plain text (no symbol glyphs): `+` done, `>` current, `·` pending, `x` failed.
ASCII (`NO_COLOR` or a non-UTF-8 locale): `ok`, `>>`, `..`, `xx`.

**Style.** XMR4EL's palette is five brand colours, paper `#FAF6EE`, ink `#2B2A28`, terracotta `#B8492A`, steel
blue `#3F6E8C`, olive `#7A8F4E` (the README figure uses them as they are), plus two status-only colours, ochre
`#E2B35A` (warning) and rose `#E9707F` (error, never the accent's hue). `xmr4el_tui/theme.py` is the only file with
hex values (a test enforces it). A palette is a row of `theme.BRANDS` (paper, ink, accent, info, success, warning,
error, dark or light); `theme.build` derives every semantic token from it: application (background, surface, panel
border, text / muted / disabled), selection (accent, `on-accent`, `cursor-row` = 40 % accent over surface,
`focus-outline`), status (success, warning, error, info, unknown, `warning-on-accent`), comparison (baseline / member
rows, delta), the beam chips. Dark palettes put text in paper on an ink surface, light ones the reverse; each hue is
mixed toward the text colour by the smallest step that reaches WCAG AA (4.5:1) on the surface, so every palette is
readable by construction (`tests/tui/test_palette.py`). Palettes: **Notebook** (default, dark), **Notebook light**,
**Classic** (the first teal look). `app.THEMES` registers one Textual theme per palette (Textual's own themes are not
offered) and exposes every token as `$xmr-<token>` in app.tcss; Python Rich styles use `theme.color` /
`theme.style`, and style tables are `theme.Lazy`, read at each use.

**Changing the palette.** On the Dashboard, `ctrl+p` lists `Palette: <name>`. Choosing one switches every colour at
once (`App.set_palette`: `theme.use` swaps `theme.TOKENS` in place, the Textual theme changes, the Dashboard redraws
and its explorer is relabelled in place) and keeps the choice in `outputs/tui/settings.json` (`{"palette": name}`,
other keys kept; read at start, a missing / unreadable / unknown value means the default). Not offered above the
Dashboard: open screens keep the colours they were built with, so the switch happens where nothing else is open.
A new palette = one more `BRANDS` row. The cursor row is
painted across the whole row (explorer guides, every table column, the padding after them); a blurred panel keeps
a dim accent cursor (selected, not focused); the cursor row of Description is one colour (an olive COMPLETED
would vanish on the accent); explorer guides of the cursor's branch stay accent (on a leaf or a collapsed node,
its parent's branch); the focused panel has the accent border and title. Status colours
stay semantic (`fmt.STATE_STYLE`). Textual honours `NO_COLOR`; that is the monochrome mode.
No animation except the encoding progress bar.


## 5. Screens

### 5.1 Dashboard

```text
 Dashboard                                XMR4EL                         Notifications

╭ Explorer ──────────╮╭ Description — Trained tree ─────────────────────────────────────╮
│▼ Models            ││ ncbi_disease · SapBERT D2                                       │
│├── ▼ ncbi_disease  ││ Tree id       a64127ef…                                         │
││  ├── ncbi_disease ││ Dataset       ncbi_disease                                      │
││  │   · SapBERT D2 ││ Saved at      outputs/saved_trees/xmodel_2026-10-28_…           │
││  │   Oct 28 ·     ││               on this machine                                   │
││  │   a64127ef     ││ Training run  ncbi_disease · 2026-10-28_…                       │
││  │   └── dev      ││ View Logs     [Open]                                            │
│▶ Runs without a    ││ Run state     COMPLETED                                         │
│  tree              │╰─────────────────────────────────────────────────────────────────╯
│                    │╭ Active runs ─────────────────────────────────────────────────────╮
│                    ││ Run        Type   Model     State    Elapsed  Stage  Started  …    │
╰────────────────────╯╰─────────────────────────────────────────────────────────────────╯
 r results  b batches  n new  c resume  ? keys
```

*Illustrative.* The home screen; every other screen is pushed over it and `esc` returns to it as it was left.

- **Explorer** (22 % of the width, resizable, `ctrl+b` hides it): `explorer.build` over the app's scan. `Models` >
  dataset (training file's directory) > tree (two rows: `name · descriptor` from its training run's model config,
  then training date and short tree id) > evaluation (split, state). Then `Active, no saved tree yet`
  (recorded-running runs without a tree) and `Runs without a tree`. `/` searches (§7). The same widget is the
  Results explorer.
- **Description** (2/3 of the right column): read-only rows of the selected item; files are read in a worker
  (`ExperimentRepository.details`; `xmodel.json`, never a pickle). Tree: id, dataset, saved location (`on this
  machine` / `missing: … (moved, deleted or not synced)`), labels, training run, `View Logs`, run state, training
  file, model config, the tree's saved configuration and search defaults. Evaluation: run, `View Logs`, state,
  tree, test file, strings file, times, metrics and search. Run without a tree: id, type, state, stage, device,
  files. A failed run starts with its error banner. `left` opens Results, `right` the Run Monitor, `View Logs` the
  fullscreen log.
- **Active runs** (1/3): one row per running batch (`Batch`, tree count, state, current tree) above every
  RUNNING / UNVERIFIED run (type, model, state, elapsed, stage, started, device, run id). `enter` opens the Batch
  Monitor or the Run Monitor.
- Footer: `r` Results with the selected evaluation as baseline (a tree with one evaluation with metrics: that
  one; several: the tree opens to choose), `b` Batches, `n` New Experiment, `c` Resume (only while an unfinished
  draft exists).
- Read-only: nothing here writes, moves or deletes a file (a palette switch, §4, writes only
  `outputs/tui/settings.json`). Without `outputs/runs/` the group description says
  `outputs/runs/ not found under <root>. Start in the repository or pass --root.`

### 5.2 Run Monitor

```text
 Dashboard / Run Monitor                 XMR4EL                                        Notifications
╭ Run overview ─────────────────╮╭ Resources — Encode features ─────────────────────────────────╮
│ bc5cdr_chemical               ││ Completed  4:00                                              │
│ Type     Training + Evaluation││ 49 samples, every 5 s plus stage start and end               │
│ Dataset  datasets/bc5cdr      ││ CPU avg     412.5 %   this process (all threads) + exited …  │
│ State    RUNNING              ││ RAM avg     9.0 GiB   this process                           │
│ Elapsed  14:38                ││ RAM peak    11.0 GiB                                         │
│ Device   cuda                 ││ GPU util    81.5 %    whole CUDA device, sampled (NVML)      │
│ [ Cancel Run ]                ││ GPU memory  8.0 GiB   peak, this process, PyTorch tensors: … │
╰───────────────────────────────╯╰──────────────────────────────────────────────────────────────╯
╭ Pipeline ─────────────────────╮╭ Live logs ───────────────────────────────────────────────────╮
│   Whole run       14:38  RUN  ││ level all · following                                        │
│   + Load data      0:06       ││ 2026-10-23 14:19:00,000 [INFO] xmr4el.hierarchy: …           │
│ ▶ + Encode features 4:00 …    ││ …                                                            │
│ ▼ > Hierarchy training  layer…││                                                              │
│ ├── + layer 1   2:00  1 nodes ││                                                              │
│ └── > layer 2   node 3/6 · …  ││                                                              │
│   · Save tree                 ││                                                              │
╰───────────────────────────────╯╰──────────────────────────────────────────────────────────────╯
esc dashboard  s cancel run  c follow stage  f fullscreen  ? keys
```

- One run, by run id (`DashboardScreen.open_run`); `esc` returns to the Dashboard as it was left. A run that
  ends stays on screen (final state, durations, log); the screen never moves to another run.
- Left column 35 %, right 65 % (resizable, §7). Run overview is content-sized; Pipeline fills the rest; Resources
  is 35 % of the right column, Live logs the rest. Below 74 columns or 20 rows: one panel at a time (the focused
  one, maximized; `]` / `[` switch, `esc` goes back).
- Pipeline rows: Whole run, then `progress.stages` (the runner's events only): encode parts, eval parts per
  split, hierarchy layers (a layer has only a start event: it ended when the next one started or with the
  hierarchy). Durations of ended stages only; no percentages; the transformer tick shows done / total as
  emitted. Opens on, and follows, the current stage (else the failed / stopped one) and expands its parts once;
  a cursor move stops following, `c` resumes. Rows are relabelled in place, so cursor and expansion survive.
- Resources: the resources.json summary of the selected top-level stage, matched by (stage, split), the last one
  recorded. A part shows its parent's figures, labelled so. A running stage: `Collecting resource statistics…`;
  pending: `Not started.`; a run without resources.json says so. Missing values are `N/A`, never 0. Scopes are
  resources.json's own `scope` strings (GPU utilisation is device-wide NVML; GPU memory is PyTorch tensors of
  this process). Whole run: only what stage summaries give exactly: CPU time / wall time over the ended stages,
  the highest RSS and GPU-memory peaks; RAM and GPU-utilisation averages are left to the stages.
- Live logs: `LogPanel`, the fullscreen viewer's reader (one LogTail, 1 s while active, one final read).
- Cancel Run: `process.stop(run)` (SIGTERM, graceful only; no force button) in a worker after `StopDialog`.
  Disabled once requested until the outcome; a refused stop (`not signalled: …`) re-enables it. A run that ends
  while the dialog is open closes it and sends nothing; `process.stop` also re-reads status.json first.
- Polling: events.jsonl from its last offset (`progress.EventTail`, shared with the scan's Activity) and
  resources.json only when its (inode, mtime, size) changed, every 1 s while active; state from the app's 5 s scan.


### 5.3 Results

```text
 Dashboard / Results                          XMR4EL                                        Notifications
╭ Current Baseline ─────────────────╮╭ Comparison Summary ──────────────────────────────────────────────╮
│ ncbi_disease · a64127ef · dev  B10││      Evaluation                   Acc@1_all     Acc@1        Cov… │
╰───────────────────────────────────╯│Base B10 ncbi_disease · a64127ef…  89.17%        92.59%       96.… │
 X Show incompatible                 │1    B10 ncbi_disease · 878b59c2…  89.17% +0.00  92.59% +0.00 96.… │
╭ Explorer ─────────────────────────╮│2 !  B10 ncbi_disease · 8528b624…  89.04% -0.13  92.46%  Δ —  96.… │
│▼ ncbi_disease  3                  ││──────────────────────────────────────────────────────────────────│
│├── ▼ ncbi_disease · SapBERT D2    ││! 2 ncbi_disease · 8528b624 · dev: in-vocabulary population not   │
││      Oct 28 · a64127ef           ││    verified: no population hash, and runs not all at one clean …  │
││     └── i B10 dev  baseline      ││    Matches test file sha256, 757 loaded rows; no deltas for …     │
││                                  │╰──────────────────────────────────────────────────────────────────╯
│├── ▼ ncbi_disease · SapBERT D2    │╭ Detailed Analysis ───────────────────────────────────────────────╮
││      Oct 28 · 878b59c2           ││Metric      Base B10         1 B10             2 B10               │
││     └── i B10 dev                ││Acc@1_all   89.17%           89.17% +0.00      89.04% -0.13        │
││                                  ││R@5         96.71%           96.71% +0.00      96.71% Δ —          │
│└── ▼ ncbi_disease · SapBERT D2  ! ││Δ vs baseline in percentage points … · Δ — not verifiably compar… │
│       Oct 20 · 8528b624           │╰─── 1 Performance  2 Ranking  3 Generalization  4 Efficiency  5 Specs╯
╰───────────────────────────────────╯
⏎ inspect  space add / remove  / search  esc back  1-5 metric group  ? keys
```

Baseline row (Summary) and member rows (Explorer) are on the accent; the explorer cursor row is `cursor-row`.

- Opened from the Dashboard (`r`, or `left` in Description) with an evaluation as the **baseline**: the selected
  evaluation, or a tree's only evaluation with metrics. A tree with several opens in the explorer instead, so the
  user chooses; Results never picks the latest or best. Each opening is a new session (no challengers); `esc`
  returns to the Dashboard as it was left. The baseline cannot be changed inside Results.
- One layout for every state: zero to three **challengers** (`comparison.MAX_CHALLENGERS`) next to the baseline.
  Identity is the explorer key `("eval", run id, split)`, never a label. Left column 30 % (resizable as the
  Dashboard's: `<` `>`, border drag), Current Baseline (one line) over the Explorer; right: Comparison Summary
  (content height, at most 40 %, then it scrolls) over Detailed Analysis (its table keeps a body of at least
  three metric rows at 80x24).
- Explorer = the Dashboard's `Explorer` (same model, `explorer.build` Models branch: dataset > tree >
  evaluation; same two-row trees, gap rows and `/` search, §7), as `ResultsExplorer`; trees open at once. Shows
  only the baseline and evaluations assessed VERIFIED or CAUTION (per evaluation: a tree stays while one of its
  evaluations does, a dataset while one of its trees does; not yet assessed = hidden). **Show incompatible** (a
  one-row view filter above it, focus with `ctrl+tab`, `space` / click toggles, off in every new session) also
  lists INCOMPATIBLE ones (dim, `x`): inspectable, never addable; toggling keeps members, baseline, group,
  inspection and expansion, and a hidden cursor row falls back to its nearest shown ancestor. A search or the
  filter hiding a member never removes it. Rows: `i` (click = inspect), beam chip, a member's role number (`1`-`3`,
  its Summary row and Analysis column, before the label so a narrow explorer keeps it), label, `baseline` / `!` Caution /
  `x` Incompatible (dim); a tree shows `!` when one of its evaluations is Caution. Keyboard cursor = `cursor-row`
  background; comparison member (and the baseline) = accent background, kept when the cursor leaves; the cursor
  on a member keeps the accent and adds `focus-outline` bars at both row edges. `space` / click on an evaluation
  adds or removes it; a click on a tree or dataset expands it; `enter` / `i` inspects. A fourth challenger, the
  baseline and Incompatible evaluations are refused with a toast.
- **Compatibility** (`services/comparison.assess`, one place): INCOMPATIBLE = no metrics, a metrics file whose
  `test_path` / `train_path` / `split` disagrees with its spec, test-file sha256 missing or different
  (provenance.json, never file names), or different / missing `rows_total`. A hash or commit that is not a
  non-empty string counts as not recorded. Otherwise
  VERIFIED, unless something needed for a delta cannot be checked: CAUTION, with `reason` (what is missing),
  `matches` (sha256, loaded rows) and `affects` (which deltas are withheld). Every delta needs one recorded
  `eval_protocol` (metric definitions, `xmr4el.eval.EVAL_PROTOCOL`): missing = CAUTION, different = a note; either
  withholds all deltas. In-vocabulary deltas (acc@1, MRR,
  R@k, R@cand, hybrid) need equal `population_sha256`, or the §11 derivation (same tree training-file sha256,
  preprocessing, one clean commit); a known difference (hashes or training files differ) withholds them as a note,
  not a Caution. Vocabulary, beam and search settings never block. Generalization deltas need equal
  `strings_sha256` (per evaluated row: loaded position, train labels of its mention string, dictionary answer, so
  vocabulary and abbreviation expansion count); missing = CAUTION. An empty string group records null rates
  (`N/A`, no delta). No timing deltas at all (hardware not recorded).
- Summary: baseline row on the accent, challengers in selection order (`2 !` marks Caution by the role);
  Acc@1_all (the comparison metric, fixed), Acc@1, Coverage (rows / rows_total) beside the name (name cut to the
  panel, at most 30 cells; values and deltas padded into aligned columns; narrow: the delta wraps under its value).
  Below a rule, per Caution challenger: its reason, then (muted) what matches and which deltas it affects. No
  instructions, no ranking or winner; as tall as its content. Detailed Analysis: one group (`1`-`5`: Performance,
  Ranking, Generalization, Efficiency, Specs) for every evaluation at once, one column each (scrolls horizontally
  when narrow); value and delta (percentage points; MRR on its own scale). A delta the compatibility service
  withholds is `Δ —` (muted; both values recorded), never 0 or blank; the value stays. A missing metric is `N/A`.
  Notes: the group's note, known differences, one Δ legend; Caution reasons stay in the Summary. Efficiency =
  measured times and candidates/query (no timing deltas). Specs = recorded `beam_size` (beam colour), `topk`,
  `knn_beta`, `leaf_candidates`, `features`, no deltas, a value other than the baseline's in `spec-differs`.
- Inspection (`enter`): the explorer's evaluation (member or not) replaces the right column, read-only: identity,
  beam, standing against the baseline (level, reason, matches, affects), every group. The explorer is disabled;
  `esc` restores members, group, cursor and expansion. Not a screen.
- `esc`: inspection → closes it (focus on the explorer); Summary, Analysis or the filter → the explorer; an
  explorer search → cleared; the explorer → the Dashboard as it was left. Focus moves only: nothing reloads.
- Data: metrics from the scan; `comparison.read_facts` (provenance / git / spec / model_config of the run and its
  tree's training run) for every evaluation in a worker, rerun only when the scan's runs change (provenance.json
  and git.json are in the scan signature). A FAILED run's written metrics stay usable. A baseline whose run
  vanished keeps its last read values, marked `no longer found` in Current Baseline; an inspected evaluation that
  vanished says so (not "Reading").
- Beam colour: `theme.beam_color(beam) = palette[beam mod 6]`, the same everywhere, never discovery order; the
  number is always written (`B10`).


### 5.4 New Experiment: Dataset Setup

```text
 Dashboard / New Experiment                XMR4EL                        Notifications

╭ Datasets ──────────╮╭ Dataset Configuration ──────────────────────────────────────╮
│   BC5CDR — Chemical││ BioRED — Chemical                                           │
│    Ready           ││ Variant      Chemical                                       │
│ > BioRED — Chemical││ Format       PubTator (the experiment runner's input)       │
│   Ready · confirmed││ Directory    datasets/BioRED/chemical                       │
│   ncbi tiny        ││ Recognized   biored_chemical.json, …, datasets/ scan        │
│    Missing files   ││ Status       Ready  ·  Confirmed                            │
│ Register local     ││                                                             │
│ dataset            ││ Training file                                               │
│                    ││ ( ) datasets/BioRED/chemical/train.pubtator            Ready │
│                    ││ (x) datasets/BioRED/chemical/train_plus_ctd.pubtator   Ready │
│                    ││     Choose another file…                                    │
│                    ││ Evaluation files                                            │
│                    ││ [x] Development  datasets/BioRED/chemical/dev.pubtator   Ready │
│                    ││ [ ] Test         datasets/BioRED/chemical/test.pubtator  Ready │
│                    ││   Confirm Dataset                                           │
╰────────────────────╯╰─────────────────────────────────────────────────────────────╯
 enter choose file  space toggle  esc back  ctrl+s confirm dataset  u discard changes
```

- `n` (Dashboard footer) opens it; the breadcrumb is `Dashboard / New Experiment`. Datasets 30 % (min 24 cells),
  Dataset Configuration the rest; both scroll on their own.
- Datasets = `datasets.discover`: one logical dataset per directory of the files the experiment specs name
  (`configs/experiments/*.json`; eval-only specs too; model configs there are skipped; a spec's split role is its
  declared split name, `dev`, `dev_*` -> Development, `test`, `test_*` -> Test), plus the split files of those
  directories and of every `datasets/` directory with a dev or test file (`runs.split_files`,
  `runs.training_files`). Disease / chemical subsets are separate directories, so separate datasets. Then the
  registered datasets (`outputs/tui/datasets.json`, label `local`), then `Register local dataset`.
- A recognized path is a suggestion: availability is `datasets.inspect` (stat + the first 64 KiB: UTF-8 text,
  `<id>|t|` before abstracts and annotations, tab-separated annotation lines, at least one annotation when the whole
  file fits; cached by mtime and size). No full parse, hash or embedding: launch-time checks stay with the runner and
  Review Batch. List statuses are checked in a worker.
- Defaults: a training file only where one is recognized, or exactly one a spec names (train_plus_ctd over
  train.pubtator; both listed, `space` / `enter` picks one); Development enabled when recognized; Test never enabled
  by default. Several recognized files for one role and none chosen: Needs review.
- Status (list and panel): Missing files > Invalid mapping > Needs review > Ready; Ready only when every check above
  passed. The specific message sits under the affected row (`Development file not found: datasets/C/dev.pubtator`);
  Confirm Dataset with a problem says the first one in the footer and moves the cursor to the control that fixes it.
- Checks: training file set and found; at least one evaluation split enabled; each enabled split's file set and
  found; readable; PubTator; an evaluation file that is the training file; duplicate split names; registered name
  set and unique.
- Draft (`app.draft`, `draft.ExperimentDraft`, memory only): `selected` = the dataset in the right panel (enter on
  a dataset; moving the cursor changes nothing); `edits` = per dataset, changes not confirmed (kept when switching
  datasets or leaving the screen; the list says `edited`); `confirmed` = the `datasets.Selection` Confirm Dataset
  (`ctrl+s` or the row) validated: dataset key, name, variant, training file, evaluation splits (path, enabled),
  format. The next New Experiment screens read `draft.confirmed`; every tree of the batch shares it (`draft.trees`,
  Model Builder). Changes after confirming leave `confirmed` as it was until confirmed again (`Confirmed · changed
  since, not confirmed`). `u` discards the dataset's unconfirmed changes after the footer confirmation (`u` again,
  `esc` keeps them).
- Register local dataset: a Name field, then the training / Development / Test files chosen with the picker (a
  split's role is the row it is chosen for, never guessed from its name); confirming writes the registry (only file
  this screen writes; an unreadable registry is never overwritten) and confirms it. Bundled specs and datasets are
  never modified.
- File picker (`screens/file_picker.py`, a screen, crumb `Training file` / `Development file` / `Test file`): a path
  field over its directory's entries, on the machine the TUI runs on. Typing filters by the text after the last
  `/`; a typed or pasted path to a directory opens it, to a file chooses it; `up` `down` move, `enter` opens / chooses
  the highlighted entry, `..` goes up (past the project root too); `.pubtator` / `.txt` normal, other files dimmed
  but selectable. Starts in the current file's directory with it highlighted. `esc` cancels: mapping, focus and
  row unchanged. A file chosen for a split enables it. Paths are kept relative to the project root when inside it.


### 5.5 New Experiment: Model Builder

```text
 Dashboard / New Experiment / Model Builder        XMR4EL                         Notifications

╭ Model Builder ─────────────────────────────────────────────────────────────────────────────────╮
│ 1 Tree 1  2 Small k   Add tree                                                                 │
│             Feature construction                                                               │
│ Transformer [Transformer       ]──────────────────────┐                                        │
│ Mention     [Character TF-IDF  ]──[Mention SVD       ]┼──[PIFA ]──[Hierarchy ]──[Matcher ]──[Search ]
│ Context     [Context TF-IDF    ]──[Context SVD       ]┘                                        │
╰──────────────────────────────── NCBI Disease · Training: train_plus_ctd · Evaluation: dev ─────╯
╭ Hierarchy · Balanced KMeans ─────╮╭ Configuration · Tree 1 ───────────────────────────────────────╮
│ Clusters per node   128          ││   "clustering_config": {            (accent tint: this slot) │
│ Iteration limit     20           ││     "type": "balancedkmeans", ...                             │
│ Tree depth          2            ││   "depth": 2,                                                 │
│ Cut half cluster    Yes        v ││   "matcher_config": {                       (muted: the rest) │
│ Apply   Cancel                   │╰───────────────────────────────────────── unapplied changes ───╯
╰──────────────────────────────────╯
```

- Opened from Dataset Setup (`m`, or the `Model Builder` row) once a dataset is confirmed; crumb `Model Builder`;
  `esc` from the grid returns to Dataset Setup (trees and positions are kept in `app.draft`). Model Builder 45 % /
  lower panel 55 % of the rows between the navbar and the footer; the lower panel is the Component Library, or the
  Component Editor (Attributes 40 % | JSON preview 60 %, each scrolling on its own) while a component is edited.
- Hybrid recipe only (`features: "sapbert_char_context"`): eight slots, `components.SLOTS`: Transformer
  (`transformer_config`), Character TF-IDF (`vectorizer_config` + `abbrev_expansion`), Mention SVD
  (`dimension_config`), Context TF-IDF (`context_vectorizer_config` + `context_window`), Context SVD
  (`context_dimension_config`), Hierarchy (`clustering_config` + `depth`, `min_leaf_size`, `cut_half_cluster`,
  `routing_features`), Matcher (`matcher_config`), Search (`predict_config`: the tree's default search, saved with
  it; evaluations may override). PIFA is drawn as a fixed step, never selectable. `verbose` and unknown keys are
  kept as they are, not edited.
- Library = only what the backend loads: SapBERT, BioBERT, Sentence-T BioBERT (`MODEL_NAMES`), Custom Hugging Face
  model (`kwargs.model_name`, no `type`); TF-IDF; Truncated SVD; Balanced KMeans; Joint Logistic Regression; Beam
  search. The installed one says `installed`. Moving changes nothing; `enter` / click opens the editor: the
  installed component with its applied settings, another one from its defaults (`{"type": ..., "kwargs": {}}`), the
  slot's other keys kept.
- Editor: one row per attribute (`components.*_FIELDS`, labels, types, ranges and defaults mirroring the backend;
  `tests/test_tui_components.py` checks them against it). Text for numbers / paths / ids, Yes/No and enumerations
  as selects, n-gram range as two numbers. A field the config leaves out shows the backend default and stays left
  out unless set to something else; cleared optional fields are left out (`revision`, `max_seq_length`,
  `max_features`) or null where null means something (`random_state` unseeded, `context_window` whole document).
  Floats stay floats (`10` -> `10.0`); TF-IDF `max_df` / `min_df` keep the scikit-learn split (`1.0` share, `1`
  count). Text that does not parse is an error under the field, never 0; the preview keeps the last valid value and
  its border says `1 invalid field: last valid value shown`.
- Draft / Apply: the editor works on a copy of the slot's keys; the tree is unchanged until Apply (`ctrl+s` or the
  button): every field of the slot is checked (`components.slot_problems`), then the tree's configuration is
  replaced in one assignment, the editor closes, the grid keeps the slot, the footer says `Hierarchy: Balanced KMeans
  applied to Tree 1`. A problem: nothing applied, the first bad field focused, the footer names it. Cancel drops the
  copy. Component validity only: launch readiness is Review Batch's (§5.6).
- JSON preview: `json.dumps(config, indent=2)` of the tree with the copy merged in (the dict Apply commits). Lines
  are owned by their top-level key (`components.json_lines`), so the slot's keys (all of them, e.g. `clustering_config`
  and `depth`) get the `section` tint by path; the rest is muted. Opens scrolled to the section; read-only; `y` copies
  it; mouse drag selects and copies as elsewhere.
- Unapplied changes: `esc`, another slot, another tree, add / duplicate, a breadcrumb or `n` ask in the footer
  (`Discard unapplied changes to Matcher?  u discard  esc cancel`); `esc` keeps editing (focus back in the field),
  `u` discards and continues. A clean editor closes at once. Nothing is applied by navigation.
- Trees: tabs in execution order (`1 Tree 1`, a `!` when a slot needs attention), `Add tree`. Tab strip focused:
  `left` / `right` switch, `shift+left` / `shift+right` reorder, Tree Selection (`esc` from the grid; the tab strip is focused): `left` / `right` move the highlight only (nothing
  switches or is discarded), `enter` shows the highlighted tree and goes to its grid (asks first with unapplied editor
  changes), `1`-`9` jump to that tree (numbers `[n]` are shown on the tabs; later tabs have none), `a` opens Tree Actions
  (also the `Actions` button: Rename, Duplicate, Move earlier / later, Add; Remove only when opened from Tree Selection),
  `x` removes the highlighted tree (the footer asks, `y` confirms, `esc` cancels; the last tree stays; Tree Selection
  stays on a surviving tree), `shift+left` / `shift+right` reorder it, `d` duplicates (applied configuration only,
  new id, deep copy), `n` adds (from the template). Removal keys are inactive outside Tree Selection; `delete` still works there. `F2` renames anywhere (1-40
  printable characters, unique, case-insensitive); `1`-`9` show tree N. Stable ids (`Tree.id`) keep runs apart;
  names are display only.
- Template of a new tree: the model config of an experiment spec that trains on the confirmed training file, else
  `configs/xmr4el_base_config.json`; a `features: "tfidf"` config is refused with a footer note, never converted.
  Changing the dataset never rewrites the trees.
- Grid: blocks of name, installed component and short settings (3, 2 or 1 rows as the height allows; scrolls in both
  directions at 80x24); the cursor block is accent when the grid is focused, dim accent otherwise; an invalid slot
  says `needs attention` in the error colour. `left` / `right` follow a branch (a branch end goes on to Hierarchy,
  Hierarchy goes back to the last feature slot), `up` / `down` change branch; `enter` opens the library.


### 5.6 New Experiment: Review Batch

Final inspection before launch; read-only except the batch name and the execution order. The `Review Batch` button of the Model Builder's top
panel (Tab to it, enter) opens it. In Review, enter on the queue reads the details, `e` edits the tree (the Model Builder
banner says `EDITING TREE`), reordering = space picks a tree up, up / down carry it, space drops it, esc restores the old order (alt+left / alt+right
move one place; the mouse buttons are Reorder / Drop, Details, Edit),
esc leaving a screen needs a second press,
left / right switch Overview / Differences, Tab moves between panels. Compact header (batch name `F2`, shared dataset, training file,
enabled splits, tree count, "On tree failure: continue", readiness) over Execution Queue 30 % | Review Details 70 %
(Overview, Differences; `1` / `2` or click; no Checks tab). Code: `review.py` (pure: `Issue`, `validate`,
`diff`, `plan`, `run_spec`), `screens/review_batch.py`, `screens/review_details.py`; the draft gained `name`,
`reference`, `Tree.seq`.

- **Draft:** the batch has a display `name` (`rename_batch`, 60 chars) apart from its identity `id`; trees keep
  their ids; `[` / `]` or Up / Down buttons call `draft.move` (the objects change place, nothing is copied).
- **Reference (Differences):** the first-created surviving tree (`Tree.seq`), by id: reordering and renaming never
  change it; removal falls back to the next oldest and the footer says so. Not the Results baseline.
- **Differences:** `review.effective` = every Model Builder field with the backend default where the config leaves it
  out, plus any other key as written (missing stays distinct from null); compared by path (numbers by value,
  booleans only as booleans, lists element-wise), grouped Features / Hierarchy / Matcher / Search / Other.
- **Validation (automatic):** trees = `components.problems` + tree name + unknown top-level keys (`XModel(**cfg)`
  would refuse them) + duplicate-configuration Caution; dataset = `datasets.validate` on the confirmed selection plus
  runner split-name check, run once in a worker (Validating until it lands) and shown once under SHARED DATASET.
  Ready / Caution (never blocks) / Invalid. No hashing or parsing of whole files: provenance stays with the runner's
  `check_inputs`, so Ready is preflight only.
- **Fix Issue** (enter on an issue row): `App.open_builder(issue)` / `open_setup(issue)` push the existing editor
  over Review Batch (it stays in the stack, so selection, view, scroll and focus return as left). Model Builder: tree,
  slot, editor opened for the installed component, field focused, error under it, JSON line of the path emphasised;
  slot without a supported component: its library; name issue: the rename box. Dataset Setup (`fix=`): the
  confirmed dataset, the mapping focused, normal picker and Confirm Dataset; `m` or esc returns.
- **Launch Batch** (§5.7) is enabled when nothing blocks; it reads `review.plan(draft)` (copy of id, name, dataset,
  ordered trees with applied configs) and `review.run_spec(plan, tree, model_config_path)` (runner spec, training
  only; run name = `<batch id>_<tree id>`).


### 5.7 Batch execution and Batch Monitor

Launch Batch (Review Batch, `l` or the button) first asks in the footer ("Launch N trees?", `l` confirms, esc
cancels), then starts a **detached coordinator**; the TUI never owns a run.

- **Launch.** Tree and name issues are rechecked, then a worker re-validates the dataset and calls `batch.launch`
  (`review.plan` copy of the draft, id = launch time + draft id). Hashing the inputs and starting the coordinator never
  block the UI; the header says "Preparing the batch…" and a second activation is refused. Failure: footer error, the
  review screen and draft untouched, batch directory left `failed` (diagnosable) or removed (`prepare` failed).
  Success ends the experiment (`app.batch_launched`): the New Experiment screens close (kept in `draft.resume`
  for `c`), the Batch Monitor opens over the Dashboard, and the next `n` starts a new draft (`draft.launched`).
- **Files** `outputs/batches/<id>/`: `batch.json` (immutable plan: dataset, input sha256, git commit/dirty, python,
  per tree the runner spec and model-config sha; written once with `x`; its sha256 is in status.json and re-checked by
  the coordinator), `status.json` (mutable, atomic; only the coordinator writes it, plus `mark_lost` for a dead one),
  `stop` (the stop request), `coordinator.log`, `coordinator.lock`, `configs/<tree id>.json` (what each run reads).
  Runs are ordinary `outputs/runs/<timestamp>_<batch id>_<tree id>/` runs, so Dashboard and Results need nothing new.
- **Coordinator** `python -m xmr4el_tui.coordinator <dir>` (stdlib, `start_new_session`, stdio to the log). One tree
  at a time through `process.spawn` (own session/group, launch record, `-model_config_sha256`). Policy
  `continue_on_tree_failure`, no retries. Before each tree the shared inputs are re-hashed; a change, a `batch.json`
  that no longer matches its recorded hash, a stop, or an unwritable status are batch-wide: unstarted trees become
  `cancelled`, finished runs are kept. States: batch preparing / running / stopping / completed /
  completed_with_failures / stopped / blocked / failed / lost; tree pending / running / succeeded / failed /
  cancelled / interrupted.
- **Locks (flock).** `coordinator.lock` (one coordinator per batch; its holder is alive, so liveness cannot be a reused
  PID) and `outputs/batches/.active.lock` (V1: one running batch; it names the holder). Both are held for the
  coordinator's life, opened non-inheritable, and runners are started with `close_fds`, so a runner that outlives a
  killed coordinator holds neither. The coordinator's `procinfo` identity (macOS and Linux) is recorded in status.json.
- **Stop.** The TUI only creates the `stop` file (idempotent). The coordinator sends SIGTERM to its runner's group (it
  is its own unreaped child leading that group; the group is verified before each signal), SIGKILL after `GRACE`
  (30 s) and records the forced stop like `process.record_forced`, cancels the rest and writes `stopped`.
- **Lost.** A scan finding a recorded-active batch whose lock is free marks it `lost` (a `preparing` one only after
  60 s): the running tree becomes `interrupted`, pending trees stay pending, nothing is relaunched or signalled.
  A runner that survived is left running and reported through its own run.
- **Batch monitor** (`screens/batch_monitor.py`): Batch overview + Execution queue (35 %), selected tree's state and
  the existing LogPanel on its run (65 %); `o` / enter opens the tree's run in the ordinary Run Monitor (pipeline,
  resources, Cancel Run); `s` Stop Batch (dialog, Keep running is the default). Status is read every second from
  status.json; run state/stage comes from the 5 s scan. `b` on the Dashboard lists every batch; running batches are
  also a row of Active runs.
- **Quit** with an active batch (read from the files): dialog Leave batch running / Stop batch and exit / Stay in
  XMR4EL (default, esc, a second ctrl+c). Stop-and-exit exits only after every stop request was written.
- **Notifications** (`Notifications.observe_batches`): batch started, tree started/completed/failed/interrupted, batch
  ended; first scan silent, each transition once.
- **Test seam.** `XMR4EL_TEST_RUNNER` (read only by the coordinator, set by `batch.start(runner=...)`) replaces
  `scripts/run_experiment.py` in tests; no user-facing option sets it.
- **Not built:** concurrent trees/batches, retry, resume of a lost batch, per-tree Cancel inside the batch monitor
  (use the tree's Run Monitor), Results grouping by batch.


## 6. View states

Each screen's empty, loading and error states are in its section (§5).

## 7. Keyboard

| Where | Key | Action |
| --- | --- | --- |
| Everywhere | `?` (`F1` inside text fields) | Help for this screen and the focused field. |
| Everywhere | `ctrl+p` | Command palette (every action below is also a command). On the Dashboard: `Palette: <name>` (§4). |
| Everywhere | `esc` | Close overlay, clear filter, then go back. |
| Everywhere | `ctrl+c` (also `ctrl+q`), then `ctrl+c` again / `esc` | Quit after the footer confirmation (no timer) / cancel. Running children keep running (§2). `ctrl+c` no longer copies (`super+c` does). |
| Everywhere | `ctrl+n` | Notification center (§4): `↑↓`, `enter` opens, `c` clears all, `esc` / `ctrl+n` closes. |
| Dashboard | `tab` / `shift+tab` (also `ctrl+tab` / `ctrl+shift+tab`, `]` / `[`); the Explorer moves with the arrows | Next / previous panel: Explorer, Description, Active runs. |
| Explorer | `up` `down`, `tab` `shift+tab` | Previous / next visible item. Shared widget (Dashboard, Results): a tree is two rows, `name · descriptor` (features and depth of its training run's model_config, nothing invented) then `Oct 28 · a64127ef` (training date, short tree id: 8 characters, more where two ids share them; `explorer.short_ids`); a gap row above every later tree / dataset / group, none between evaluations; a click anywhere on an item's rows selects it. |
| Explorer | `left` / `right` | Collapse, or go to the parent / expand. |
| Explorer | `enter` | Describe the item and focus Description. |
| Explorer (Dashboard, Results) | `/`, then text, `backspace`, `enter`, `esc` | Search: every word in a node's name, descriptor, date, id (short or full), dataset, run id, split or features, or in an ancestor's; matches keep their subtree, their ancestors open while the search lasts (the user's expansion returns after). `enter` keeps the query (shown in the border), `esc` clears it. |
| Dashboard, Description | `r`, `left` / `right` / `esc` | Results with the evaluation as the baseline (a tree: its only evaluation with metrics; several: the tree opens to choose) / Run monitor of its run (tree: training run; evaluation: its split's stage selected) / back to Explorer. |
| Results | `up` `down`, `left` / `right`, `space`, `enter` | Explorer: move, collapse / expand, add or remove the evaluation (a tree: expand), inspect. Mouse: click an evaluation adds / removes, its `i` inspects, a tree row expands. |
| Results | `1`-`5`, `ctrl+tab` / `ctrl+shift+tab` (`]` / `[`), `<` / `>`, `esc` | Metric group; next / previous panel (Show incompatible, Explorer, Summary, Analysis); narrower / wider left column; close inspection, else to the Explorer, else back to the Dashboard. |
| Active runs | `up` `down`, `enter` | Move, open the run in the Run monitor. |
| Run monitor | `ctrl+tab` / `ctrl+shift+tab` (or `]` / `[`) | Next / previous panel: Run overview (its Cancel Run button when usable), Pipeline, Resources, Live logs. |
| Run monitor | `up` `down`, `tab` `shift+tab`, `left` / `right`, `enter` (Pipeline) | Move between stages / collapse or parent, expand / toggle a stage's parts. The selected row's resources show in Resources. |
| Run monitor | `c` | Follow the current stage again (moving the cursor stops following). |
| Run monitor | `s`, `Cancel Run` | Cancel Run confirmation (§5.2), only while the run is RUNNING with a verified isolated group, a TUI launch record (§9) and no cancel is pending. |
| Run monitor | `f`, `esc` | The focused panel fills the screen (Live logs: the fullscreen log, same reader, search and filter); `esc` restores, then goes back to the Dashboard with its selection kept. |
| Run monitor | `<` / `>`, drag the left column's right border | Narrower / wider left column by 2 cells: default 35 %, at least 30 cells, at most 60 %, the right side never under 44; kept for the session. |
| Run monitor | Live logs: `/` `n` `N` `l`, scroll keys | As the log viewer. |
| Dashboard | `ctrl+r` | Rescan. |
| Dashboard | `ctrl+b`, `<` / `>`, drag the explorer's right border | Hide / show the explorer; narrower / wider by 2 cells. Default 22 % of the width, at least 18 cells, at most half the screen and never leaving the right panels under 40 (the minimum wins on tiny screens); kept for the session, re-clamped on terminal resize. |
| Description | `up` `down`, click | Select a row (read-only; headings and blank lines are skipped, the list wraps). A rescan of the same item keeps the row with the same field label. |
| Description | `enter` / click on `View Logs  [Open]` | Fullscreen read-only log of the item's run (evaluation: its run, between Run and State; run: between Type and State; tree: its training run, between Training run and Run state). `esc` returns with the row still selected. |
| Log viewer | `↑` `↓` `pgup` `pgdn` `home` `end`, `/` search (enter finds, `esc` closes the box), `n` / `N` next / previous match, `l` level all → INFO+ → WARNING+ → ERROR+, `esc` back | Header: run name, id, state, file, level, match i/n. An active run is polled every second and follows the end while the view is at the end; a finished run is read once. A missing log shows why, nothing is generated. At most 100 000 lines kept, first read ≤ 8 MiB from the end. |
| Explorer, Description, Active runs | mouse drag | Select text (blue); on release it is copied to the clipboard by OSC 52 (`App.copy_to_clipboard`), so it reaches the terminal the TUI runs in, also through `ssh` / `docker exec`; a toast says how many characters. A click without a drag copies nothing; dragging the explorer border resizes, never selects. The terminal must allow OSC 52. |
| Everywhere | `n` | New Experiment: Dataset Setup (§5.4); shown in the Dashboard footer. Already open: back to it; after a launch: a new draft. |
| Dataset Setup | Datasets: `up` `down`, `enter` / click; `ctrl+tab` / `ctrl+shift+tab` (`]` / `[`); `esc` | Move (nothing changes) / configure the dataset; next / previous panel (Datasets, Name, configuration); back to the previous screen (unconfirmed edits stay in the draft). |
| Dataset Setup | Configuration: `tab` `shift+tab` (or `up` `down`), `enter`, `space`, click on `[x]`, `ctrl+s`, `u`, `esc` | Move between controls; a file row opens the picker (training variant: selects it); select a training variant / toggle a split; toggle; Confirm Dataset; discard unconfirmed changes (asks); back to Datasets. `ctrl+r` rereads datasets. |
| Dataset Setup | `m` | Model Builder (§5.5) with the confirmed dataset. |
| Model Builder | grid: arrows, `enter`, click; library: `up` `down`, `enter` / click; `esc` | Move between slots (branches) / open the slot's components; open the component's editor; editor -> library -> grid -> Dataset Setup. |
| Model Builder | editor: `tab` `shift+tab`, `enter` (next field / Apply), `ctrl+s`, `u`, `esc`; preview `y` | Move between attributes; Apply; discard (asks); back to the library (asks with unapplied changes); copy the JSON. |
| Model Builder | `tab` / `shift+tab` | Next / previous control inside the focused panel only (Model Builder: tabs, grid; attributes: fields, help, Apply, Cancel); wraps. |
| Model Builder | `enter` / `esc` | Levels: Tree Selection (tab strip) > grid > library > editor (attributes, `Preview` button for the JSON panel, Apply, Cancel). `enter` goes in, `esc` goes out (editor asks with unapplied changes; esc from Preview returns to the attributes; esc from Tree Selection leaves the screen). No panel-switching key (no ctrl+tab, no F6). |
| Model Builder | attributes: `up` / `down`, `enter` | Previous / next parameter (nothing is applied; its help shows below the list: name, description, JSON path, constraints); `enter` in a text field moves on, on a collapsed dropdown opens it (its own arrows and `esc` then apply). |
| Model Builder | grid: `+` / `-` / `0`, or the zoom bar | Compact / Normal / Expanded density, `0` = Fit to the viewport width (UI state only, kept across trees; the model is centred when it fits, scrolls when not). |
| Model Builder | tabs: `left` / `right`, `shift+left` / `shift+right`, `a`, `d`, `delete`; `F2`, `1`-`9` | Switch / reorder trees, add, duplicate, remove (asks); rename; show tree N. |
| File picker | `up` `down`, typing, `enter`, `esc` | Move; filter or type / paste a path; open the directory or choose the file; cancel. |
| Dashboard | `r`, `b`, `n`, `c` | Results with the selected evaluation as baseline; Batches; New Experiment; resume the unfinished experiment (only while one exists). |
| Review Batch | queue: `up` `down`, `enter`, `e`, `space` then `up` `down` then `space` (`esc` restores), `alt+left` / `alt+right` | Move; read the tree's details; edit the tree in the Model Builder; pick up, carry and drop a tree (reorder); move one place. |
| Review Batch | details: `1` / `2` (or `left` / `right`), `enter` on an issue | Overview / Differences; Fix Issue (opens the editor that fixes it). |
| Review Batch | `F2`, `l`, `tab` `shift+tab`, `esc` | Rename the batch; Launch Batch (asks, `l` confirms); next / previous panel; back (a second `esc` leaves). |
| Batch Monitor | `up` `down`, `enter` / `o`, `s`, `esc` | Move in the execution queue; the tree's run in the Run Monitor; Stop Batch (dialog, Keep running is the default); back to the Dashboard. |
| Batches | `up` `down`, `enter`, `esc` | Move; open the batch's monitor; back. |

- Printable keys never act inside text fields.
- Going back restores the selected row, scroll position and focus.

## 8. Progress model

The runner writes `events.jsonl` in the run directory, one JSON object per line:

```json
{"t": "2026-10-20T13:44:11.153", "stage": "encode", "phase": "start", "rows": 474854}
{"t": "2026-10-20T13:44:11.160", "stage": "encode", "part": "mention_tfidf", "phase": "start"}
{"t": "2026-10-20T13:51:54.612", "stage": "encode", "part": "transformer", "phase": "cache_hit", "distinct": 470409}
{"t": "2026-10-20T13:52:09.650", "stage": "hierarchy", "phase": "start", "layer": 1, "layers": 2}
{"t": "2026-10-20T13:52:09.651", "stage": "node", "phase": "start", "layer": 1, "i": 1, "n": 1, "rows": 474854}
```

A transformer tick, when the cache misses (*illustrative*):

```json
{"t": "2026-10-23T14:17:40.002", "stage": "encode", "part": "transformer", "phase": "tick", "unit": "distinct_mentions", "done": 50000, "total": 100000}
```

Stages, matching the existing `logger.info` calls:

| Stage | Existing log line | Training | Eval-only |
| --- | --- | --- | --- |
| `load` | `Run started` -> `Preparing data` | + | tree load |
| `encode` | `Encoding started/completed`; parts `mention_tfidf`, `context_tfidf`, `transformer`, `assemble` | + | |
| `labels` | `Label embeddings started/completed` | + | |
| `hierarchy`, `layer`, `node` | `Hierarchy`, `Layer`, `Node started/completed` | + | |
| `save` | `Model saved` | + | |
| `eval` (per split) | `Prediction encoding` ... `Eval <split>` | + | + |

- The events are `extra=` fields on those existing log calls, written by one handler
  in the runner. Log wording is never parsed and can change freely.
- Encode ticks go to a separate logger (`xmr4el.progress`, propagation off) at most
  every 5 s, so `run.log` is unchanged.
- Part start/complete events come from the encoder: `_tfidf_block` (prefix `""` ->
  `mention_tfidf`, `"context_"` -> `context_tfidf`), the transformer call, and the
  normalise/concatenate step (`assemble`). Prediction encoding emits no part events.
- Emit the final transformer count even if the last batch finishes within 5 s of
  the previous tick. Cache hits emit a `cache_hit` event; TF-IDF-only paths emit no
  transformer events. No part completes the parent Encode stage; its completed event
  follows `assemble`.
- Without `events.jsonl` (runs from before this change), the Run Monitor's pipeline says the stage list is
  unknown; the log works as usual.

**Schema v1 (`xmr4el/telemetry/events.py`).** Every line also carries `"v": 1`; `t` is
local time with its UTC offset (older lines: naive local time, still read). Events stay
`stage` + `phase` (no second vocabulary): `phase` is `start`, `complete`, `failed`, `stopped`,
`tick` or `cache_hit`, so stage_started = `start`, stage_completed = `complete`,
stage_failed / stage_cancelled = `failed` / `stopped`, stage_progress = `tick`.

- `stage: "run"` brackets the whole run: `start` (with `run_id`, `kind` train/eval,
  `splits`), then `complete`, `failed` (`error`) or `stopped` (`signal`).
- Every end phase of a started stage or part has `duration_seconds`, from `time.monotonic()`
  at its start (the reader prefers it to the `t` difference).
- On failure or a stop, the runner closes every stage and part still open (innermost first)
  with `failed` + `error` / `stopped` + `signal`, then the run. A SIGKILLed run has none: the
  reader shows its open stage failed (LOST) or as status.json says.
- Eval parts (per split): `data` (read + vocabulary filter), `predict`, `metrics` (from
  `evaluate_tree(..., events={"split": s})`), `write` (metrics_<split>.json, by the runner).
- `layer` / `node` stay start-only detail events (hierarchy = clustering and matcher training
  per node, one pass: no separate clustering / matcher stages exist).
- Writes: one `write` + `flush` per line (no fsync); readers skip a torn last line
  (`progress.read_events`). Event rate is bounded by stage boundaries, nodes and ticks (≤ 1 per 5 s).

**resources.json** (`xmr4el/telemetry/resources.py`; `-resource_interval` seconds, default 5,
`0` = off). One summary per top-level stage (`load`, `encode`, `labels`, `hierarchy`, `save`,
`eval` + `split`), written (write-then-rename) when the stage ends; raw samples are not kept:

```json
{"v": 1, "device": "cuda", "interval_seconds": 5, "scope": {...},
 "stages": [{"stage": "encode", "status": "completed", "duration_seconds": 123.4, "samples": 26,
             "resources": {"cpu_avg_percent": 65.2, "rss_peak_bytes": 10737418240, "rss_avg_bytes": 9663676416,
                           "gpu_util_avg_percent": 81.5, "torch_gpu_allocated_peak_bytes": 8589934592}}]}
```

| Field | Scope, how |
| --- | --- |
| `cpu_avg_percent` | this process (all threads) + exited children, CPU time / wall time (os.times, exact); 100 = one core |
| `rss_peak_bytes`, `rss_avg_bytes` | this process, sampled (Linux `/proc/self/statm`, macOS `proc_pidinfo`); the peak is exact when the stage raised the lifetime peak (`getrusage`) |
| `gpu_util_avg_percent` | CUDA only, whole device incl. other processes, sampled (NVML via `torch.cuda.utilization(index)`; an index, so the sampler never initialises CUDA); null without the optional `nvml` extra (`nvidia-ml-py`, in the Docker image) or an NVML driver library |
| `torch_gpu_allocated_peak_bytes` | PyTorch tensors of this process on its GPU (the caching allocator's *allocated* bytes). CUDA: `torch.cuda.max_memory_allocated`, exact, reset at stage start only if CUDA is already initialised (never creates a context; a stage that initialises CUDA counts from then). MPS: `torch.mps.current_allocated_memory`, sampled maximum. Not total process GPU memory: excludes the allocator's reserved-but-free cache, the CUDA context (~0.3–1 GB) and non-PyTorch allocations; not device memory as `nvidia-smi` shows it. Null on CPU or while CUDA is uninitialised |

Unavailable = `null`, never estimated. Stage start and end sample too (≥ 2 samples per stage).
Measured on a 0.6 s synthetic train + eval (5 runs each, median): no telemetry 0.587 s,
events + run.log + resources 0.589 s, sampler at 0.1 s 0.586 s, plus a 1 Hz log poller process
0.599 s; one sample ≈ 1 µs, one event line ≈ 6 µs; files 3.1 KB events, 1.9 KB resources.

**Logs.** `run.log` (format `%(asctime)s [%(levelname)s] %(name)s: %(message)s`, INFO from xmr4el,
WARNING from libraries, Python warnings via `logging.captureWarnings`, tracebacks of failures) is
the log the viewer shows; TUI launches also capture stdout/stderr in `outputs/runs/.console/<id>.log`,
shown when `run.log` does not exist. `xmr4el_tui/logs.py` `LogTail` reads from a byte offset
(torn last line held back, truncation / replacement restart, at most 8 MiB per read, so the
first read of a large log starts near its end); continuation lines inherit their record's level.

## 9. Ownership, liveness, stopping

`status.json` while running gains `pid`, `host` (`socket.gethostname()`), `uid`, an OS
process-start token, runtime identity (host boot and, on Linux, PID namespace),
`pgid`, `sid`, and `isolated_group`. Record actual process properties; a hostname or
a matching command line alone is not proof of ownership. Keep `started` and the
identity fields in the final record. The runner touches a `heartbeat` file every
30 s (separate file, so it never races the final status write).

| Recorded | Seen from | Check | Shown |
| --- | --- | --- | --- |
| running | same runtime | PID, UID and process-start token match | RUNNING |
| running | same runtime | process absent or start token differs | LOST; recorded process is gone |
| running | other or unknown runtime | identity cannot be checked | UNVERIFIED, heartbeat age when available |
| running | same runtime | inspection denied or incomplete identity | UNVERIFIED; stopping unavailable |

- TUI children use `start_new_session=True`. The runner sets `isolated_group` only
  after verifying `pid == pgid == sid`; ordinary CLI launches may share a shell's
  group and must not be assumed isolated.
- Stop is offered only for RUNNING with a verified isolated group **and** a TUI launch record:
  `process.spawn` writes `outputs/runs/.console/<id>.launch.json` (run id, canonical run dir, child pid,
  start token, UID, runtime); `process.owned` requires all of them to match the recorded identity. An
  isolated group is no ownership: a shell run under `setsid` is monitored, never cancelled. `stop` also
  refuses when status.json's identity changed since the run was read. Before SIGTERM
  and again before SIGKILL, recheck process-start identity, runtime, UID, PGID and
  SID. On mismatch, denial, or process exit, send nothing and refresh the state.
  Never fall back to signalling a shell's group. Command arguments corroborate
  identity; substring matching a run path does not establish it.
- A reattached TUI in the same runtime can stop a verified isolated run started by
  an earlier TUI (the launch record persists). A verified non-isolated CLI run stays RUNNING with
  `Stop unavailable: process group is shared`, an isolated one without a launch record with
  `not launched by the TUI (monitor only)`; use its launching terminal to stop it.
- The runner writes its final status last: open stages closed, resources.json and the final log
  record first, so a reader that sees a final status has every artifact. The monitor and log panel
  coalesce a tick that arrives during a read and stop polling only after a read that started after
  the run ended.
- `procps` is already in the image; `ps` exists on macOS.

**Cancellation.** `process.stop(run, force=False)` is the one service a screen calls
(confirmation is the screen's job, §5.2). It re-reads `status.json` first: a run that already
recorded `completed` / `failed` / `stopped` is not signalled. Graceful first: SIGTERM to the group;
SIGKILL only on an explicit `force=True`. A SIGKILLed runner writes nothing, so the supervisor records a
forced stop (`process.record_forced`): it waits up to 10 s until the recorded process is gone (absent,
a zombie, or its pid reused: `procinfo.identity` treats zombies as gone), then, if `status.json` is still
`running`, rewrites it (write-then-rename, identity kept) as `stopped`, `error: "SIGKILL"`, `forced: true`;
nothing can race that write, the writer is dead. A final status the runner wrote first is kept. If the
process is not gone in time, nothing is written (the call says so). An unrequested SIGKILL (OOM killer,
`kill -9`) never passes there: that run stays `running` and shows LOST. `force=True` blocks up to 10 s:
call it from a worker. No events are appended for a forced stop: status.json is the record and the
stage reader shows the open stage `stopped` from it. Runner side:
the first SIGTERM raises `Stopped` -> status `stopped` (error `SIGTERM`), open stages and the run
end `stopped` in events.jsonl, completed splits' metrics stay; a repeated SIGTERM, or one that
arrives after the final status is decided (stop racing completion), raises nothing.

| State | Recorded | Meaning |
| --- | --- | --- |
| COMPLETED | completed | every eval written |
| FAILED | failed | an exception; `error` in status.json and on the failed stage |
| STOPPED (cancelled) | stopped | SIGTERM/SIGINT (runner), or `error: SIGKILL` + `forced: true` (verified force stop, supervisor) |
| LOST (interrupted) | running | process gone without a final status (SIGKILL, OOM, power) |
| RUNNING / UNVERIFIED | running | live identity verified / not checkable |
| UNKNOWN | missing | no readable status.json (old or partial runs) |

## 10. Technical design

**Package.** A separate top-level package `xmr4el_tui/`, optional extra `xmr4el[tui]` (Textual, pinned in
`uv.lock`), entry point `xmr4el-tui`, `app.tcss` as package data. It imports nothing from `xmr4el`:
`xmr4el/__init__` imports torch (about 1 s warm), which a run browser must not pay. The `xmr4el` image adds
`--extra tui` to its dependency sync; because it uses `--no-install-project`, launch the bind-mounted package with
`python3 -m xmr4el_tui`. Module layout: [tui_architecture.md](tui_architecture.md) §2.

**Runner interface** (`scripts/run_experiment.py`):

| Flag / file | Use |
| --- | --- |
| `-spec -` | Spec on stdin (the coordinator passes each tree's spec; no throwaway specs in `configs/experiments/`). |
| `-run_dir PATH` (must not exist) | The coordinator knows the run directory before the child starts. |
| `-model_config_sha256 SHA` | Binds the run to the batch's model config (`outputs/batches/<id>/configs/<tree>.json`): a mismatch exits 3 before any run directory exists, with one JSON line on stdout (`{"error": "model_config_changed", "expected": …, "actual": …}`); the coordinator records the tree `failed` (`not started: model_config_changed`). |
| `-xmodel_path`, `-tree_id ID` | Eval-only runs (CLI); `tree_changed` is the matching exit-3 error. |
| `-check` | Validate only, one JSON object on stdout. Kept for the CLI and tests; the TUI does not call it (Review Batch validates with `review.validate`; input hashes are the runner's `check_inputs`). |
| `status.json` identity, `stopped` on SIGTERM / Ctrl+C, `heartbeat` every 30 s | §9. |
| `events.jsonl`, `resources.json` | §8. |
| `population_sha256`, `strings_sha256`, `eval_protocol`, preprocessing in `metrics_<split>.json` | §11. |

The runner reads the model-config bytes once: hash, validation, `model_config.json` and the model all come from
that captured content (`XModel(**json.loads(raw))`), so the source changing mid-run cannot change the run.

**Spawning** (`process.spawn`, called by the coordinator): `sys.executable scripts/run_experiment.py -spec -
-run_dir … -model_config_sha256 …`, argument list, no shell, working directory = `--root`,
`start_new_session=True`, stdout and stderr to `outputs/runs/.console/<run-id>.log` (created exclusively; a dot
directory, ignored by the scan), plus the launch record `.console/<run-id>.launch.json` (§9). The console log keeps
errors raised before `run.log` exists.

**Run data.** Everything comes from the run directory: `status.json`, `spec.json`, `tree.json`, `metrics_*.json`,
`model_config.json`, `provenance.json`, `git.json`, `events.jsonl`, `resources.json`, `heartbeat`, `run.log`. A tree
is "on this machine" when `<xmodel_path>/xmodel.pkl` exists. Pickles are never loaded. Recorded `spec.json` adds
`xmodel_path` (invocation metadata, not an input key; null = training).

## 11. Comparison rules

Results decides comparability in one place, `services/comparison.assess(baseline, challenger)`; the evidence each
recorded hash provides, and which deltas it gates, is the table in [tui_architecture.md](tui_architecture.md) §3a.
The rules behind it:

- Every check and metric lookup applies to the selected evaluation (run id + split), never to a run's first one.
- Same test-file sha256 (`provenance.json`) and the same `rows_total` before any delta; otherwise INCOMPATIBLE.
- Every delta also needs one recorded `eval_protocol`: population hashes say nothing about how a metric was
  computed. Missing: CAUTION, no deltas; different: a note, no deltas.
- In-vocabulary metrics (acc@1, MRR, recall, R@cand, hybrid) need equal `population_sha256` (ordered retained
  row positions and gold labels). Equal row counts or coverage do not establish equal populations. Without the
  hash, populations count as equal by derivation only when all hold: same evaluation-file sha256; same
  training-file sha256 and preprocessing (`context_window`, `abbrev_expansion`) for each tree, read from the run
  that trained it (found by `tree_id`); every involved run at one git commit with `dirty: false`. Any input missing
  (e.g. the training run not synced): CAUTION, no subset deltas. The browser never reads a corpus or loads a
  pickle to decide.
- `acc@1_all` counts every loaded row (uncovered labels are misses), so it keeps its delta when only the
  vocabulary differs; it is the fixed comparison metric in the Summary.
- Generalization (seen / unseen) needs equal `strings_sha256` (the effective dictionary per evaluated row: the
  tree's vocabulary and abbreviation expansion change it, equal reference files are not enough).
- No timing deltas: hardware is not recorded. Vocabulary, beam and search settings never block; they show in Specs.

## 12. Tests

Offline and synthetic; map in [tui_architecture.md](tui_architecture.md) §8. Screens: Textual `run_test()` +
Pilot at 80x24 and larger, `NO_COLOR`. Runner and telemetry: `tests/test_run_experiment.py`,
`tests/test_telemetry.py`. Manual checks still open (SSH drop and reattach through `docker exec`, color inside the
container with and without `-e TERM -e COLORTERM`, Linux): [Step 6 audit](../audits/new_experiment_step6.md).

## 13. Out of scope

Charts, dataset browsing, tree visualisation, artifact deletion, cache management, a prediction playground,
eval-only launches, concurrent trees or batches, retries, resuming a lost batch, persistent drafts.
