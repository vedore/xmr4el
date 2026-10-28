# XMR4EL TUI design

Status (2026-10-27): only the Dashboard is built (§7 Dashboard rows); the screens of §5 were removed and are to be
rebuilt on the same backend. Earlier status: proposed (2026-10-23). Wireframe values come from the ten
runs in `outputs/runs/` (2026-10-20) unless marked *illustrative*.

## 1. Concept

An experiment workbench over `outputs/runs/`. Two places only:

- **Runs:** every run, newest first, with its tree and its score.
- **Run:** one run in any state: Overview, Log, Inputs.

Setup, Review and Compare are full-screen states reached from those two. The TUI
reads run directories and spawns `scripts/run_experiment.py`; it holds no training or
evaluation logic. Run files are the only durable state, so a run outlives the TUI that
started it and any TUI can show it.

## 2. Where it runs

| Place | Typical use | What works |
| --- | --- | --- |
| Server, in the `xmr4el` container, over SSH | Launch, watch, stop, evaluate trees | Everything. |
| Laptop with synced `outputs/runs/` | Browse, compare, read logs | Everything that needs no tree; trees (several GB each) are usually not synced. |

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

Run children are started in their own session (`start_new_session=True`) with stdout
and stderr written to a file, never a pipe. An SSH drop or a TUI crash therefore
does not signal or block the run. Reattaching is running the `docker exec` line again:
the TUI rebuilds everything from the run directory. tmux is not required.

### Laptop

Sync only run records, e.g. `rsync -a server:xmr4el/outputs/runs/ outputs/runs/`.
Trees are absent, so tree actions show their reason instead of failing:

```text
tree f15e98b4   not on this machine
  missing: outputs/saved_trees/xmodel_2026-10-20_15-02-58/xmodel.pkl
```

A run recorded as running on another host shows **UNVERIFIED** with the heartbeat
age (§9); the laptop never signals anything.

## 3. Workflows

| Task | Keys |
| --- | --- |
| Train and evaluate | Runs `n` -> Setup -> Review -> `enter` start -> Run |
| Evaluate a tree again (other split, other search) | Runs `t` -> Trees `e` -> Setup (tree prefilled) -> Review -> start |
| Repeat with edits | Run `d` duplicate -> Setup (copied fields) -> Review -> start |
| Watch a run | Runs `enter` on a RUNNING row -> Overview |
| Recover from a failure | Run (banner shows error) -> `2` log -> `d` duplicate, edit, start |
| Compare | Runs `space` on two rows -> `c` -> select an evaluation on each side |

Duplicate copies setup, never execution state. There is no training resume.

Commands outside the TUI keep their CLI: grouped-TSV training (`train.py`), dataset
conversion and splitting, `diagnose_routing.py` / `diagnose_unseen.py`, rerankers and
external baselines. Run Inputs show the equivalent `run_experiment.py` command line
and a valid input specification so a run can be repeated outside the TUI (§10).

## 4. Visual language

**Layout.** One header line, one footer line, content between. Blank lines separate
groups. The Runs and Trees details block is a 40-column panel on the right (under the table below
100 columns); header and footer span the full width. Help (`?`) replaces the panel and `esc`
closes it. No boxes around sections. Scrollbars are one cell wide and muted.
`ctrl+c` / `ctrl+q` asks in the footer (`Confirm  Quit XMR4EL?`), a second `ctrl+c` quits, `esc` cancels; no
timer. Runs keep running after a quit (TUI launches are detached: own session, output to files), and the question
says so when runs are active. `ctrl+r` says the run count in the footer.

**Navbar (2026-10-28, replaces the 2026-10-27 areas).** Breadcrumbs left (`Dashboard / Results / Inspect`), `XMR4EL`
centred on the app width (never moved by the crumbs, the bell or the help panel, which docks under the navbar and
above the footer), `Notifications` right with the unseen count (`Notif` when that does not fit; text: no emoji anywhere in the TUI); one blank row under the
navbar separates it from the panels. Crumbs come from
the screen stack (each screen's `crumbs`; modal screens add none): no second navigation history. Focus, panels and
metric groups never change them. A click on an earlier crumb goes back to it (screens above close as with `esc`;
`Results` while inspecting closes the inspection); too long, they are cut on the left (`… / Inspect`). No host,
counts or other summaries (they live in the panels).

**Three message layers (2026-10-28).** (1) *Notifications*: experiment lifecycle only, in memory for the session
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
challengers, incompatible evaluation, missing run, copy, rescan) left of the shortcuts for 3.5 s; a newer one
replaces it; it belongs to its screen (navigation drops it); never kept as a notification. (3) *Footer confirmation*:
quit; replaces the shortcuts until answered; feedback and notifications never replace it. Comparison Caution stays
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
The Runs table's State column uses short words so every column fits 80 wide: RUN, UNVER,
LOST, DONE, FAIL, STOP, `?` (same colours; the filter matches both forms). Headers and
details use the full word.

**Log lines.** The Log tab shows `HH:MM:SS L logger message`. `L` is the level's first
letter (W yellow, E/C red). The `xmr4el.` package path is dropped, leaving the last logger
component. Lines in other formats are shown unchanged; `run.log` itself is not rewritten.

**Formats.**
- Timestamps: `MM-DD HH:MM` in tables, `YYYY-MM-DD HH:MM` in details.
- Durations: `m:ss` under an hour, `h:mm:ss` above. Everywhere, including the header.
- Metrics: 4 decimals; a percent only for coverage (`96.3%`).
- Counts: thousands separators. Latency: `2.99 ms/query`.
- Tree IDs: first 8 hex characters; full ID in Inputs.

**Stage markers.** Plain text (no symbol glyphs): `+` done, `>` current, `·` pending, `x` failed.
ASCII (`NO_COLOR` or a non-UTF-8 locale): `ok`, `>>`, `..`, `xx`.

**Style (2026-10-27; tokens 2026-10-28).** One high-contrast theme `xmr4el`; no theme picker (the palette's
Theme command is removed). Every colour is a semantic token in `xmr4el_tui/theme.py` (`DARK`, the only file with
hex values; a test enforces it): application (background `#101418`, surface `#1B2229`, panel border, text
`#F0F4F8` / muted `#A5B4C3` / disabled), selection (accent `#2CF5C5`, `on-accent`, `cursor-row` = 40 % accent over
surface, `focus-outline`), status (success, warning, error, info, unknown, `warning-on-accent` for Caution on a
selected row), comparison (baseline / member rows, delta) and the beam palette. `app.THEME` maps them to Textual's
variables and exposes every token as `$xmr-<token>` in app.tcss; Python Rich styles use `theme.color` /
`theme.style`. A later theme = another dict with the same keys assigned to `theme.TOKENS`. The cursor row is
painted across the whole row (explorer guides, every table column, the padding after them); a blurred panel keeps
a dim accent cursor (selected, not focused); the cursor row of Description is one colour (a green COMPLETED
would vanish on the accent); explorer guides of the cursor's branch stay accent (on a leaf or a collapsed node,
its parent's branch); the focused panel has the accent border and title. Status colours
stay semantic (`dashboard.STATE_STYLE`). Textual honours `NO_COLOR`; that is the monochrome mode.
No animation except the encoding progress bar.

## 5. Wireframes

### 5.1 Runs (80 columns)

```text
XMR4EL  Runs  10                                    gpu01-xmr4el · 1 running
  Started      Run                   Kind   Tree      Split   acc@1  State
> 10-23 14:10  bc5cdr_chemical       train  —         dev         —  RUN
  10-20 15:04  nlm_chem_test         eval   f15e98b4  test   0.8423  DONE
  10-20 14:41  nlm_chem              train  f15e98b4  dev    0.8753  DONE
  10-20 14:37  biored_chemical_test  eval   e1348683  test   0.9121  DONE
  10-20 14:14  biored_chemical       train  e1348683  dev    0.8703  DONE
  10-20 14:05  bc5cdr_chemical_test  eval   906338e3  test   0.9303  DONE
  10-20 13:44  bc5cdr_chemical       train  906338e3  dev    0.9463  DONE
  10-20 13:42  biored_disease_test   eval   dd438500  test   0.8816  DONE
  10-20 13:40  biored_disease        train  dd438500  dev    0.8828  DONE
  10-20 13:38  ncbi_disease_test     eval   8528b624  test   0.8960  DONE
───────────────────────────────────────────────────────────────────────────────
 bc5cdr_chemical · train · started 2026-10-23 14:10 · 14:38 so far
 stage  Hierarchy layer 1/2 · previous bc5cdr_chemical took 21:15
enter open  n new  d duplicate  space mark 0/2  / filter  ? help
```

Details for a completed row:

```text
───────────────────────────────────────────────────────────────────────────────
 nlm_chem · train · 2026-10-20 14:41 · 23:16 · sapbert_char_context
 tree f15e98b4 · 179,696 labels · not on this machine · also: nlm_chem_test
 dev  acc@1 0.8753 · acc@1_all 0.8686 · 4,907 / 4,945 rows in vocabulary
```

- **Kind:** `train` (trained a tree) or `eval` (evaluated an existing tree).
- **Tree:** `—` until a training run has saved its tree.
- **Split / acc@1:** the run's only evaluation. With more than one, Split shows
  `dev +1` and acc@1 the first evaluation's value; details list all of them.
  acc@1 is in-vocabulary accuracy, the same number as `metrics_<split>.json`.
  acc@1_all is in the details.
- Rows on the selected row's tree show their Tree cell in bold. The details line
  `also:` names them, so this does not rely on bold alone.
- A FAILED or STOPPED run that wrote metrics shows its acc@1. A missing value is `—`.
- The footer lists only actions the selected row allows: `e evaluate tree` appears
  when the row has a saved tree on this machine, so not for a running training row.
- Sorting: newest first. `o` cycles Started / Run / acc@1. Sorting by acc@1 groups
  by Split first, so dev and test scores are never ranked together.

Column priority as width shrinks: Duration (only at >= 100 columns), then Started,
then Kind. Run, Tree, Split, acc@1 and State always stay.

### 5.2 Run: running

```text
XMR4EL  bc5cdr_chemical                     RUNNING 14:38 · gpu01-xmr4el
 Overview   Log   Inputs
 ========
 + Load corpus                                   0:06
 + Encode  474,854 rows                          7:55
 + Label embeddings  179,709 labels              0:04
 > Hierarchy  layer 1/2 · node 1/1 · 474,854 rows        6:33
 · Save tree
 · Evaluate dev

 Previous bc5cdr_chemical (10-20): 21:15 total · hierarchy 11:43
 Last event 6:33 ago. The root node logs nothing until it is fitted.

 1 warning  13:55:02 JointOvRLogistic: cuda out of memory (n=…), fitting on CPU
s stop  w warnings  2 log  esc runs  ? help
```

Encode is shown as its encoder blocks, in the order the encoder fits them. In the
real `bc5cdr_chemical` run, 7:43 of the 7:55 went to the two TF-IDF -> SVD blocks;
the transformer was a cache hit. The blocks are therefore the main progress signal;
the transformer bar counts **distinct mention strings**, not corpus rows, and only
appears when the transformer actually runs. This example is *illustrative*:

```text
 > Encode  474,854 corpus rows                            5:12
   + mention TF-IDF -> SVD                                3:58
   + context TF-IDF -> SVD                                0:57
   > transformer  50,000 / 100,000 distinct mentions  ████████░░░░░░░░  50%
   · assemble features
```

- Repeated mentions are encoded once, then expanded back to corpus rows. Successful
  transformer batches advance this substep only; OOM retries do not advance it.
- Each block line has its own duration. A TF-IDF -> SVD block is a single fit with
  no inner progress; its line shows elapsed time only.
- A cache hit shows `+ transformer  cache hit`, without a batch percentage.
  `"features": "tfidf"` runs show one `TF-IDF -> SVD` block and no transformer line.
- `assemble features` (normalise and concatenate the blocks) ends with the existing
  Encoding completed event, which completes the parent Encode stage.
- Times on the right are stage durations; the header has the total.
- Previous timings come from the newest COMPLETED run with the same Run name. No
  ETA is computed; the previous run is the honest reference.
- "Last event" appears only after 60 s without an event. The explanatory sentence
  appears when the current stage is a single large node, where silence is normal.
- Warnings stay listed until the run ends; `w` opens the log filtered to warnings.
- An evaluation that finished shows its acc@1 on its stage line immediately:
  `+ Evaluate dev   acc@1 0.9463   0:32`.

### 5.3 Run: completed

```text
XMR4EL  ncbi_disease                                   COMPLETED 2:07
 Overview   Log   Inputs
 ========
 Tree   8528b624 · trained here · not on this machine · also: ncbi_disease_test
 Took   encode 1:18 · hierarchy 0:43 · save 0:02 · evaluate dev 0:03

 Evaluation  [ dev v ]  datasets/ncbi_disease/dev.pubtator
 Search      beam_size 10 · leaf_candidates 100 · topk 0 · knn_beta 10
 Rows        729 / 757 have a gold label in the tree (96.3%) · 13,385 labels

 acc@1      0.9246    acc@1_all   0.8904    MRR       0.9459
 R@5        0.9671    R@10        0.9739    R@20      0.9739
 R@50       0.9822    R@100       0.9835    R@cand    0.9835
 hybrid     0.9287    hybrid_all  0.8943    2.99 ms/query, batch incl. encoding

 Mention strings    n     tree    dict   either  routed
 all              729   0.9246  0.7394  0.9287  0.9835
 seen, 1 label    537   0.9683  0.9721  0.9721  1.0000
 seen, >1 label    17   0.9412  1.0000  1.0000  1.0000
 unseen string    175   0.7886  0.0000  0.7886  0.9314
e evaluate tree  d duplicate  y copy path  esc runs  ? help
```

- The stage checklist collapses to the one `Took` line.
- The mention-string table is always shown when the evaluation had a `train_path`.
  When it did not: `Mention strings  not recorded (no train_path in this evaluation)`.
- `e` is disabled when the tree is not on this machine; the footer shows
  `e evaluate tree (tree not here)` dimmed and pressing it explains why.
- `y` copies the run directory path (OSC 52, works over SSH in most terminals).

### 5.4 Run: failed, stopped, lost

Same tabs and layout. A banner replaces the top lines; everything that exists stays
visible below it.

```text
XMR4EL  biored_chemical                                   FAILED 19:15
 Overview   Log   Inputs
 ========
 x ValueError: eval dev: no row has a gold label in the tree's vocabulary
   recorded in status.json at 2026-10-20 14:34              2 full log

 Kept   tree e1348683 (saved, on this machine) · no metrics
 + Load corpus 0:01 · + Encode 7:40 · + Label embeddings 0:04
 + Hierarchy 11:12 · + Save tree 0:16 · x Evaluate dev 0:02
d duplicate and edit  e evaluate tree  2 log  esc runs  ? help
```
*illustrative*

| State | Banner |
| --- | --- |
| FAILED | The `error` from `status.json`, then `2 full log`. |
| STOPPED | `Stopped by SIGTERM at <time>. Completed stages and saved files are kept.` |
| LOST | `Process <pid> is gone and wrote no final status (killed, e.g. out of memory). Last event: <stage> at <time>.` |
| UNVERIFIED (other runtime) | `Recorded running on <host>, another machine or container. Heartbeat <age> ago.` |
| UNVERIFIED (same runtime) | `Recorded running here, but the process cannot be inspected (<reason>). Stop unavailable. Heartbeat <age> ago.` |
| UNKNOWN | `status.json is missing or unreadable: <reason>. Showing the files that exist.` |

`Kept` lists the saved tree (only if `xmodel.pkl` exists under the path in
`tree.json`) and each `metrics_<split>.json`. A failed run with a usable tree offers
`e`; with metrics it shows them under the stage line exactly as in §5.3.

### 5.5 Setup: new training run

User 2026-10-23 (supersedes the path fields below): files are picked, not typed. `model_config` = select of
`configs/*.json` (`ctrl+e` edits the chosen one); `dataset` = select of `datasets/` directories with dev / test
files; `training file` = select of that dataset's `train*` files (default `train_plus_ctd`); each evaluation's
`split` = select of the dataset's train / dev / test files (`test_path` follows), `breakdown` = checkbox (the
evaluation's `train_path` = the training file, or the tree's). The default name follows the dataset. A loaded value
discovery does not find is kept as an extra option; a loaded split label that is not the file's split
(`dev_knn0` on `dev.pubtator`) is kept until the split changes. A form that differs from the loaded spec says where.

```text
XMR4EL  New training run                                  gpu01-xmr4el
 name             [ nlm_chem_beam5                               ]
 model_config     [ configs/xmr4el_bc5cdr_config.json            ]
 train_path       [ datasets/nlm_chem/train_plus_ctd.pubtator    ]

 Evaluation dev
 split            [ dev        ]
 test_path        [ datasets/nlm_chem/dev.pubtator               ]
 train_path       [ datasets/nlm_chem/train_plus_ctd.pubtator    ]
                  mention-string breakdown; empty skips it
 beam_size        [ 5          ]   config 10
 leaf_candidates  [            ]   default 100
 topk             [            ]   config 0
 knn_beta         [            ]   config 10

 < + Add evaluation >                     ctrl+s review   esc discard
ctrl+o load spec  ctrl+e edit model_config  tab next  ? help
```

- Field names are the spec's JSON keys. Help (`?` on a field) gives the long
  description: e.g. `leaf_candidates: labels kept per reached leaf`.
- Empty search fields inherit. The field's placeholder shows the resolved value and its
  source (`config 10`): `config` (the model config's `predict_config`), `default` (the
  runner's defaults), `tree` (the saved tree's `predict_config`, not an earlier evaluation's
  overrides). A typed value replaces it.
- A new training draft starts with the newest training run's `model_config`; with no
  training runs it starts with the first `configs/*.json`.
- One evaluation is inline. `+ Add evaluation` appends another block below it;
  each block has its own `< Remove >`.
- `ctrl+o` fills the form from an input spec or a recorded `spec.json`, using the
  normalization in §10. `ctrl+e` opens the model
  config in `$VISUAL` / `$EDITOR` and revalidates on return; no hyperparameter forms.
- Per field, while typing: only form checks (integer, existing file). On leaving a
  field and on `ctrl+s`, `run_experiment.py -check` runs in the background and its
  result appears above the buttons (`checks passed` / the runner's error). The
  runner stays the authority; there is no Validate button.
- Every edit increments a draft revision and immediately invalidates the previous
  check. Each request carries its revision in UI state; only a result for the current
  revision may update errors, defaults, or Review availability. Discard stale results,
  including stale successes; cancel superseded check processes where possible.
- `ctrl+s` waits for validation of the current draft, then opens Review for that
  exact snapshot, including the validated model-config hash for training. Returning
  to edit invalidates Review. Start passes the reviewed snapshot and hash to the
  runner, which rejects a changed configuration and validates the captured contents
  before creating a run directory (§10).

### 5.6 Setup: evaluate a tree

Opened with `e`, the tree prefilled from the selected run:

```text
XMR4EL  Evaluate tree 8528b624                            gpu01-xmr4el
 tree             [ outputs/saved_trees/xmodel_2026-10-20_13-38-22 ]
                  trained by ncbi_disease · 13,385 labels · on this machine
 name             [ ncbi_disease_test_beam5                      ]

 Evaluation test
 ...same evaluation block as §5.5, hints say "tree"...
```

### 5.7 Review

`ctrl+s` from Setup. Shows the spec as it will be passed, so nothing is hidden.

```text
XMR4EL  Review  nlm_chem_beam5                            gpu01-xmr4el
 Trains a new tree, then evaluates dev.
 Run directory  outputs/runs/2026-10-23_14-10-02_nlm_chem_beam5 (new)

 Model config      configs/xmr4el_bc5cdr_config.json
                   sha256 8d105231… (bound; full hash in Inputs after start)
 Training file     datasets/nlm_chem/train_plus_ctd.pubtator

 dev               datasets/nlm_chem/dev.pubtator · breakdown
                   beam_size 5 · leaf_candidates 100 · topk 0 · knn_beta 10

 + runner checks passed (keys, names, search, files, model config)
   Corpus contents are checked when the run reads them.
 ! bc5cdr_chemical is running on this host and shares the GPU.

 < Back >                                                  < Start run >
enter start  j json  a save spec as…  esc back  ? help
```

- Review opens at the top. Each evaluation shows its effective search. `j` toggles the
  exact spec JSON above the summary.
- The spec is not written into `configs/experiments/`. It goes to the runner on
  stdin; the runner already stores it as `spec.json` in the run directory.
  `save spec as…` writes a file only when asked.
- Review binds training settings to the model-config contents, not just its path.
  If that file changes before Start, the runner exits with code 3 (§10); the TUI
  shows `Model configuration changed; review again`, returns to Setup, and
  revalidates. Do not silently accept new defaults.
- Evaluation-only Review binds the tree the same way: `-check` returns the
  `tree_id` from the tree's `xmodel.json`, the launch passes it as `-tree_id`, and a
  different tree at that path gives `Tree changed; review again` (same exit code).
- Data files (`train_path`, `test_path`) are not bound. `provenance.json` records
  their sha256 when the run starts, so a change after Review is visible in Inputs.
- The concurrency line appears only when another run is RUNNING on this host. It
  warns; it does not block.

### 5.8 Compare

`space` marks two runs, `c` opens Compare. Each side has its own evaluation selector,
populated from that run's `evals` in specification order. Default to the first shared
split name in A's order; if none exists, select the first evaluation on each side.
Names select evaluations but never prove comparability; apply §11 to the selected
pair. Changing either selector immediately refreshes metrics, differences, and
comparability notices. Keep selections through refresh; `x` swaps runs together
with their selected evaluations. A selected evaluation without metrics displays
`B: no metrics for dev (not produced yet)`; do not silently substitute another
evaluation. A run with a single evaluation shows its split as plain text with no
focus stop; the selector, and `tab evaluation` in the footer, appear only when a
side has more than one evaluation.

The real common pair, dev and test of one tree (one evaluation each, so no selectors):

```text
XMR4EL  Compare                                           gpu01-xmr4el
 A  ncbi_disease        tree 8528b624   Evaluation dev
 B  ncbi_disease_test   tree 8528b624   Evaluation test

 Not comparable: different evaluation files (dev.pubtator vs test.pubtator).
 Values side by side; no differences computed.

                     A          B
 rows          729 / 757  923 / 945
 acc@1            0.9246     0.8960
 acc@1_all        0.8904     0.8751
 MRR              0.9459     0.9230
 R@cand           0.9835     0.9859
 ms/query           2.99       3.07
x swap  esc runs  ? metric definitions
```

Comparable pair (same evaluation file hash), *illustrative*, A with two evaluations:

```text
 A  nlm_chem         tree f15e98b4   Evaluation [ dev v ]   beam_size 10
 B  nlm_chem_beam5   tree 3c1a77e0   Evaluation dev         beam_size 5

 Evaluation file, preprocessing, and evaluated population: hashes match
 Rows: 4,907 / 4,945 in both runs
 Differs: tree, beam_size 10 -> 5

                     A          B      B - A
 acc@1            0.8753     0.8702    -0.0051
 acc@1_all        0.8686     0.8636    -0.0050
 MRR              0.8923     0.8880    -0.0043
 R@cand           0.9309     0.9101    -0.0208
 ms/query           3.68       2.41      -1.27   same device
```

All recorded metrics are listed (scroll); the mockups are cut. Timing differences
show `same device` or `devices differ: cuda vs mps` from `provenance.json`.

### 5.9 Stop

```text
┌─ Stop bc5cdr_chemical? ─────────────────────────────────┐
│ Sends SIGTERM to bc5cdr_chemical (pid 4182) and its     │
│ workers. It is recorded as STOPPED. The current stage   │
│ is lost; stages already done and saved files are kept.  │
│                                                         │
│ < Keep running >                          < Stop run >  │
└─────────────────────────────────────────────────────────┘
```

`s` or the `Stop run` button under the header (shown only while the run is stoppable) opens it.
Focus starts on `Keep running`. After `Stop run`, the header shows `STOPPING`. If the
process still runs after 30 s (a long torch call delays the signal), the banner adds
`k force stop (SIGKILL; shown as LOST if no final status was written)`. Revalidate
process identity and group isolation before either signal (§9). A verified run that
shares its shell's process group is viewable but has no TUI stop action.

### 5.10 Compact (60 x 18)

```text
XMR4EL  bc5cdr_chemical               RUNNING 14:38
 Overview   Log   Inputs
 ========
 + Load · + Encode · + Labels
 > Hierarchy 1/2 · node 1/1                  6:33
 · Save · · Evaluate dev
 previous: 21:15 total

 1 warning (w)
s stop  esc runs  ? help
```

Runs at 60 columns: Run, Tree, Split, acc@1, State. Below 60 x 16: one line with the
current run's state and `s stop`, plus `resize to 60x16 for more`.

### 5.11 Run monitor (built 2026-10-28)

```text
                                   XMR4EL          Dashboard  New Experiment  Run Monitor  Results
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

### 5.12 Results (built 2026-10-28; V1 refinement 2026-10-28)

```text
                                        XMR4EL          Dashboard  New Experiment  Run Monitor  Results
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
  R@k, R@cand, hybrid) need equal `population_sha256`, or §11's derivation (same tree training-file sha256,
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

### 5.13 New Experiment: Dataset Setup (built 2026-10-28)

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
  Review. List statuses are checked in a worker.
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

## 6. View states

| View | Empty | Loading | Error |
| --- | --- | --- | --- |
| Runs | `No runs in outputs/runs/. n starts one; on a laptop, sync outputs/runs/ from the server.` | Only if the scan takes > 300 ms: `Reading 412 runs…` in the details line. | Unreadable runs stay in the table as UNKNOWN with the reason in details. |
| Runs (wrong place) | `outputs/runs/ not found under <cwd>. Start in the repository or pass --root.` | | |
| Run | Running with no event yet: `· Starting…` as the first stage. | | Banner (§5.4); missing files say which file. |
| Setup | Fields empty; hints show inherited values. | `checking…` beside the check line. | Runner error above the buttons; form errors under the field; focus moves to the first invalid field on `ctrl+s`. |
| Compare | One marked: `Mark one more run (space).` | | Missing metrics: `B: no metrics for dev (not produced yet)`. |

## 7. Keyboard

| Where | Key | Action |
| --- | --- | --- |
| Everywhere | `?` (`F1` inside text fields) | Help for this screen and the focused field. |
| Everywhere | `ctrl+p` | Command palette (every action below is also a command). |
| Everywhere | `esc` | Close overlay, clear filter, then go back. |
| Everywhere | `ctrl+c` (also `ctrl+q`), then `ctrl+c` again / `esc` | Quit after the footer confirmation (no timer) / cancel. Running children keep running (§2). `ctrl+c` no longer copies (`super+c` does). |
| Everywhere | `ctrl+n` | Notification center (§4): `↑↓`, `enter` opens, `c` clears all, `esc` / `ctrl+n` closes. |
| Dashboard | `ctrl+tab` / `ctrl+shift+tab` (`]` / `[` where the terminal sends ctrl+tab as tab) | Next / previous panel: Explorer, Description, Active runs. |
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
| Run monitor | `s`, `Cancel Run` | Cancel Run confirmation (§5.9), only while the run is RUNNING with a verified isolated group, a TUI launch record (§9) and no cancel is pending. |
| Run monitor | `f`, `esc` | The focused panel fills the screen (Live logs: the fullscreen log, same reader, search and filter); `esc` restores, then goes back to the Dashboard with its selection kept. |
| Run monitor | `<` / `>`, drag the left column's right border | Narrower / wider left column by 2 cells: default 35 %, at least 30 cells, at most 60 %, the right side never under 44; kept for the session. |
| Run monitor | Live logs: `/` `n` `N` `l`, scroll keys | As the log viewer. |
| Dashboard | `ctrl+r` | Rescan. |
| Dashboard | `ctrl+b`, `<` / `>`, drag the explorer's right border | Hide / show the explorer; narrower / wider by 2 cells. Default 22 % of the width, at least 18 cells, at most half the screen and never leaving the right panels under 40 (the minimum wins on tiny screens); kept for the session, re-clamped on terminal resize. |
| Description | `up` `down`, click | Select a row (read-only; headings and blank lines are skipped, the list wraps). A rescan of the same item keeps the row with the same field label. |
| Description | `enter` / click on `View Logs  [Open]` | Fullscreen read-only log of the item's run (evaluation: its run, between Run and State; run: between Type and State; tree: its training run, between Training run and Run state). `esc` returns with the row still selected. |
| Log viewer | `↑` `↓` `pgup` `pgdn` `home` `end`, `/` search (enter finds, `esc` closes the box), `n` / `N` next / previous match, `l` level all → INFO+ → WARNING+ → ERROR+, `esc` back | Header: run name, id, state, file, level, match i/n. An active run is polled every second and follows the end while the view is at the end; a finished run is read once. A missing log shows why, nothing is generated. At most 100 000 lines kept, first read ≤ 8 MiB from the end. |
| Explorer, Description, Active runs | mouse drag | Select text (blue); on release it is copied to the clipboard by OSC 52 (`App.copy_to_clipboard`), so it reaches the terminal the TUI runs in, also through `ssh` / `docker exec`; a toast says how many characters. A click without a drag copies nothing; dragging the explorer border resizes, never selects. The terminal must allow OSC 52. |
| Everywhere | `n` | New Experiment: Dataset Setup (§5.13); shown in the Dashboard footer. Already open: back to it. |
| Dataset Setup | Datasets: `up` `down`, `enter` / click; `ctrl+tab` / `ctrl+shift+tab` (`]` / `[`); `esc` | Move (nothing changes) / configure the dataset; next / previous panel (Datasets, Name, configuration); back to the previous screen (unconfirmed edits stay in the draft). |
| Dataset Setup | Configuration: `tab` `shift+tab` (or `up` `down`), `enter`, `space`, click on `[x]`, `ctrl+s`, `u`, `esc` | Move between controls; a file row opens the picker (training variant: selects it); select a training variant / toggle a split; toggle; Confirm Dataset; discard unconfirmed changes (asks); back to Datasets. `ctrl+r` rereads datasets. |
| File picker | `up` `down`, typing, `enter`, `esc` | Move; filter or type / paste a path; open the directory or choose the file; cancel. |
| Runs | `up` `down` `pgup` `pgdn` `home` `end` | Move. |
| Runs | `enter` | Open run. |
| Runs | `n` | New training run. |
| Runs | `t` | Trees: saved trees (runs' `tree.json` + `outputs/saved_trees/`), trained by, training file, labels, runs, on this machine. |
| Trees | `e` / `enter` | Evaluate the selected tree (refused with the reason when it is not on this machine). |
| Runs, Run | `d` | Duplicate setup. |
| Runs | `space` | Mark (at most two; the footer shows `mark n/2`). |
| Runs | `c` | Compare the marked runs. |
| Runs | `/` | Filter: substring of run name, run ID, tree ID or state. `enter` returns to the table with the filter kept. |
| Runs | `o` | Cycle sort. |
| Runs | `ctrl+r` | Rescan `outputs/runs/` (also automatic every 5 s). |
| Compare | `tab` / `shift+tab` | Focus A's or B's evaluation selector. |
| Compare | `enter`, `up` / `down`, `esc` | Open selector, move between evaluations, close without changing; Enter confirms. |
| Compare | `x` | Swap runs with their selected evaluations. |
| Run | `1` `2` `3` | Overview, Log, Inputs. |
| Run | `s` | Stop (confirmation). |
| Run | `w` | Log filtered to warnings and errors. |
| Run | `y` | Copy run directory path. |
| Log | `/` `n` `N` | Search, next, previous. |
| Log | `end` / `pgup` | Follow / stop following. |
| Setup | `ctrl+s` | Review. `enter` in a field never starts anything. |
| Setup | `up` / `down` | Previous / next field. In a select: `left` / `right` cycle, `enter` / `space` open the list. |
| Setup | `ctrl+o` | Spec picker: `configs/experiments/*.json`, then runs' recorded specs (newest first); type to filter. |
| Review | `enter` / `a` | Start run / save spec as a file. |

- Printable keys never act inside text fields.
- Going back restores the selected row, scroll position and focus.
- Marks survive filtering.

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
- Without `events.jsonl` (runs from before this change), the Overview shows the
  stage list as unknown and the Log tab works as usual.

**Schema v1 (2026-10-27, `xmr4el/telemetry/events.py`).** Every line also carries `"v": 1`; `t` is
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

**Cancellation (2026-10-27).** `process.stop(run, force=False)` is the one service a screen calls
(confirmation is the screen's job, §5.9). It re-reads `status.json` first: a run that already
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

**Package.** A separate top-level package `xmr4el_tui/`, optional extra
`xmr4el[tui]` (Textual, pinned in `uv.lock`), entry point `xmr4el-tui`. It imports
nothing from `xmr4el`: `xmr4el/__init__` imports torch (about 1 s warm), which a run
browser must not pay. It reads JSON files and spawns the runner.

Include `xmr4el_tui` and its subpackages in setuptools package discovery; the current
`xmr4el` patterns exclude this sibling package. Include `app.tcss` as package data
and register the entry point against `xmr4el_tui.__main__`. A TUI installation uses
`uv sync --locked --extra tui`; ordinary CLI installations keep the extra optional.
The one `xmr4el` image adds `--extra tui` to its existing dependency sync; Textual
does not affect CLI runs, so there is no second image. Because the image uses
`--no-install-project`, launch the bind-mounted package with `python3 -m xmr4el_tui`
as shown; test the installed entry point separately from the source checkout.

```text
xmr4el_tui/
  __main__.py     entry point, --root
  app.py          screens, keys, commands
  runs.py         read run directories into plain records (no Textual)
  process.py      spawn, liveness check, stop
  app.tcss        layout and theme variables
```

`runs.py` has no UI code, so it is testable on fake run directories and reusable for
a later CSV export.

**Runner changes** (`scripts/run_experiment.py`, instead of moving it into the
library). These are implementation requirements, not existing capabilities:

| Change | Why |
| --- | --- |
| `-check`: validate and return the JSON response defined below | Validation in a child process, no torch in the TUI; effective search for Review. |
| `-model_config_sha256 SHA` (optional; used for TUI training launches) | Bind execution to the reviewed model configuration; reject a mismatch before creating outputs. |
| `-tree_id ID` (optional; used for TUI evaluation-only launches) | Bind execution to the reviewed tree; compared with `xmodel.json` before creating outputs. |
| `-spec -` reads the spec from stdin | No throwaway specs in `configs/experiments/`. |
| `-run_dir PATH` (must not exist; the runner creates it) | The TUI knows the directory before the child starts. Refusal to overwrite stays. |
| `status.json`: process identity and group-isolation fields from §9; retain `started` | Liveness, safe stop, durations. |
| SIGTERM and Ctrl+C recorded as `stopped` | STOPPED is distinguishable from a crash, also after the TUI is gone. |
| `events.jsonl` handler and `extra=` on the stage log calls | §8. |
| `heartbeat` file every 30 s | UNVERIFIED age for runs on other hosts. |
| Evaluation population hash and resolved preprocessing settings in `metrics_<split>.json` | Distinguish equal counts from equal evaluation populations (§11). |

**Specification round trip.** Input specs keep the existing `SPEC_KEYS`. Recorded
`spec.json` adds `xmodel_path`, which is invocation metadata, not an accepted input
key. Load, Duplicate, Save spec as, and equivalent-command generation all use the
same normalization in `runs.py`: copy the recorded object, remove `xmodel_path`,
and carry a non-null value separately as the `-xmodel_path` argument. Null means
training. Leave other unknown keys for the runner to reject; do not silently drop
them. Store no private UI fields in the specification.

Equivalent commands use the normalized input on stdin with `-spec -`, or an
explicitly exported normalized input file. Never pass recorded `spec.json` unchanged
to `-spec`. Exporting an evaluation-only spec also provides the required
`-xmodel_path` command argument; it cannot be reconstructed from the input spec alone.

**Validation response.** `-check` creates no run directory and starts no model work.
After parsing its invocation, stdout contains exactly one JSON object:
`{"ok": true, "error": null, "model_config_sha256": "<64 hex characters>", "tree_id": null, "search": {"dev": {"beam_size": 10, "leaf_candidates": 100, "topk": 0, "knn_beta": 10}}}`.
For evaluation-only checks, `model_config_sha256` is null and `tree_id` is the
tree's `xmodel.json` value.
Return exit code 0 on success. On an invalid input, return code 1 with
`{"ok": false, "error": "<reason>", "model_config_sha256": null, "tree_id": null, "search": {}}`. Route logs to stderr. A startup
failure, malformed response, or inconsistent exit code is a failed check, never a
success. A cancelled/superseded check cannot update the current draft.

For training checks, read the model-config bytes once; derive the SHA-256, validate
the configuration, and resolve defaults from that same captured content. Review
retains the returned hash outside the input spec. A TUI training launch passes it
as `-model_config_sha256`; a fresh check is required for each reviewed launch.
At launch, read the source config once, compare its hash with the reviewed hash,
and reject a mismatch before creating the run directory. On success, validate and
construct the model from those captured contents, save those exact bytes as
`model_config.json`, and use their hash for that config's provenance entry. Do not
reopen the mutable source path for validation, copying, hashing, or model creation.
CLI launches without the optional hash still use one captured configuration.
Capturing needs no library change: `XModel.load_config` is `XModel(**json.load(f))`,
so the runner builds `XModel(**json.loads(raw))` from the captured bytes.

**Binding rejection.** A `-model_config_sha256` or `-tree_id` mismatch exits with
code 3 before any run directory exists and prints one JSON line on stdout:
`{"error": "model_config_changed", "expected": "…", "actual": "…"}`.
For a tree identity mismatch, `error` is `"tree_changed"` instead.
stdout goes to the console log (§10 Spawning); the TUI reads that line after exit
and returns to Setup with the message from §5.7. Any other early exit is shown as a
startup failure with the console log.

Resolve training defaults from the validated model config and core search defaults.
Resolve evaluation-only defaults from the saved tree's readable `xmodel.json`, not
its training run's evaluation metrics. Never load a tree pickle to validate a form;
if readable defaults are absent, explain that effective search cannot be resolved
and keep Review unavailable. The actual execution revalidates and loads the trusted
tree through the existing API.

**Spawning.** `sys.executable scripts/run_experiment.py -spec - -run_dir … [-model_config_sha256 …] [-xmodel_path … -tree_id …]`,
argument list, no shell, working directory = `--root`, `start_new_session=True`,
stdout and stderr to `outputs/runs/.console/<run-id>.log` (dot directory, ignored by
the browser). That file keeps errors raised before `run.log` exists.

**Run browser data.** Everything comes from the run directory: `status.json`,
`spec.json`, `tree.json`, `metrics_*.json`, `model_config.json`, `provenance.json`,
`events.jsonl`, `heartbeat`. A tree is "on this machine" when `<xmodel_path>/xmodel.pkl`
exists. Pickles are never loaded. Runs written before the runner changes get their
start time from the run ID and show UNVERIFIED when recorded running.

## 11. Comparison rules

- Apply every check and metric lookup to the evaluations selected on A and B (§5.8),
  not implicitly to the first evaluation in either run.
- Require matching evaluation file sha256 and resolved preprocessing settings
  (`context_window`, `abbrev_expansion`) before computing metric deltas. Otherwise
  show values side by side with the mismatch or missing information explained.
- For in-vocabulary acc@1, MRR, recall and other subset metrics, also require matching
  evaluated-population hashes. Produce this hash from the ordered retained source-row
  positions and gold label IDs using an unambiguous serialization. Equal row counts
  or coverage percentages do not establish equal populations.
- Same file and preprocessing but different populations: suppress subset deltas and
  show `in-vocabulary populations differ`. `acc@1_all` may still have a delta because
  it counts every loaded row, with uncovered labels as misses. Label that denominator
  explicitly. Compare optional hybrid and seen/unseen metrics only with matching
  `strings_sha256` (the effective dictionary per evaluated row: equal reference files
  are not enough, the tree's vocabulary and abbreviation expansion change it) and
  their appropriate population as well.
- Every delta also needs equal recorded `eval_protocol` (the evaluator's metric
  definitions); population hashes say nothing about how a metric was computed.
- Runs recorded before the population hash existed (all current runs) are compared
  by derivation instead, from files the browser already reads. Their in-vocabulary
  populations count as equal when all of these hold:
  - same evaluation-file sha256 (`provenance.json` of each run);
  - same training-file sha256 and preprocessing (`context_window`,
    `abbrev_expansion`) for each tree, read from the run that trained it, found by
    `tree_id` (an eval-only run has no `model_config.json`; its training run does);
  - every involved run (both trainings, both evaluations) at the same git commit
    with `dirty: false` in `git.json`, since reader changes move rows.
  The header then says `Populations equal by derivation (same files, preprocessing,
  commit)` instead of `hashes match`. If any input is missing (e.g. the training run
  was not synced), comparability is unknown: values without subset deltas. The
  browser never reads a corpus or loads a pickle to decide.
- The `Differs:` line lists tree, every search setting, features and package
  versions that differ.
- acc@1 and acc@1_all are always both shown. `R@cand` is labelled
  `after leaf_candidates and topk` in metric help.
- ms/query is batch prediction including encoding; deltas additionally require the
  same evaluated population and device. Device names alone do not establish equal
  accelerator hardware; display the available provenance beside the timing.

## 12. Roadmap and tests

| Increment | Content | Runner changes needed |
| --- | --- | --- |
| 1. Browser | Runs, Run (Overview from metrics, Log, Inputs), Compare; package discovery, stylesheet data, optional extra, installed entry point and Docker extra | evaluation population hash and preprocessing in `metrics_<split>.json` (eval side only); older runs use the §11 derivation |
| 2. Launch | Setup, Review, spawn, stage checklist, stop, liveness | the rest of §10 |
| 3. Reuse | Duplicate, evaluate tree, previous-run timings | none beyond 2 |

Tests (offline, synthetic):
- `runs.py` on fake run dirs: every state of §4, missing/corrupt files, pre-change
  runs without `events.jsonl` or `pid`.
- Runner: `-check` output, `-spec -`, `-run_dir` refusal, `stopped` on SIGTERM,
  `events.jsonl` stages on the tiny synthetic tree.
- Specification round trips: train and evaluation-only recorded specs normalize,
  pass `-check`, and reproduce their invocation mode; unrelated unknown keys fail.
- Encoding: part events arrive in fit order (mention TF-IDF, context TF-IDF,
  transformer, assemble); duplicate mentions count once; cache hits and TF-IDF-only
  paths show no batch percentage; failed OOM batches do not advance; the parent
  Encode stage finishes only after `assemble`.
- Comparison: the same file with equal covered-row counts but different retained
  rows has no subset delta; matching population/preprocessing permits it. All-row
  accuracy, missing metadata, and optional breakdown inputs use their own guards.
  Derivation for old runs: equal only with matching eval sha, train sha,
  preprocessing and one clean commit; a missing training run or a dirty tree gives
  unknown.
- Comparison selection: compare a run containing dev and test with a test-only run;
  test is selected by default, and the test-only side shows plain text, no selector. Changing A to dev suppresses deltas when its input
  differs. Also test no common split, missing selected metrics, and swapping sides
  without losing the selected evaluations.
- Textual `run_test()` + Pilot: keys of §7, focus restore, modal default, layouts at
  120x40, 80x24, 60x18, `NO_COLOR`.
- Validation: an old success arriving after a new failure cannot enable Review;
  stale errors/defaults are discarded too. Malformed JSON and nonzero process exits
  cannot pass a check; editing invalidates the reviewed snapshot.
- Reviewed configuration: change an inherited beam size after Review; Start rejects
  the hash mismatch without creating a run directory. With unchanged contents,
  execution, the saved config, provenance, and reviewed defaults agree. Changing
  the source after the runner captures it cannot affect the configuration used.
  The mismatch exits 3 with the JSON line; the TUI maps it (and `tree_changed`) to
  the review-again message and returns to Setup, while other early exits show the
  console log.
- Process: fake child that sleeps; stop, delayed stop, force stop, TUI exit with the
  child still running, reused PID not signalled. A live child sharing its launching
  process group is viewable but cannot be stopped through the TUI. Changed runtime,
  start token or group identity before either signal prevents signalling.
- Packaging: build and install a wheel with the TUI extra outside the checkout;
  check `xmr4el_tui` import, the entry point, and stylesheet loading. Build the
  image with the extra and check `python3 -m xmr4el_tui --help` against mounted code.

Manual before release: SSH drop during a run, reattach through `docker exec`, color
inside the container with and without `-e TERM -e COLORTERM`.

## 13. Out of scope

Dashboards and charts, dataset browsing, tree visualisation, hyperparameter forms,
queues and schedulers, artifact deletion, cache management, a prediction playground.
Unattended batches stay on the CLI (`run_*.sh`, `docker run -d`).
