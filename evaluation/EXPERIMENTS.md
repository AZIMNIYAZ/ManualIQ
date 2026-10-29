# Step 12 experiments

Every experiment is compared with the baseline (`evaluation/BASELINE.md`) on the development
set (dataset `4d4be1280743`: 40 questions + 6 conversations) with `scripts/eval_compare.py`. The
held-out set (15 questions, `f2b8451e9083`) is frozen and is only run on the baseline and on the
final configuration.

The development set's hash became `c03b2b01c65f` on 2026-10-28: the key fact of cv-02-2 now accepts
both of the manual's spellings, "Network Service Filter" or "Network Services Filter" (see "Fix B"). Runs before
and after are compared with `eval_compare.py --force`.

**Final configuration (the defaults since Step 12):** `rag_500_60_ctx` (500/60 + contextual headers),
retrieval fallback on, answer prompt 1.2, answer output limit 1,500 tokens. Results: see "Final results".

## Rules

1. Change one thing at a time. The free retrieval evaluation screens a change; a paid answer run
   decides whether it is kept.
2. Keep a change only if accuracy goes up, false answers stay at 0, citations and wrong-procedure
   text do not get worse, and latency stays acceptable.
3. Report the questions that flipped, not only percentages. 32 answerable questions give Hit@5 a
   95% interval of about ±14 points (baseline 78%: 61%–89%), so a change of one or two questions
   is treated as noise.

## Baseline

| | |
|---|---|
| Configuration | `rag_350_60`, top 5, prompt 1.1, rewrite prompt 1.1, `grok-4.3` |
| Retrieval | Hit@1 50%, **Hit@5 78% (61%–89%)**, Hit@10 94%, MRR 0.64; search median 58 ms, p95 86 ms |
| Answers | **accuracy 75% (64%–84%)**, 0 false answers, 10 false refusals, key facts 75%, 0 wrong-procedure answers; chat median 5.4 s, p95 11.7 s |

## Experiments

| # | Change | Hit@1 | Hit@5 | Hit@10 | MRR | Flips (Hit@5) | Accuracy | False answers | Decision |
|---|---|---|---|---|---|---|---|---|---|
| — | baseline `rag_350_60` | 50% | 78% | 94% | 0.64 | — | 75% | 0 | reference |
| R1a | chunks 200/40 | 47% | 84% | 91% | 0.64 | +4 / −2 | – | – | rejected: tables 80%→60%, t-ng-01 and d-tp-02 lost, slower search (83 ms) |
| R1b | chunks 350/0 | 47% | 81% | 94% | 0.61 | +2 / −1 | – | – | rejected: within noise, lower MRR |
| R1c | **chunks 500/60** | **62%** | **88%** | 94% | **0.74** | **+4 / −0** | pending | pending | **base for R2–R4**; answer run to confirm |
| R2a | **500/60 + contextual headers** (vs R1c) | **84%** | **94%** | 94% | **0.88** | **+2 / −0** vs R1c | **88%** (78–93%) | **0** | **kept: confirmed by the paid answer run** |
| R2b | 350/60 + contextual headers (vs baseline) | 66% | 94% | 94% | 0.77 | +6 / −0 vs baseline | – | – | confirms R2a: the header effect does not depend on chunk size |
| R3a | 500/60 + headers **+ hybrid search** (vs R2a) | 84% | 94% | 94% | 0.89 | +1 / −0 vs R2a | – | – | **rejected**: below the agreed bar; dependency not adopted |
| R3b | 350/60 + hybrid search (vs baseline) | 56% | 91% | 91% | 0.70 | +5 / −0 vs baseline | – | – | helps on its own, but less than headers (R2b) |
| R4a | 500/60 + headers **+ re-ranking** (vs R2a) | 84% | 94% | 94% | 0.88 | +0 / −0 vs R2a (Hit@1: +2 / −2) | – | – | **rejected**: no gain, re-ranking p95 2.65 s |
| R4b | 350/60 + re-ranking (vs baseline) | 66% | 88% | 91% | 0.74 | +4 / −1 vs baseline | – | – | helps on its own, but less than headers and breaks p-tp-02 |
| P1a | parent-child: children 200/40 + headers, parents 500, 5 sent (vs R2a) | 84% | 91% | 94% | 0.88 | +0 / −1 vs R2a (evidence reaching LLM) | – | – | **rejected**; context confusion Hit@1 100% |
| P1b | parent-child: parents ~1000, 3 sent (vs R2a) | 84% | 88% | 94% | 0.87 | +0 / −2 vs R2a | – | – | **rejected**: tables 80%, +21% context tokens |
| Q1 | search-query rewriting on every question, prompt 1.0 (vs R2a) | 78–84% | 91% | – | – | +0 / −1 (p-dl-02) | – | – | **rejected** (branch `q1-query-rewrite`): rewrites kept the user's words; broke a working question |
| F1 | R2a **+ retrieval fallback after a refusal** | – | – | – | – | 4 of 4 wording-gap failures recovered | **93%** automatic | **0** | **kept** (below) |
| F2 | F1 + output limit 1,500 + **answer prompt 1.2** | – | – | – | – | nb-02, cv-02-2 fixed | **94%** automatic, **100%** accepted | **0** | **kept: final** (below) |

### R1: chunk size and overlap

All collections were built with cleaner 1.3 (`rag_200_40` was rebuilt), and no key phrase is split
across chunks at any of these sizes (`check_labels.py --size/--overlap`).

- **500/60** fixed c-as-01 (ASUS Network Service filter), cv-01-2 (D-Link guest zone follow-up), d-dl-01
  (D-Link website filter) and p-as-01 (ASUS time scheduling), and broke nothing. Hit@1 rose 50%→62% and
  MRR 0.64→0.74.
- **Cost:** with the 2,500-token budget, at least 4 (usually 5) chunks still fit, and no hit was dropped
  by the budget. The median context grows from 1,628 to 2,140 tokens (+31% input tokens per answer).
  500 is the maximum chunk size for bge-small (512 tokens including special tokens).
- **Caveat:** Hit@K favours bigger chunks, because a bigger chunk covers more pages and more text, so
  it "matches" more easily. Whether the model actually answers better is decided by the paid answer
  run, not by Hit@5.
- **Still missed at 500/60:** t-te-01 (Tenda default IP, table), p-as-02, p-dl-02, p-te-01 (paraphrases)
  and cv-04-1 (D-Link QoS). These are the targets for R2–R4.

### R2: contextual headers

The embedding input becomes `Brand Model | Section` + chunk text, for example `D-Link DIR-X1560 | Website
Filter`. The stored text, the context sent to the LLM and the citations are unchanged. The setting is
`CONTEXTUAL_HEADERS` (off by default); with headers on, the collection name gets `_ctx`.

**Section labels (header version 1.1):** only trusted labels are used. A label is trusted if it is in
the manual's table of contents, a numbered heading, or a procedure title. Otherwise a trusted heading on
the **same page** is used, or no section at all. At 500/60 the counts were: trusted 332, fallback 2,
omitted 98 (of 432). The rejected labels and their replacements are in `evaluation/r2_rejected_labels.md`.

How the fallback rule got there:
- **Any earlier heading:** gave the D-Link "LED Control" chunk the header "Quick VPN".
- **Up to 2 pages back (version 1.0):** still left three wrong topics, found in review: D-Link "Time
  Configuration" → "Quick VPN", Netgear MU-MIMO → "Delete a static route", Tenda "PPTP server" → "DNS".
  The D-Link one is the chunk with "Time & Schedule - Schedule".
- **Same page only (1.1):** removes all three. A missing section loses a little help; a wrong one points
  retrieval at the wrong topic. Re-running both arms showed no cost: R2a gave identical numbers, and
  R2b moved from Hit@5 91% / Hit@1 69% / MRR 0.79 to 94% / 66% / 0.77 (within noise, one more question
  fixed: p-dl-02).

**No truncation:** headers are shortened to fit the model's 512 tokens, counted with its own tokenizer
including special tokens. At 500/60, 12 headers lost the section and 6 kept the brand only; no chunk text
was cut. At 350/60 no header had to be shortened.

**Baseline unaffected:** with headers off, `rag_350_60` returns identical top-10 chunks and scores for
all 53 items (0 flips).

**Results:**

| Comparison | Hit@1 | Hit@5 | MRR | Fixed | Broken |
|---|---|---|---|---|---|
| R2a: `rag_500_60` → `rag_500_60_ctx` | 62% → **84%** | 88% → **94%** (80–98%) | 0.74 → **0.88** | p-dl-02, **t-te-01** | none |
| R2b: `rag_350_60` → `rag_350_60_ctx` | 50% → 66% | 78% → 94% (80–98%) | 0.64 → 0.77 | c-as-01, cv-01-2, d-dl-01, p-as-01, p-dl-02, **t-te-01** | none |

- **Both arms agree.** Headers mostly move the right chunk to rank 1: Hit@1 rose by 16–22 points.
  Hit@10 is unchanged (94%), so headers re-rank what was already found rather than finding new
  passages.
- **The Tenda default-IP table (t-te-01), missed since Step 5, is now found in both arms**, but not
  because the header names the table. The matching chunk (pp. 153–155) starts with the FAQ entry and
  continues into the A.1 Factory settings table, so its header is "Tenda AX3000 … | 10.4 Forgot my
  password". The likely reason it now ranks higher is that "Tenda … login password" is close to
  "default login IP of the Tenda router". This was checked in Qdrant after a first write-up wrongly
  assumed the header said "Factory settings".
- **Limitation:** the section label is the heading where a chunk *starts*. A chunk that spans two
  sections only gets the first one, and the 500-token chunks span sections more often.
- **Category checks:** tables 80% → 100% and context confusion 100% in both arms; no-brand questions
  stay at 100% Hit@5, with the same total coverage (16/24 manual matches at 500/60, distributed
  differently). Brand filter 100%, regulatory leakage 0.
- **Search time** looked faster, but query embedding and search are unchanged, so that is run-to-run
  noise, not an effect of the headers.
- **Still missed (both arms):** p-as-02 ("keep everyone else out" → MAC filter), p-te-01 ("kick it off for
  good" → blacklist) and cv-04-1 (D-Link "highest priority" → QoS Engine). All three are vocabulary
  gaps between the user's words and the manual's. These are the targets for R3 (keywords will not help
  much here) and R4 (re-ranking).
- The interval for Hit@5 still overlaps the baseline's, but the gains are consistent across both arms
  with nothing broken. The paid answer run decides.

### R3: hybrid search (dense + BM25 keywords) - rejected

**Agreed before the run:** hybrid is kept only if, against the current best (R2a), it fixes at least 2
questions with 0 broken, or raises Hit@1/MRR by about 5 points or more, with no regression in tables,
context confusion or no-brand questions and acceptable latency. Otherwise the extra dependency
(`fastembed` 0.9.0 and onnxruntime) and code are not worth having.

**Setup** (branch `r3-hybrid`, commit 3660329, not merged):
- BM25 keyword vectors from fastembed's `Qdrant/bm25`, built from the same text as the dense embedding.
  Qdrant applied the IDF.
- Both searches got the same brand and regulatory filter (tested against real Qdrant: a stronger keyword
  match in another brand never got through).
- The results were merged with RRF (k = 60), 20 candidates from each search.
- The dense vectors were identical to the compared collections (max difference 0), so differences come
  from the keywords alone.

**Results:**

| Comparison | Hit@1 | Hit@5 | MRR | Fixed | Broken | Search median |
|---|---|---|---|---|---|---|
| R3a: `rag_500_60_ctx` → `+ hybrid` | 84% → 84% | 94% → 94% | 0.88 → 0.89 | cv-04-1 | none | 46 → 56 ms |
| R3b: `rag_350_60` → `+ hybrid` | 50% → 56% | 78% → 91% | 0.64 → 0.70 | c-as-01, cv-01-2, d-dl-01, p-dl-02, t-te-01 | none | 58 → 57 ms |

- **R3a misses the bar.** It fixed 1 question: cv-04-1, where "highest priority" matched "priority" on
  the D-Link QoS page. Hit@1 is unchanged and MRR is +0.01. Search is about 9 ms slower, and no-brand
  coverage fell from 16 to 14 of 24 manual matches.
- **R3b explains why.** Keywords help the plain baseline (5 fixed, 0 broken), but contextual headers
  already gave the same Hit@5 (94%) with a better rank 1 (Hit@1 66% vs 56%). The headers already capture
  what the keywords add.
- **As predicted, keywords did not fix the wording gaps** (p-as-02 "keep everyone else out", p-te-01
  "kick it off for good"), because the user's words do not appear in the manual.
- **Clean-up:** the code stays only on its branch. `fastembed` and its 9 extra packages were uninstalled
  (the package list is identical to before R3), the BM25 model cache and the `_hyb` collections were
  deleted, and `requirements.txt` is unchanged.

### R4: cross-encoder re-ranking - rejected

**Agreed before the run** (on the 32 standalone questions, against R2a), keep re-ranking only if:
- **Gain (any one):** Hit@5 at least 2 fixed with 0 broken; or Hit@1 net +2 with 0 broken at Hit@5; or
  MRR +0.05 or more.
- **No regression** in tables, context confusion or no-brand questions.
- **Latency:** re-ranking adds at most 1 s at p95.

**Setup** (branch `r4-rerank`, commit 1244d9b, not merged):
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision 233902d, Apache 2.0, 92 MB. Downloaded with only
  the needed files; the full repository is 890 MB because of ONNX and OpenVINO copies.
- It re-ranks the top 20 filtered dense candidates and keeps the top 5. Each passage is scored with the
  header it was embedded with.
- No new package: it uses the existing `sentence-transformers`.
- With re-ranking off, results were identical to before (53/53 items).

**Before the run:** on R2a, 43 of the 44 answerable items had their evidence in the top 20 (32 questions
+ 12 conversation turns). That is the ceiling for re-ranking, not a prediction:
- p-as-02 ("keep everyone else out") was at rank 20, and cv-04-1 at rank 14;
- p-te-01 ("kick it off for good") was not in the top 20, so re-ranking could not fix it.

**Results:**

| Comparison | Hit@1 | Hit@5 | MRR | Hit@5 fixed / broken | Hit@1 fixed / broken | Re-rank time |
|---|---|---|---|---|---|---|
| R4a: R2a → `+ re-ranking` | 84% → 84% | 94% → 94% | 0.88 → 0.88 | none / none | c-dl-02, p-dl-02 / c-dl-01, p-tp-01 | median 2.53 s, p95 2.65 s |
| R4b: `rag_350_60` → `+ re-ranking` | 50% → 66% | 78% → 88% | 0.64 → 0.74 | c-as-01, d-dl-01, p-dl-02, t-te-01 / **p-tp-02** | 7 / 2 | median 1.75 s, p95 1.86 s |

- **R4a fails on gain and on latency.** No question was fixed at Hit@5; Hit@1 swapped two questions for two
  others; MRR +0.01. Re-ranking 20 passages of about 500 tokens takes about 2.5 s on this CPU (10 threads).
  The only gain was the conversation turn cv-04-1 (D-Link "highest priority"), which moved into the top 5.
- **The target miss was not fixed.** p-as-02 ("keep everyone else out" → MAC filter) stayed outside the top
  5. A single example before the run had already shown the model scoring the MAC-filter passage below an
  unrelated setup passage. This model was trained on web search questions (MS MARCO) and does not bridge this
  wording gap.
- **R4b:** re-ranking helps the plain baseline, but less than contextual headers (R2b: 6 fixed / 0 broken,
  Hit@1 66%). It broke p-tp-02 and gave no gain on paraphrased questions.
- **Truncation:** at 500/60, 175 of 1,060 question + passage pairs were longer than 512 tokens and their end
  was cut for scoring. For every evidence chunk, the key phrase was in the part the model read, so truncation
  did not hide any evidence (checked with the exact header + text the re-ranker scored).
- **Clean-up:** the code stays on its branch only. The model cache was deleted; no package was added or
  changed.

### Retrieval: where things stand

The current best is **500/60 + contextual headers** (R2a): Hit@1 84%, Hit@5 94%, MRR 0.88 on the 32
questions, against the baseline's 50%, 78% and 0.64.

- **R5 (top-k 5 → 8) is not run.** On R2a, Hit@10 equals Hit@5 (94%): no evidence sits at ranks 6–10, so
  more chunks would only add tokens and noise.
- **Remaining misses:** two paraphrases whose wording is absent from the manual. p-as-02 is at rank 20, and
  p-te-01 is outside the top 20. Retrieval experiments did not fix them; whether the answer prompt and model
  handle the context well is decided by the paid answer runs.

## Diagnosis of the remaining misses (mentor's suggestion: understand before experimenting)

After R4, the misses were traced on the current best (R2a) **before** further experiments: the
evidence chunk, its similarity to the question, and what ranked above it.

| Question | Evidence chunk | Cosine: evidence vs 1st | What ranked above |
|---|---|---|---|
| p-te-01 "device I don't recognise … kick it off for good" | pp. 60–63: 7.3 Add to blacklist + 7.4 + 7.5 + start of 8 | 0.625 vs 0.783 | generic Wi-Fi pages (hidden network, login) |
| p-as-02 "only my own devices … keep everyone else out" | pp. 85–87: MAC filter + RADIUS + Professional | 0.670 vs 0.713 (rank 20) | FAQ, Bridge, Router Setup |
| cv-04-1 "give my laptop the highest priority" | pp. 69–70: QoS Engine | 0.708 vs 0.761 (rank 14) | Before you Begin, Troubleshooting, Modem |

Four candidate causes:
1. **Several topics in one chunk** (the 500-token chunks above).
2. **Header lost.** The Tenda chunk starts with the trusted heading "7.3 Add a client to the blacklist",
   but it is so long that its header was shortened to "Tenda" to fit 512 tokens. This happened to 17
   chunks.
3. **Generic pages** attract vague wording.
4. **Wording gap.** The user's words do not appear in the manual.

P1 was designed to test causes 1 and 2.

### P1: parent-child chunking - rejected

**Setup** (branch `p1-parent-child`, commit b5547b0, not merged):
- Children of 200 tokens (overlap 40, with contextual headers) are searched; children never cross their
  parent.
- Each child's parent is sent to the LLM: heading-aligned, no overlap, stored in a separate collection
  without vectors (`…_parents`).
- Children are de-duplicated by parent and ranked by their best child; the brand and regulatory filters
  apply to the children.
- No child header had to be shortened (against 17 at 500/60).
- With flat chunking the search was unchanged (53/53 items).

**Predictions, written before the run:** p-te-01 should improve, p-as-02 may improve, cv-04-1 should
not.

**Agreed bar:** against R2a on the 32 questions, at least 2 fixed with 0 broken, or Hit@1 net +2, or MRR
+0.05; no regression in tables, context confusion or no-brand questions; context tokens at most +25%.

**What reaches the LLM** (new measures, computed on the first TOP_K results within the token budget):

| | Evidence in context | All key facts in context | Context tokens (median) | Fixed / broken | Hit@1 | Search median |
|---|---|---|---|---|---|---|
| R2a: 500/60 + headers | **94%** | **93%** | 2,128 | – | 84% | 55 ms |
| P1a: parents 500, 5 sent | 91% | 89% | 2,073 | 0 / 1 (p-dl-02) | 84% (2 fixed, 2 broken) | 88 ms |
| P1b: parents ~1,000, 3 sent | 88% | 82% | 2,569 (+21%) | 0 / 2 (p-dl-02, t-tp-01) | 84% | 85 ms |

- **The predictions failed.** For p-te-01 and p-as-02 even the clean single-topic child was outside the
  top 20 children. Causes 1 and 2 were real but minor.
- **The real cause is the wording gap (cause 4).** The same clean evidence child, ranked against
  different wordings of the question:

| Wording | Evidence child rank | Cosine |
|---|---|---|
| "There's a device I don't recognise… kick it off for good?" | 126 of 201 | 0.624 |
| "How do I block an unknown device on my Tenda router?" | 9 | 0.717 |
| "How do I add a client to the blacklist on my Tenda router?" | **1** | **0.864** |
| "Only my own devices… keep everyone else out?" | 90 of 153 | 0.625 |
| "How do I allow only my own devices to connect…?" | 60 | 0.649 |
| "How do I set up the wireless MAC filter on my ASUS router?" | **1** | **0.887** |

  With the manual's words, retrieval is already perfect; with the user's words, the embedding model
  does not make the connection. This cannot be fixed by chunking, re-ranking or keywords (R3, R4, P1);
  it needs the question to be expressed in the manual's vocabulary.
- **One positive result: context confusion Hit@1 rose to 100% in both arms** (small children pinpoint
  the right OS variant or add/remove procedure). It did not change what reaches the LLM (already 100%),
  so it does not outweigh the losses.
- **Kept on the main branch:** the "what reaches the LLM" measures (evidence in context, all key facts in
  context, context tokens), which apply to any strategy. **Clean-up:** the P1 code stays on its branch
  only, and the four P1 collections were deleted.

## Answer run on the current best (R2a: 500/60 + contextual headers)

Run on the same protocol as the Step 11 baseline (40 questions + 6 conversations, 3 runs for tables and
context confusion), on the main branch. Estimated at most $0.62; actual cost $0.53. Run folder:
`evaluation/results/20261027-161627_rag_500_60_ctx_generation_r2a`.

| | Baseline (Step 11) | R2a | Change |
|---|---|---|---|
| **Accuracy, all 73 items** | 75% (64–84%) | **88% (78–93%)** | +12 pts |
| Question runs | 78% | 92% | +13 pts |
| Key facts in answers | 75% | 94% | +19 pts |
| **False answers** | 0 | **0** | – |
| False refusals | 10 | 2 | −8 |
| Refusal precision / recall | 44% / 100% | 80% / 100% | |
| Conversation turns passed | 62% | 69% | |
| Chat time median / p95 | 5.4 s / 11.7 s | 4.5 s / 8.4 s | faster |

**Flips:** 11 fixed, 2 "broken". The fixed ones are c-as-01 ×3, cv-01-2, d-dl-01, d-tp-02, p-as-01, p-dl-02 and
t-te-01 ×3. The 2 broken are both judge mistakes (below), so in reality 0 broke. Every repeated question
gave the same result in all 3 runs, except c-dl-01 (✗✓✗, a judge mistake).

**The 9 failed items, read one by one:**

| Item | Category | Cause |
|---|---|---|
| p-te-01, p-as-02 | retrieval | wording gap (evidence ranks 1st with the manual's words, ~100th with the user's) |
| cv-04-1, cv-04-2 | retrieval | "highest priority" does not retrieve the QoS Engine page; the follow-up inherits the miss |
| cv-02-2 | rewrite / generation | the follow-up was rewritten literally ("Can a URL filter … block FTP?"); the Network Services Filter page was in the context, but the model refused |
| c-dl-01 runs 1 and 3 | **judge mistake** | the answer is the manual's own Windows 10 instruction (p. 125), whose menu path itself mixes Windows 7 and 10; the judge called it a hybrid. Run 2 of the same answer was judged grounded |
| cv-05-1 | **judge mistake** | the manual says 5 s (p. 109) and 7 s (p. 158); the answer gives both with correct citations (already graded "grounded" by the human reviewer in Step 11) |
| p-dl-01 | **strict label** | "Can I control my D-Link router by talking to Alexa?" was answered "yes" with the voice commands, supported by the sources, but without the setup steps the key facts require (flagged as strict in Step 11) |

**What is working and what needs work:**
- Counting the three verified judge mistakes as correct, **67 of 73 items (92%)** are correct; with the
  strict label too, 68 of 73 (93%).
- **Genuine failures: 5 of 73 items (7%).** Four are retrieval misses with one cause (the wording gap: two
  questions and one conversation with its follow-up), and one is a literal follow-up rewrite.
- The judge's mistakes all involve manuals that contradict themselves or use odd menu paths. The judge's
  validation (12/12 planted errors caught, kappa 0.86) still holds, but its "not grounded" verdicts are
  checked by hand before being counted as failures.

**Accepted accuracy for R2a: 93% (68 of 73 items, 95% interval about 85–97%).** The human reviewer accepted
the four items above as correct: c-dl-01 runs 1 and 3 and cv-05-1 (judge mistakes) and p-dl-01 (strict
label). The automatic score was 88%. By type: standalone questions 38/40 (95%), conversation turns 10/13
(77%). Remaining failures to fix: p-te-01, p-as-02, cv-04-1, cv-04-2, cv-02-2.

## F1: retrieval fallback after a refusal (fixes the wording gap)

**Idea** (from the mentor's suggestion of an LLM layer that knows the manuals' vocabulary): the normal
pipeline runs first. Only when the answer is a refusal (or search returns nothing), an LLM that is given
the manuals' own feature names writes alternative searches, and the answer model gets **one** more
attempt at the **original** question. Answered questions never enter it, so they cannot be changed by it.

**Design** (`app/services/retrieval_fallback.py`, `app/services/manual_glossary.py`, setting `RETRIEVAL_FALLBACK`):
- **Glossary:** every heading of every manual's table of contents (`documents/glossary.json`, built by
  `scripts/build_glossary.py`; 77–186 headings per manual, about 340–1,500 tokens for one brand). Nothing in it
  was chosen for particular questions.
- **One call** to the rewrite model (`grok-4.20-0309-non-reasoning`) writes 3 alternative queries. It gets the
  question (the standalone rewrite for follow-ups) and the glossary of the detected brand (all manuals if none).
- **Safeguards per alternative** (it is dropped if any fails):
  - it may not name a brand the question does not name;
  - the question's technical terms (anything with a digit or two capitals: "AiMesh", "6 GHz", "OpenVPN") must
    survive, so an unsupported feature is never swapped for a supported one;
  - brand and model names are exempt from that check (this was Q1's flaw);
  - an off-topic question gets no alternatives, at no extra cost.
- **Search:** the original question plus each alternative, with the **original** brand filter, merged by
  reciprocal rank fusion (fixed before testing). Then one more answer to the original question. **No loop:** a
  second refusal is final.
- **Why refusal is the trigger:** in the R2a run, all 4 wording-gap failures ended in a refusal and no answered
  question did. A score threshold was not used, because right and wrong chunks score too close together.

**Probes** (`evaluation/fallback_probes.jsonl`, a separate file outside the dev hash): 11 questions about features a
manual does not have but that sound like ones it does. They cover lookalike pages, brand-mixing traps and real-world
features absent from the manual, and 6 are in everyday wording. Each feature was checked absent from its manual.
pr-as-02 ("check how fast my internet is" on ASUS) was **relabelled answerable** after the first run: p. 54 says the
traffic monitor shows the "speed of your Internet". The absence check had searched for "internet speed".

**Rule B** (reviewer's decision): an answer to an unanswerable question is correct if it:
1. says the asked feature is not in the manual;
2. gives a related feature from the same brand's manual, cited;
3. does not present that feature as doing what was asked.

It is a scoring rule, not an instruction to the model. Such answers are listed under "Review under rule B" in
`summary.md` and count as false answers until reviewed.

**Targeted run** (`…150627_…fb_targeted`, `…150900_…fb_probes`, about $0.24). Bar agreed before the run:
at least 3 of 4 recovered, no false answers, passing turns unchanged.

| Item | Alternatives written | Evidence rank after | Result |
|---|---|---|---|
| p-te-01 | "Client management: Add a client to the blacklist", … | 2 | answered correctly |
| p-as-02 | "Wireless MAC Filter", … | 1 | answered correctly |
| cv-04-1 | "QoS Engine", "Settings QoS Engine", … | 1 | answered correctly |
| cv-04-2 | "Connected Clients", "QoS Engine", … | 2 | answered correctly |

All 9 unanswerable and out-of-scope items were refused again. The term check rejected alternatives that dropped
"OpenWrt", "MLO", "6 GHz", "OpenVPN", "AiMesh" and "USB". In the probes, all 10 unanswerable questions were refused.

**Full run with F1** (`…152747_…fb_full`, about $0.59): automatic **93%** (68/73, 85–97%), 0 false answers,
**0 false refusals** (R2a: 2). The fallback ran on 12 items and turned 3 into answers; the other 9 were correct
refusals and stayed refusals. Latency rose (median 7.8 s), but items without the fallback rose equally, because
xAI's generation was slower that day (median 4.3 s → 7.2 s).

### Answer truncation found (nb-02) and fixed

nb-02 ("How do I set up port forwarding?", no brand) is answered for 3 manuals. It hit the **800-token output
limit in every run, R2a included**, so its answers ended mid-sentence. No check caught it: the key facts come early,
and the judge does not check completeness.

Fixes:
- `LLM_MAX_OUTPUT_TOKENS` raised from 800 to **1,500**. nb-02 answers are now 989–1,363 tokens, complete in 3 of 3 runs.
- A `truncated_answer` warning when the model stops at the limit, which the evaluation counts as a failure.

No other answer in any run reached 700 tokens.

### Fix B: answer prompt 1.2 (cv-02-2)

cv-02-2 ("Can it [the URL filter] also block a whole service like FTP?") had the Network Services Filter page at
**rank 1 in every run**, with the same rewrite. Even so, the model answered 3 times out of 5 and refused twice
(repeat runs `…cv02_rep*`). The fallback could not help, because the evidence was already retrieved.

Prompt 1.2 adds rule 8: *if the question asks whether a named feature can do something, and the sources show it
cannot but a different feature of the same router does exactly what is asked, say so and answer with that
feature, cited. If no feature does it, give the exact not-found reply.* The rule is deliberately narrow: loosely
related features (e.g. Tenda's Bandwidth Control for a priority question) still get a refusal.

| Check (with F1) | Before | Prompt 1.2 |
|---|---|---|
| cv-02 conversation × 5 | cv-02-2 3/5 | **5/5** |
| 11 probes | 11/11 (after relabel) | **11/11**, 0 false answers |
| Full run (`…170113_…p12_full`, about $0.62) | 93% automatic | **94% automatic (69/73, 87–98%)**, 0 false answers, 0 false refusals, conversation turns 92%, chat median 5.5 s, p95 11.6 s |

Two items flipped to failure in the full run: p-dl-01 and p-dl-02, answered "Yes, feature X does it" without
steps. Repeats under **both** prompts gave the same rate (p-dl-01 0/3 and p-dl-02 1/3, under 1.1 and under 1.2),
so prompt 1.2 did not cause them. They follow rule 7 ("answer only the question asked") and were accepted.

## Final results

**Accepted items** (reviewer's decisions; each answer read by hand):

| Item | Failed by | Why accepted |
|---|---|---|
| p-dl-01, p-dl-02, h-p-te-01 | key facts (no setup steps) | "Can it…?" answered "yes, feature X does it", correct and cited; rule 7 keeps answers to the question asked |
| c-dl-01 run 2 | judge | copies the manual's own Windows 10 menu path, which mixes Windows 7 and 10 (p. 125) |
| cv-05-1 | judge | the manual itself says 5 s (p. 109) and 7 s (p. 158); both are given with citations |

(nb-02 in the F1 run is not in this table: the judge's specific complaint was wrong, because every flagged sentence
is word for word in source [1], but the answer was truncated and counted as a failure. It was fixed as above.)

| | Step 11 baseline | R2a | **Final** |
|---|---|---|---|
| Dev set, automatic | 75% | 88% | **94% (69/73)** |
| Dev set, accepted | – | 93% (68/73) | **100% (73/73)** |
| **Held-out (15 questions, 23 runs)**, automatic | **100% (23/23)** | – | **91% (21/23)** |
| Held-out, accepted | 100% | – | **96% (22/23)** |
| False answers (every set, probes included) | 0 | 0 | **0** |

**Held-out runs.** Each setup was run once:
- **Baseline:** `…233318_rag_350_60_generation_holdout_final_baseline`, run from commit f8e820c in a separate
  worktree with the Step 11 settings. 4 calls hit xAI's rate limit and were rerun once with the same setup
  (`…233719_…retry`).
- **Final:** `…233903_rag_500_60_ctx_generation_holdout_final`.
- About $0.32 for both.

What they show:
- **On unseen questions, the final configuration shows no gain over the baseline** and lost one question. That is
  within the margin: one run is about 4 points of 23. The held-out questions were already easy for the baseline;
  the Step 12 gains are on the failures diagnosed on the dev set (the wording gap, truncation and the URL-filter
  follow-up).
- **h-p-ng-01 failed (a false refusal) and is a known limitation.** It was not fixed, because it was found on the
  held-out set.
  - **The question:** "My home IP address keeps changing. Can I reach my Netgear router from outside using a name
    instead?"
  - **What happened:** the first search missed the Dynamic DNS page; 500/60 ranks it below 5, while the baseline had
    it at rank 5. The fallback wrote the right searches ("Set up and manage Dynamic DNS", …), but the term check
    rejected all three for dropping "IP", which it treats as a protected term because of its two capitals.
  - **Proposed fix (future work):** treat common networking words (IP, DNS, LAN, WAN, MAC, USB) as ordinary words,
    then re-test that real feature names stay protected.

**Cost of this work** (F1, truncation fix, prompt 1.2, repeats, held-out): about $2.30.

## Exploratory tests (no changes made)

- **Negatively worded questions: 13/13 correct.** Examples: "I don't want the lights on my TP-Link to stay on at
  night", "Can't I reset my Netgear without the button?", "Is it not possible to not use WPS?".
  - Search found the same page as the positive wording, and the model resolved the negation.
  - Leading questions ("Doesn't the AX55 support 6 GHz?", "Isn't there an OpenVPN server?") were still refused.
  - One reply to "Is there no way…?" began "No, there is a way", which is logical but easy to misread.
- **Opposite actions on different pages: 5/5 correct.** Examples: remove from the blacklist vs add, separate vs unify
  bands, delete keywords vs block by keyword.
  - Search ranked the "opposite" section first in 4 of 5.
  - In 4 cases both procedures were in the same chunk, because manuals put "add X" and "remove X" side by side.
  - The one true test (Netgear keywords: the opposite at rank 1, the correct procedure at rank 2 in a separate
    chunk) was answered from the correct chunk.

## Known limitations

1. **The fallback only triggers on a refusal.** An answer built on wrong-but-similar pages does not trigger it. None
   was seen in any run.
2. **The term check is too strict for common abbreviations** ("IP" in h-p-ng-01 above) and **does not protect plain
   capitalised names** such as "Alexa". Only the prompt rule protects those, and the Alexa probes were refused.
3. **"Can it…?" questions** may be answered with the feature and no steps (rule 7). The user can then ask "how?".
4. **Loosely related features are not offered.** Prompt rule 8 is deliberately narrow, so such questions get the
   not-found reply.
5. **Refusals take longer** with the fallback, because of one more search and answer: about 1–4 s more, longer when
   xAI is slow.
6. **Icons in procedures are lost** in the PDF text ("click [icon] under Network Topology").
7. **The judge misreads** manuals that contradict themselves, and procedures split across chunks. Its "not grounded"
   verdicts are checked by hand.
