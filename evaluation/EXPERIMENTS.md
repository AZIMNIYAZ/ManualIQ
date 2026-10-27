# Step 12 experiments

Every experiment is compared with the baseline (`evaluation/BASELINE.md`) on the development
set (dataset `4d4be1280743`: 40 questions + 6 conversations) with `scripts/eval_compare.py`. The
held-out set (15 questions, `f2b8451e9083`) is frozen and is only run on the baseline and on the
final configuration.

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
