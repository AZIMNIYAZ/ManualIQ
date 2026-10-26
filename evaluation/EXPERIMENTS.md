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
| R2a | **500/60 + contextual headers** (vs R1c) | **84%** | **94%** | 94% | **0.88** | **+2 / −0** vs R1c | pending | pending | **kept**: best so far; answer run to confirm |
| R2b | 350/60 + contextual headers (vs baseline) | 66% | 94% | 94% | 0.77 | +6 / −0 vs baseline | – | – | confirms R2a: the header effect does not depend on chunk size |

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
