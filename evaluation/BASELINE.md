# Baseline evaluation (Step 11)

The reference point for every Step 12 experiment. Run on 2026-10-16.

| Setting | Value |
|---|---|
| Dataset | `4d4be1280743`: 40 questions + 6 conversations (13 turns); labels reviewed against the PDFs |
| Collection | `rag_350_60` (chunks of 350 tokens, 60 overlap), cleaner 1.3, `BAAI/bge-small-en-v1.5` |
| Retrieval | top 5 chunks, context budget 2,500 tokens, regulatory pages excluded |
| Answer model | `grok-4.3`, temperature 0.1, prompt 1.1 |
| Rewrite model | `grok-4.20-0309-non-reasoning`, rewrite prompt 1.1 |
| Judge | `grok-4.20-0309-non-reasoning`, temperature 0, judge prompt 1.0 |

Raw runs, which Git ignores: `evaluation/results/20261016-012320_rag_350_60_retrieval_baseline` and
`evaluation/results/20261016-012345_rag_350_60_generation_baseline`.

## Headline numbers

| Metric | Result |
|---|---|
| **Accuracy, all answers** (questions incl. repeats + conversation turns) | **75%** (55/73) |
| Accuracy, the 40 questions (first run) | 78% (31/40) |
| **Retrieval recall, Hit@5** (right passage among the 5 chunks the model sees) | **78%** (25/32); Hit@10 94%; MRR 0.64 |
| **Answer recall** (answerable items answered correctly) | 72% (46/64) |
| **Refusal recall** (should-refuse items refused) | **100%** (9/9) |
| False answers (answered when the manuals do not have the answer) | **0** |
| Brand filter correct | 100% |
| Regulatory chunks in results | 0 |

"Correct" means one of two things:
- **Answerable items:** the system answered, every key fact is present, there's no wrong-procedure text, and the judge found it grounded.
- **Unanswerable items:** the system refused.

With 40 questions, each question moves the accuracy by about 2.5 points, so read the figures as approximate.

## By category

| Category | Runs | Pass | Hit@5 | Notes |
|---|---|---|---|---|
| Direct | 8 | 75% | 88% | |
| Paraphrased | 9 | **44%** | **56%** | weakest: the user's wording differs from the manual's |
| No brand named | 5 | 100% | 100% | always answered per brand and never mixed; covers only the brands in the top 5 (2–5 of 5) |
| Tables | 15 (5 × 3) | 80% | 80% | Tenda default IP failed in all 3 runs (retrieval) |
| Context confusion | 15 (5 × 3) | 80% | 80% | 0 wrong-procedure text; ASUS Network Service filter failed in all 3 runs (retrieval) |
| Unanswerable | 5 | 100% | – | all refused, including the tempting OpenWrt and ASUS OpenVPN questions |
| Out of scope | 3 | 100% | – | |
| Conversations | 13 turns | 62% | – | rewrite brand 7/7, required terms 7/7; failures: retrieval 2, generation 2, rewrite 1 |

**Repeats:** the 10 table and context-confusion questions ran 3 times each. Every one gave the same outcome in all 3 runs: 8 passed 3/3 and 2 failed 3/3. The answers were stable at temperature 0.1.

## Main finding

**The system does not make things up, but it refuses too often, and retrieval causes the refusals.**
- 14 of the 64 answerable items were refused. In 13 of them, no chunk in the top 5 matched the evidence.
- The exception is cv-02-2 ("Can it also block a whole service like FTP?"). The evidence was in the context, but the rewrite asked the question literally ("Can a URL filter block a service?"), and the model refused.
- Of the 50 items whose evidence was in the top 5, 49 were answered. p-te-01 was answered from an unlabelled but valid section (see Label notes).

The retrieval misses share these patterns:
- **Generic pages outrank specific ones.** For example, D-Link Troubleshooting (p. 138) and the ASUS setup/login pages (pp. 11–12, 18–19) are top results for many unrelated questions.
- **Paraphrases fail.** "kick it off for good" does not retrieve the blacklist page, and "keep everyone else out" does not retrieve the MAC filter page.
- **Flattened tables fail.** The Tenda factory-settings table is not retrieved for "default login IP".
- **Steps get cut off at chunk boundaries.** For TP-Link port forwarding, steps 5–7 are in the next chunk; the answer said that the steps were incomplete.

These failures are the targets for Step 12: contextual headers, hybrid keyword search, retrieving neighbouring chunks, chunk size/overlap, and top-k.

## Is the judge trustworthy?

| Test | Result |
|---|---|
| Human grades (20 answers, mixed categories) | groundedness agreement 90%, citations 95% |
| Planted errors (12 real answers with one error each, made by code) | **12/12 caught**: wrong menu path 3/3, changed value 3/3, step from another manual 2/2, invented claim 2/2, wrong citation 2/2 |
| Groundedness, human grades + planted errors (30 answers) | agreement 93%, **Cohen's kappa 0.86** |

- **Why the planted errors were needed:** the human sample alone gave a kappa of about 0. Almost every real answer was grounded, so a judge that always says "grounded" would also have agreed. The planted errors supply the missing negative cases.
- **Known judge error:** the Netgear factory reset answer. The manual itself says 5 seconds (p. 109) and 7 seconds (p. 158), and the answer quoted both with the right citations. The judge treated the conflicting sources as unsupported, and the human grade was "grounded".
- **Not validated:** the judge's **brand-mixing flag**. It stayed false on both answers with another manual's step, although it marked them as not grounded. Those steps did not name their brand and that brand's manual was not in the context, so the test could not show brand mixing as such. "0 brand mixing" therefore rests on the groundedness verdicts, the wrong-procedure text checks and the human reading, not on this flag.
- **Limitation:** the judge is from the same model family as the answer model.

One human grade (cv-03-2, "not grounded") was based on retrieval rank rather than on the grading rubric. It is kept as given; it lowers the measured agreement slightly.

## Label notes

- **Corrected during evaluation:**
  - nb-05: Netgear also describes turning off the LEDs (p. 106).
  - t-ng-01: the DHCP range is also on pp. 60 and 151, not only in the table on p. 158.

  Both were found by the retrieval run, before the final dataset was fixed.
- **Kept as reviewed, but worth knowing:**
  - p-te-01: the answer used the MAC Address Filter blacklist (pp. 118–121), a second valid method that is not labelled. That answer was incomplete anyway.
  - p-dl-01: the key fact "D-Link Cloud Service" is strict; the answer gave correct steps without that prerequisite.
- **Dropped:** 5 questions were removed to reach 40, because they duplicated conversation turns or were trivially out of scope (d-te-01, d-as-01, p-ng-02, o-04, o-05).
- **Absence:** the unanswerable questions' terms had 0 hits in the raw and cleaned text of the listed manuals, and the PDFs were reviewed. An answer that exists only in an image cannot be ruled out, since the system is text-only.
- **Citations:** a citation shows the page range of the whole chunk (e.g. pp. 124–128), so it can be wider than the page that actually holds the answer.

## Usage

The baseline used about 262k input and 36k output/reasoning tokens:

| Calls | Input tokens | Output tokens |
|---|---|---|
| Answers | 146k | 9k + 23k reasoning |
| Judge | 112k | 4k |
| Rewrites | 4k | 0.2k |

Median generation time was 5.1 s.
