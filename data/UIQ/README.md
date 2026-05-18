# User-Intent Queries (UIQ) benchmark

This directory contains the released UIQ benchmark introduced in **Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval** (ACL 2026 — Oral) by HaeJun Yoo, Yongseop Shin, Insung Lee, Myoung-Wan Koo, and Du-Seong Chang (Sogang University).

UIQ tests retrieval robustness beyond caption-style queries by reformulating each evaluation caption into **five** user-intent–oriented variants:

| Category               | Query type     | Example                                                                            |
|------------------------|----------------|------------------------------------------------------------------------------------|
| Conversational         | `question`     | *"Can you find clear dog barks echoing in a large hall?"*                          |
| Conversational         | `imperative`   | *"Find crisp footsteps on gravel with light echo."*                                |
| Reformulation          | `tagging`      | *"dog barks, echoing hall, reverberant"*                                           |
| Reformulation          | `paraphrase`   | *"Echoing dog barks resonate through a large empty hall."*                         |
| Exclusionary           | `negative`     | *"Heavy rain and wind on metal surfaces without thunder or engine noise."*         |

Positive queries (question / imperative / paraphrase / tagging) preserve the audio content and stay within ±2 words of the original caption length. Each `negative` query targets a clip and explicitly excludes content from a mined hard-negative clip — see §3.2 of the paper.

## Files

```
audiocaps/  audiocaps_test_{question,imperative,paraphrase,tagging,negative}_queries.jsonl
clotho/     clotho_evaluation_{question,imperative,paraphrase,tagging,negative}_queries.jsonl
mecat/      mecat_{question,imperative,paraphrase,tagging,negative}_queries.jsonl
```

| Dataset              | question | imperative | paraphrase | tagging | negative | **Total** |
|----------------------|---------:|-----------:|-----------:|--------:|---------:|----------:|
| AudioCaps (test)     | 975      | 975        | 975        | 975     | 630      | 4 530     |
| Clotho (evaluation)  | 1 045    | 1 045      | 1 045      | 1 045   | 542      | 4 722     |
| MECAT                | 848      | 848        | 848        | 848     | 409      | 3 801     |
| **All**              | 2 868    | 2 868      | 2 868      | 2 868   | 1 581    | **13 053**|

## JSONL schema (positive queries)

```jsonc
{
  "audio_id":          "Santa Motor.wav",                  // identifier in source dataset
  "dataset":           "clotho",
  "dataset_slug":      "clotho_evaluation",
  "query_type":        "question",                          // one of: question | imperative | paraphrase | tagging | negative
  "generated_query":   "Can you find audio of a machine whining and squealing while stamping?",
  "original_captions": ["...", "..."],                      // captions used as positive grounding
  "metadata":          {"split": "evaluation", "num_captions": 5},
  "source_model":      "gpt-5.1",
  "regen_model":       "gpt-5.1"
}
```

`negative` queries follow the same schema but additionally reference a pre-mined hard-negative clip; see the paper (§3.2.2) and `AudioRetrieval/uiq_toolkit/` for the mining and generation pipeline.

## How the queries were produced

- Generator: GPT-5.1 with vocabulary-grounding constraints (queries must reuse caption wording).
- Length control: outputs stay within ±2 words of the original caption length to avoid length-induced distribution shift.
- Hard negatives (for `negative` queries): mined via the 4-stage pipeline in `AudioRetrieval/scripts/mine_hard_negatives_laion.py`.
- Quality: human spot-checked; mean rating 4.15/5.0 (9 annotators, 675 ratings). See Table 1 of the paper.

## Audio is not redistributed

UIQ ships **only** the generated text queries. Source audio must be obtained from the original benchmarks under their respective licenses:
- AudioCaps — <https://audiocaps.github.io/>
- Clotho — <https://zenodo.org/records/4783391>
- MECAT — see the MECAT release page.

## License

The UIQ queries in this directory are released under **CC BY 4.0** (see [`LICENSE`](LICENSE)). Please cite the paper when using or redistributing them.
