# Generated Skill Banks

The offline AlfWorld generator writes its default outputs here:

- `<dataset>_<model>_<count>.json`: hierarchical task and step skills loaded by training.
- `<dataset>_<model>_<count>.audit.json`: reflection prompts, raw responses, and generation statistics.

`count` is the actual number of generated task bundles, for example
`alfworld_gpt-5.5-2026-04-24_287.json`.

Both paths can be overridden with `--output` and `--audit-output`.
