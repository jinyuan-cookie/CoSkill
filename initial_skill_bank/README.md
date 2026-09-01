# Initial Skill Bank

This directory contains the offline ALFWorld and WebShop pipelines used to
create the hierarchical skill JSON loaded before reinforcement learning starts.
Each task still collects all configured rollouts for success-rate statistics,
but Reflection receives representative evidence: one successful trajectory for
an all-success group, one successful plus one failed trajectory for a mixed
group, or one failed trajectory for an all-failure group. All three outcome
types are eligible for initialization, and sampling is reproducible from
`--seed`.

```bash
export OPENAI_API_KEY=...

python3 initial_skill_bank/generate_alfworld_initial_skill_bank.py \
  --api-base-url https://your-api.example/v1 \
  --model your-model \
  --target-task-bundles 300
```

The command writes:

- `skill_bank/<dataset>_<model>_<count>.json`: task skills and linked child step skills used by training. `count` is the actual number of generated task bundles.
- `skill_bank/<dataset>_<model>_<count>.audit.json`: reflection prompts, responses, and generation statistics.

For example, 287 bundles generated on AlfWorld with
`gpt-5.5-2026-04-24` produce `alfworld_gpt-5.5-2026-04-24_287.json`.
Model names containing `/` or spaces are converted to filename-safe `-` characters.
Passing `--output` continues to use the exact path supplied by the caller.

Pass the generated bank to training with:

```bash
INITIAL_SKILL_BANK_PATH=initial_skill_bank/skill_bank/alfworld_gpt-5.5-2026-04-24_287.json \
  ./examples_coskill/run_alfworld_coskill.sh vllm
```

The CoSkill ALFWorld launcher defaults to the bundled 300-task Skill Bank. Set
`INITIAL_SKILL_BANK_PATH` explicitly when using a newly generated file.

The bundled collection launcher reads its API URL and model from
`SOPHON_API_URL` and `SOPHON_MODEL`. Store these values in the ignored `env.sh`
file or export them before running. If `SOPHON_API_KEY` is not set, the
launcher asks for it using hidden terminal input:

```bash
./initial_skill_bank/run_generate_alfworld_initial_skill_bank.sh
```

For non-interactive jobs, provide it through the environment:

```bash
SOPHON_API_KEY='<your-api-key>' \
  ./initial_skill_bank/run_generate_alfworld_initial_skill_bank.sh
```

No concrete API endpoint or credential is stored in the launcher; configure
the endpoint and model explicitly. The launcher collects one task group per
batch, eight rollouts per task, and allows at most eight concurrent API calls.

Reasoning and Reflection share a global request-rate limiter. It defaults to
60 request starts per minute and counts retries. Override it with either the
environment variable or the CLI argument:

```bash
MAX_API_REQUESTS_PER_MINUTE=30 \
  ./initial_skill_bank/run_generate_alfworld_initial_skill_bank.sh

./initial_skill_bank/run_generate_alfworld_initial_skill_bank.sh \
  --max-api-requests-per-minute 30
```

## WebShop

WebShop uses the same task/step hierarchy and Reflection schema. Each sampled
WebShop goal is repeated for all configured rollouts and grouped by its stable
goal index. Full success is defined by the environment's terminal
`task_score == 1.0`; the original `task_score` is also retained in trajectory
audit data. Reflection still receives one success for an all-success group,
one success plus one failure for a mixed group, or one failure for an
all-failure group.

Verify that the WebShop data and Lucene index are installed before collection:

```bash
PYTHONPATH=. python3 agent_system/environments/env_package/webshop/verify_webshop_env.py
```

Then run the bundled collection launcher:

```bash
./initial_skill_bank/run_generate_webshop_initial_skill_bank.sh
```

By default it uses the 1k-product files, collects eight rollouts for one goal
group at a time, allows eight concurrent API calls, and caps each episode at 15
steps. Use the full WebShop files with:

```bash
WEBSHOP_USE_FULL=1 \
  ./initial_skill_bank/run_generate_webshop_initial_skill_bank.sh
```

Machine-specific data locations can be supplied without editing the script:

```bash
WEBSHOP_ITEMS_PATH=/path/to/items_shuffle_1000.json \
WEBSHOP_ATTRIBUTES_PATH=/path/to/items_ins_v2_1000.json \
  ./initial_skill_bank/run_generate_webshop_initial_skill_bank.sh
```

The generated files follow the same convention as AlfWorld:

- `skill_bank/webshop_<model>_<count>.json`
- `skill_bank/webshop_<model>_<count>.audit.json`
