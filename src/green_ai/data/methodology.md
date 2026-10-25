# Methodology

Green AI estimates the electricity, greenhouse gas emissions (GHG), water, primary
energy and abiotic resource depletion of **text inference** with large language
models. It works at three levels: a single request, a usage log, and the average
work needed to complete one task. All results are model based estimates with
explicit assumptions, not metered measurements.

The calculation has three layers:

1. **Base model.** Energy and impacts of generating the output tokens.
2. **Input correction.** GPU energy of processing the prompt (prefill), which the
   base model does not include.
3. **Per task footprint.** Both layers applied to the tokens and calls a model
   needs to finish one task.

## 1. Base model

The base model follows the inference methodology published by EcoLogits
(Rincé and Banse, JOSS 2025) and uses its open model and electricity dataset,
currently version **0.11.3**. The test suite checks it against the EcoLogits
reference code for every registered model (see [VALIDATION.md](VALIDATION.md)).

| Input | Meaning | Source |
|---|---|---|
| `output_tokens` | generated tokens, including reasoning | API usage field or log |
| `active_B`, `total_B` | active and total parameters, in billions | model registry, ranges for undisclosed models |
| `tps`, `TTFT` | provider throughput and time to first token | model registry |
| `B` | concurrent requests per server (batch size) | 64 by default |
| PUE, WUE | datacenter power and water usage effectiveness | provider defaults, overridable |
| grid factors | kgCO2eq, MJ, kgSbeq and litres of water per kWh | per country, 215 zones |

For each request, with 80 GB GPUs and 16 bit weights:

```
GPU energy per output token  = (1.1665e-6 * exp(-0.011206 * B) * active_B + 4.0529e-5) / 1000   kWh
GPU count                    = next power of two above ceil(1.2 * total_B * 16 / 8 / 80)
Generation time              = output_tokens / tps + TTFT
Host server energy           = time / 3600 * 1.2 kW * (GPU count / 8) / B
Facility energy              = PUE * (GPU count * output_tokens * GPU energy per token + host energy)
Operational GHG              = facility energy * grid kgCO2eq/kWh
Water                        = IT energy * (WUE + PUE * grid litres/kWh)
Embodied impacts             = (GPU + host manufacturing impacts) * time / (3 years * B)
```

When a model has no throughput data, generation time comes from a regression on
model size and batch size. A measured request latency can be passed to cap it.

Provider defaults describe a reference deployment, not the facility that serves
a given request. Mapping an Azure model name to its OpenAI architecture does not
say where the Azure deployment runs. Grid, PUE and WUE can be overridden
whenever better information is available.

## 2. Input correction

The base model has no term for prompt length. Reading the prompt (prefill) is
compute bound and processes all prompt tokens in parallel, so one prompt token
costs a small, roughly constant fraction `alpha` of the GPU energy of one
generated token. Green AI adds that energy:

```
equivalent tokens = alpha * (fresh_input_tokens + cache_factor * cached_input_tokens)
extra GPU energy  = equivalent tokens * GPU count * GPU energy per output token
extra facility    = PUE * extra GPU energy
```

The added energy is converted to GHG, water, primary energy and ADPe with the
same grid factors as the base model.

**Scope.** `alpha` is a ratio of GPU energy measured per phase. It is therefore
applied to GPU energy only, with PUE applied once. Host server energy, generation
time and embodied allocation stay as in the base model, because a GPU energy
ratio says nothing about extra host time or hardware occupancy.

**Coefficients.**

| Preset | alpha | Derivation |
|---|---:|---|
| `none` | 0 | base model only |
| `low` | 0.005 | mean of 20 published model and workload results (Solovyeva et al.) |
| `central` | 0.008 | median of 10 model and precision groups, 99,990 requests on H100 (Delavande et al.) |
| custom | any | user supplied |

The central value pools prefill and decode GPU energy within each group,
`(prefill kWh / input tokens) / (decode kWh / output tokens)`, and takes the
median across groups (0.798%). The low value averages the same ratio over the
published table rows (0.489%). `python scripts/reproduce_evidence.py` recomputes
both from the aggregates bundled in `src/green_ai/data/prefill_evidence.json`.
The original measurements belong to the cited authors.

**Caching.** `input_tokens` always includes cache reads. Cached tokens are
weighted by `cache_factor`, 0.1 by default, which follows how several providers
price cache reads relative to fresh input. Cache writes count as fresh input.
Set the factor to 1 to remove the discount.

**Effect size.** For a prompt `r` times longer than the answer, the correction
adds roughly `alpha * r` times the GPU energy of the answer. At `alpha = 0.008`
that is about +8% at 10:1, +40% at 50:1 and +80% at 100:1 of the GPU part, less
of the total because host energy is unchanged.

## 3. Per task footprint

A per token figure says how efficient a model is per token, not how much energy
it needs to finish a job. Reasoning models and multi step agents can use many
more tokens for the same task.

A **task profile** holds the average input, cached and output tokens per task
(output includes reasoning) and the average number of calls per task. The
footprint per task is the full model above evaluated on those averages, with the
fixed per request overhead counted once per call. Impacts are linear in tokens,
so this equals the average footprint over the tasks.

Rankings can sort by Wh, gCO2eq or mL of water per task, or by quality score per
Wh, with an optional minimum score. They use a common grid (world average by
default) so they compare models rather than locations. Each row also shows the
rank the model would get on energy per 1,000 generated tokens, which makes the
difference between the two views visible. Grades A to E and the quality versus
impact Pareto set are relative to the compared set and use interval midpoints.

**Sources of task profiles.**

* **Own traces.** Averages measured on one's own workload, as a CSV. The most
  relevant source for a decision about that workload.
* **Artificial Analysis per task exports.** Input, output, answer and reasoning
  tokens per task from an AA export, imported as observed values. If only output
  tokens are available, input is set from a chosen input to output ratio range
  and marked as assumed.
* **Artificial Analysis API.** The model list gives cost per task (a weighted
  average over benchmarks) and list prices. Aggregate token counts divided by
  `total_cost / cost_per_task` would not give tokens per task, because that ratio
  is not a task count. Tokens are instead derived from cost under an assumed
  output share `s` of the cost:

  ```
  output tokens per task = s * cost_per_task / output price per token
  input tokens per task  = (1 - s) * cost_per_task / input price per token
  ```

  Both ends of the range for `s` (0.5 to 0.9 by default) are evaluated, and the
  rows are labelled `cost-inferred`.
* **Artificial Analysis Pro evaluation counts.** With a versioned plan of
  benchmark weights and task counts, tokens per task are
  `sum(weight * evaluation tokens / evaluation tasks)`. Plans must cover every
  weighted benchmark and match the index version.

Details and examples: [AA_INTEGRATION.md](AA_INTEGRATION.md).

## 4. Uncertainty

Undisclosed model sizes, PUE and WUE are carried as intervals. Every formula is
increasing in its inputs, so each result is evaluated at the low and high end of
its inputs and reported as a range with its midpoint. `alpha` is handled as a
scenario rather than an interval, so its effect is shown by comparing presets.

## 5. Limitations

* Estimates, not measurements. No result has been compared with metered energy of
  a production deployment.
* Closed model sizes are estimates. They drive most of the base model range.
* The GPU regression was fitted on one accelerator generation (H100).
* Provider datacenter figures are public averages. Grid factors are annual
  country averages, not hourly or marginal values.
* `alpha` comes from research measurements at specific batch sizes and hardware.
  Transfer to production serving (large dynamic batches, chunked or
  disaggregated prefill, quantization) is not established. Very long contexts,
  where attention grows faster than linearly, may exceed the linear estimate.
* `cache_factor` follows pricing, not energy measurements.
* Benchmark tasks are not production tasks, and a completed benchmark attempt is
  not always a successful one. Scores are only compared within one task set and
  index version.
* Networking, storage, training and idle capacity are out of scope.

## References

* Rincé, S. and Banse, A. (2025). EcoLogits: Evaluating the Environmental Impacts
  of Generative AI. *JOSS* 10(111), 7471. [doi:10.21105/joss.07471](https://doi.org/10.21105/joss.07471). Data under MPL-2.0.
* Delavande, J., Pierrard, R. and Luccioni, S. (2026). Small Talk, Big Impact: The
  Energy Cost of Thanking AI. [arXiv:2601.22357](https://arxiv.org/abs/2601.22357).
  [Dataset](https://huggingface.co/datasets/Anon152425/energy_dtype_all_models), revision `d0e7e7b3`.
* Solovyeva et al. (2026). Towards Green AI: Decoding the Energy of LLM Inference
  in Software Development. [arXiv:2602.05712](https://arxiv.org/abs/2602.05712), tables 1 to 4.
* Artificial Analysis. [API reference](https://artificialanalysis.ai/data-api/docs/reference) and
  [benchmark methodology](https://artificialanalysis.ai/methodology).
