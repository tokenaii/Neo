# Neo Training History

The training process emits a JSON record every 100 optimizer steps containing:

- epoch;
- optimizer step;
- training loss.

The authoritative live history is maintained on AWS while training is active:

`/mnt/opet-data/neo/logs/neo-english-train.log`

It is not yet copied into the public repository because the run is still in
progress. After completion, the final history will be frozen together with the
checkpoint hash and the evaluation artifacts.

## Evaluation records

The automatic post-training stage will write these local artifacts:

- `benchmark.json`: held-out decision metrics by decision type and tool routing;
- `latency.json`: batch latency, decision latency, and throughput;
- `router_smoke.json`: typed output from a tool-router integration check;
- `completed_at.txt`: evaluation completion time.

The results will remain private until they are reviewed and explicitly approved
for publication.

## Current status

The active run has reached step 7,000 of 15,640. The latest observed loss is
`1.6072139739990234`. This is an interim training value, not a final quality
result.
