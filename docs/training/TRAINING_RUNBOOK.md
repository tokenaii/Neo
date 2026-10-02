# Neo Training Runbook

1. Run the CPU smoke test using a small JSONL slice.
2. Run the GPU preflight and confirm that the shared H-Preview service is
   intentionally paused for the handoff.
3. Generate and validate the synthetic JSONL corpus.
4. Record the manifest hash, generator seed, configuration hash, and runtime
   versions.
5. Start training with checkpoints and append-only logs under `runs/` and
   `logs/`, preferably through `scripts/train_and_evaluate.sh`.
6. Evaluate only after training: accuracy, NLL, Brier, ECE, selective
   accuracy, abstention, latency, and option-order invariance.
7. Fit calibration temperature on the calibration split only.
8. Verify checkpoint hashes and model-card provenance. The automatic post-run
   stage writes benchmark, latency, and tool-router smoke-test artifacts; do
   not publish those artifacts without approval.
9. Upload the model to `tokenaii/neo` with `MODEL_LICENSE.md` and the canonical
   `tokenaii/neo` reference.
10. Restore H-Preview and its watchdog after Neo's GPU workload is stopped.
