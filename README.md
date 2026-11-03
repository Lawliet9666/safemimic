# SafetyMimic

Full-policy learning for G1 whole-body dodgeball. Initialize the original
SuperMimic actor from its pretrained checkpoint, add a ball encoder to its
motion latent, and train the original motion encoder, action decoder, and ball
encoder together. The ball branch is zero initialized, preserving pretrained
actions at update zero. There is no action-residual network.

Training differentiates through a CBF-QP and a frozen learned world model.
**Simulator collection, evaluation, and deployment never execute the QP.**
Deployment loads only the complete actor checkpoint; world-model weights and
pretrained actor files are unnecessary after fine-tuning.

## Architecture

- The tracker receives 10 x 99 robot observations and 10 x 73 future reference
  features. Its original goal encoder and action decoder remain trainable.
- The ball encoder receives the ball's relative position and relative linear
  velocity in the pelvis frame (6 values). Its 32 outputs are added to the
  motion encoder output before the latent tanh. It does not add an action.
- The actor produces 29 scaled joint-position commands. Ordinary joint-target
  bounds apply at collection and execution; these are not a CBF-QP filter.
- The dynamics model predicts G1 state increments at 20 ms. Its parameters and
  normalization buffers stay fixed throughout actor training. State/action
  derivatives remain enabled.
- Training uses exactly one imagined rollout:
  **actor -> differentiable CBF-QP -> fixed world model -> task loss**.
  Task gradients pass through both the QP and dynamics into the entire actor.
  On this same rollout, an action-correction loss encourages the raw policy to
  imitate validated QP corrections, and a slack loss retains its gradients.
  These are the only two added loss terms. There is no separate nominal rollout.
- The task reward directly reuses SuperMimic's eight motion-tracking terms,
  original weights, control-period scaling, discount, and soft survival weights.
  A ball-conditioned terminal critic bootstraps the same augmented return and
  is fit separately to detached lambda-return targets. Its weights are frozen
  during actor backpropagation; its state derivatives remain enabled.


The simulator currently supplies privileged robot and ball state. Perception is
outside this implementation. The geometric and velocity-augmented barriers are
local learned-model approximations, not a certified physical safety guarantee.
The dynamics weights remain frozen throughout fine-tuning, unlike the original
algorithm's alternating dynamics updates. The task uses a standing reference,
not the original large motion library. This is not a full reproduction of the
original experiments or an established speedup result.

## Runtime

Use the existing Docker image `safetymimic:46d6cc3-cu12`. The wrapper restricts
all containers to **GPU 0 and CPUs 0-5**, with a six-core quota and BLAS thread
limits. Python entrypoints enforce the same CPU affinity and GPU choice. No
local virtual environment is used for training or evaluation.

```bash
cd /home/daiyp/CODE/safetymimic
mkdir -p outputs/logs
bash scripts/container.sh python -m pytest -q
```

Pinned dependencies and original assets remain in `Dockerfile`,
`source_pins.json`, `asset_pins.json`, and `vendor/supermimic`.

## Train

```bash
bash scripts/container.sh python scripts/train.py \
  --pretrained checkpoints/g1-lafan1-040000 \
  --world_model outputs/checkpoints/world_model_broad.pt \
  --updates 300 --horizon 32 --num_envs 8 --batch_size 4 \
  --output outputs/runs/safetymimic_seed0
```

A bounded integration check uses `--updates 2 --horizon 2 --num_envs 2
--batch_size 1 --linearization_steps 1` and a fresh output directory. It checks
computation, not dodgeball performance. Training directories cannot be reused or
silently overwritten.

`policy.pt` and `policy_XXXXXX.pt` contain the entire actor, normalization buffers,
architecture, configuration, and training metadata. Separate `critic.pt` and
`critic_XXXXXX.pt` files hold training-only critic weights and optimizer state;
deployment does not load them. Only
`safetymimic.full_policy.v1` checkpoints are accepted. Old residual checkpoints
are intentionally unsupported. `checks.json` records updates to all three
parameter groups, unchanged world-model weights, absent dynamics gradients, and
unchanged actor normalization. `metrics.jsonl` includes task, correction, slack,
critic fitting loss, correction-teacher coverage, nonlinear violation telemetry,
QP failures, gradient norms, and actor/critic timing. The objective marker is
`safetymimic.supermimic_correction_slack.v1`, with `imagined_rollouts=1` and
`terminal_critic=true`. Initial auxiliary weights are experimental settings,
not validated performance choices.

For full-actor baselines, use `--method reward` (original reward plus explicit
geometric avoidance shaping, without QP) or `--method identity` (same QP forward and losses, identity
surrogate action backward). The proposed method is `--method safetymimic`.
Every method deploys without a QP. The identity surrogate is not DYS-JFB.

## Evaluate and render

Reuse the explicit existing manifests, or create new development/validation
sets in a new directory with `scripts/make_eval_manifests.py --output ...`.

```bash
bash scripts/container.sh python scripts/evaluate_safety.py \
  --manifest outputs/manifests/development.json \
  --checkpoint outputs/runs/safetymimic_seed0/policy.pt \
  --label safetymimic_seed0 \
  --output outputs/runs/eval_safetymimic_seed0
```

Omit `--checkpoint` to evaluate the pretrained SuperMimic baseline. There is no
filter-enable flag and no world-model argument. All requested episodes retain
physical contact events at 5 ms and control diagnostics at 20 ms.

```bash
bash scripts/container.sh python scripts/render_evaluation.py \
  outputs/runs/eval_safetymimic_seed0 \
  --output outputs/results/full_policy/videos/safetymimic_seed0
bash scripts/container.sh python scripts/report.py \
  --runs outputs/runs/eval_safetymimic_seed0 \
  --training outputs/runs/safetymimic_seed0 \
  --output outputs/results/full_policy
```

The report contains task success, comparison tables, episode evidence, linked
videos, and training diagnostics. It rejects the previous residual experiment
schema rather than mixing old and new methods. See `EVALUATION.md` and `THEORY.md`.

## Source layout

- `src/safety_mimic/policy.py`: complete trainable actor, ball features, checkpoints.
- `src/safety_mimic/training.py`: single QP-guided BPTT objective.
- `src/safety_mimic/experiment.py`: policy-only controller and actuator bounds.
- `src/safety_mimic/layer.py`, `qp.py`: differentiable local CBF-QP for training.
- `src/safety_mimic/geometry.py`, `ball.py`: kinematics, barriers, ball dynamics.
- `src/safety_mimic/isaac_env.py`: simulator and physical contact instrumentation.
- `scripts/train.py`, `evaluate_safety.py`, `render_evaluation.py`, `report.py`:
  separate training, execution, video, and reporting entrypoints.
- `scripts/fit_world_model.py`: ordinary supervised dynamics preparation.
- `tests/`: initialization, gradients, checkpoint/deployment contract, QP, statistics.
- `outputs/{checkpoints,runs,logs,manifests,results}`: all generated artifacts.

Historical outputs remain untouched under `outputs/`, including the old report
`outputs/results/evaluation_20261102/index.html`. Those results describe the old
frozen-tracker/residual prototype. Earlier full-policy smoke/timing runs used a
dual-rollout objective; they also do not evaluate this single-rollout objective.
Previous timing estimates must be remeasured before budgeting a long run.
Earlier single-rollout runs with handcrafted task losses and no terminal critic
are also historical; the current report rejects their objective metadata.
