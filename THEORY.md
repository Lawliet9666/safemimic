# Mathematical formulation

The full trainable actor is pi_theta(o_history, reference, ball_relative). The
original SuperMimic goal encoder and action decoder and the added ball encoder
are all included in theta. Ball conditioning modifies the internal motion latent,
not an additive action residual. Actor observation/goal normalization is fixed.

Let s=(p,R,v,w,q,qd), a in R^29, and b=(p_ball,v_ball,r_ball). The action is a
scaled joint setpoint, not a torque. The actual target is
q_target=q_default+joint_calibration+D*a. The full physical robot is a
floating-base articulated G1 simulated at 5 ms, with 20 ms control intervals.

The learned closed-loop transition model is s_next=F_phi(s,a,xi). It predicts
local state increments and is nonlinear in actions. Phi and all model buffers
are fixed throughout policy training, but derivatives with respect to s and a
remain available. The simulator supplies transitions; it need not be differentiable.
The ball follows analytic ballistic dynamics B at the same control interval.

## Training-only barrier layer

Differentiable full-tree kinematics place 36 conservative collision spheres.
For each proxy, h_i(s,b)=||c_i(s)-p_ball||-R_i-r_ball-d. The candidate geometric
safe set is h_i>=0 for all i, with d=0.02 m. Velocity anticipation adds
psi_i=h_i+tau*w_i*hdot_i. Threat weights w_i use predicted closest approach.
This augmentation is a heuristic; independent geometric inequalities remain.
Balance/recovery are losses, not additional certified CBF constraints.

Stack H=(h,psi). Define the nonlinear discrete barrier residual
r(a)=H(F_phi(s,a,xi),B(b))-(1-alpha)*H(s,b).
At a local action a0, use ell(a)=r(a0)+J(a-a0), J=dr/da at a0.
The QP solves

    min_(a,delta) 0.5*||a-a_nom||^2 + rho*sum(delta) + 0.5*||delta||^2
    subject to ell(a)+delta >= residual_margin, delta >= 0,
               a inside hard action bounds and the local trust box.

J is computed with autograd. Training differentiates the QP solution through its
coefficients and every relinearization, including model-Jacobian dependence.
The solver is qpth with implicit differentiation. The identity ablation retains
the forward computation and replaces only the action backward by identity.
The original nonlinear residual is checked after solving. Slack or nonlinear
violations remain explicit failures; solver failure is not proof of infeasibility.
Unbounded nonnegative slack makes the relaxed problem feasible for finite
coefficients and a nonempty hard action box. Relaxation does not establish safety.

Successive local QPs are globalized with backtracking on the true nonlinear
merit `sum(relu(residual_margin-r(a))) + ||a-a_nom||^2/(2*rho)`.
The accepted interpolation supplies the next linearization point; the returned
action is a solved QP candidate, so its affine residual and slack remain aligned.
The layer preserves the best nonlinear-valid candidate independently per batch
member. Defaults allow 16 local solves, a 0.15 local action step and eight
backtracking halvings. The original failure tolerance remains 1e-4; the affine
margin is an approximation reserve, not a relaxed nonlinear safety condition.
If no batch member can improve its iterate, the search stops and retains the
failure flag instead of repeatedly solving the identical local problem.
Only the discrete backtracking/candidate choices are detached. The selected
QP and iterate retain coefficient/state/action derivatives. These derivatives
are piecewise derivatives of the executed numerical algorithm; changes of
active set or search branch need not be differentiable. The finite-difference
tests concern locally stable branches, not a global smoothness claim.

## Single-rollout policy learning

Training propagates one imagined trajectory:

    a_nom = pi_theta(observation_history, reference, ball_relative)
    a_qp = QP(s, b, a_nom; F_phi)
    s_next = F_phi(s, a_qp, xi)

The next policy input/history uses s_next and a_qp. There is no parallel nominal
trajectory. Gradients of the trajectory task loss pass through both F_phi and
the differentiable QP, updating the original actor and ball encoder together.

Let r_SM be the original eight-term SuperMimic reward, including its 20 ms
control-period factor. Each reward compares the predicted next state against
the next reference frame. Original reward weights are retained, without threat
gating. Let valid_t mean that the QP action passes affine, slack and original
nonlinear checks (under the learned model, not a physical certificate).

    C_t = valid_t * mean_j (a_nom_j-stop_gradient(a_qp_j))^2
    S_t = mean_i positive_part(delta_i)^2
    r_aug = r_SM - dt*(correction_weight*C_t + slack_weight*S_t)
    c_0 = 1; c_(t+1) = c_t*(1-soft_termination(s_(t+1)))
    L_actor = -E[sum_t gamma^t*c_t*r_aug_(t+1)
                 + gamma^H*c_H*V_psi(s_H, reference_H, xi, ball_H)].

The only added actor losses are QP correction and slack. They use per-action
and per-constraint means and the same control-period, discount and survival
weights as the task return. A failed/relaxed action is not a correction teacher;
finite failed trajectories still contribute task and slack gradients. Report
correction-teacher coverage so a masked zero is not mistaken for internalized
safety. Nonlinear violations and action saturation are diagnostic metrics, not
additional loss terms. The QP's internal slack coefficient rho=1e4 differs from
the outer slack-loss weight (initially 10).

Only the correction target is detached. Task, QP coefficients, slack, rollout
states and the terminal critic's state inputs retain actor gradients. Critic
parameters are fixed during actor backpropagation, then fit separately to
detached lambda-return targets of this same augmented reward. Critic features
reuse the original privileged state/reference/context features and append the
same six relative ball features as the actor. Critic weights are initialized
afresh for this task. No pretrained critic for the new task is assumed.

The reward-only baseline uses one unfiltered rollout with the original task
reward plus explicitly labeled geometric avoidance shaping; the proposed method
has no geometric avoidance reward. The identity baseline retains the proposed
method's complete forward calculation and losses.

Simulator collection and evaluation execute nominal actor actions with ordinary
actuator clipping. Deployment has no QP, dynamics rollout, or world-model weight
dependency. A QP-filtered training objective does not guarantee safety after the
QP is removed. Correction loss encourages internalization, whose effectiveness
must be assessed on held-out nominal-policy evaluations.

The learned-model constraints require model error, linearization, intersample,
solver and feasibility bounds before a conditional physical safety theorem
applies. Such certified bounds have not been established here.
