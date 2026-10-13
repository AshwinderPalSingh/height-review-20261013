"""Finite-horizon rollout safety filter (shield) for a frozen HEIGHT-style policy.

WHAT THIS IS. An inference-only layer. The policy proposes one of the 9 discrete
(delta_v, delta_w) actions; the filter rolls each action forward for `horizon` steps under
commanded unicycle kinematics (no wheel saturation, no arena wall) and, if the proposed
action is predicted to reach contact (clearance below zero), substitutes the action with
the largest predicted clearance, ties broken toward the proposal; if no action is strictly
safer it keeps the proposal. `margin` only stops a rollout early once it is clearly unsafe.
Weights are never touched. The file name is historical: the filter is in the spirit of a
discrete control barrier function, but it checks clearance along short rollouts and
enforces no barrier condition, so it carries no safety guarantee.

TWO VARIANTS. `obstacles=False` (the default) scores pedestrians only; static obstacles
and the arena wall are not checked, so a substitution can steer into an obstacle. This is
the variant of the reported 2x2 (Experiments/cbf_2x2.sh, 9-10 Sep 2026, logs
x_{height,ours}_{off,on}.log), whose activation banner predates the obstacles flag. In that
2x2 pedestrian collisions fell (358 -> 318 for HEIGHT, 331 -> 295 for ours), obstacle
collisions rose (58 -> 112, 62 -> 93), and success did not change detectably.
`obstacles=True` also scores the episode's static rectangles. It was added after the 2x2
and has no reported evaluation. Since 13 Oct 2026 it scores the rectangles even when no
pedestrian is present; before, the filter returned early with no pedestrians in the scene,
which mattered only in this variant.

WHAT INFORMATION IT GETS. Every pedestrian's current position and velocity from the
environment (`env.humans`), including any occluded from the policy, propagated at constant
velocity. It does not read goals, forecasts or reactivity flags, and in the default variant
it does not read obstacles at all, whereas the policy perceives obstacles through its
lidar. Its information is therefore neither a subset nor a superset of the policy's.

FAILURE MODE. A filter that is too conservative trades collisions for timeouts. Report
timeouts and every collision type alongside success.
"""
import numpy as np

DELTAS = {0: (0.05, 0.1), 1: (0.05, 0.0), 2: (0.05, -0.1),
          3: (0.0, 0.1),  4: (0.0, 0.0),  5: (0.0, -0.1),
          6: (-0.05, 0.1), 7: (-0.05, 0.0), 8: (-0.05, -0.1)}


class CBFFilter:
    def __init__(self, config, horizon=10, margin=0.10, obstacles=False, verbose=False):
        self.dt = config.env.time_step
        self.v_min = config.robot.v_min
        self.v_max = config.robot.v_max
        self.w_min = config.robot.w_min
        self.w_max = config.robot.w_max
        self.r_rob = config.robot.radius
        self.horizon = horizon      # steps of forward simulation
        self.margin = margin        # extra clearance (m) beyond the collision envelope
        # The pedestrian-only default trades collision types: it steers away from a
        # pedestrian without regard for obstacles (n=1000 pilot: 13 fewer pedestrian and
        # 10 more obstacle collisions; reported 2x2 in the module docstring). With
        # obstacles=True the same clearance test also scores static rectangles.
        self.obstacles = obstacles
        self.verbose = verbose
        self.n_checked = 0
        self.n_overridden = 0

    # ---- forward model: the env's commanded kinematics (no wheel saturation) ------
    def _roll(self, px, py, th, v, w, dv, dw):
        v = min(max(v + dv, self.v_min), self.v_max)
        w = min(max(w + dw, self.w_min), self.w_max)
        th = th + w * self.dt
        return px + v * np.cos(th) * self.dt, py + v * np.sin(th) * self.dt, th, v, w

    def _obstacle_clearance(self, env, px, py):
        """Signed clearance from the robot centre to the nearest static rectangle.

        env.obstacle_coord is a property returning, per obstacle, four world-frame
        corners [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]. Axis-aligned, so the exact
        point-to-rectangle distance is the standard clamped form; negative inside.
        """
        try:
            obs = env.obstacle_coord
        except Exception:
            return 1e9
        worst = 1e9
        for corners in obs:
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            dx = max(x0 - px, 0.0, px - x1)
            dy = max(y0 - py, 0.0, py - y1)
            if dx == 0.0 and dy == 0.0:
                # inside the rectangle: negative distance to the nearest edge
                d = -min(px - x0, x1 - px, py - y0, y1 - py)
            else:
                d = np.hypot(dx, dy)
            worst = min(worst, d - self.r_rob)
        return worst

    def _min_clearance(self, env, a, steps):
        """Smallest clearance over `steps` of holding action `a`: pedestrians always, static
        rectangles too when obstacles=True. None when there is nothing to score, i.e. no
        pedestrians and obstacles=False (the default variant's behaviour is unchanged)."""
        humans = [(h.px, h.py, h.vx, h.vy, h.radius) for h in env.humans]
        if not humans and not self.obstacles:
            return None
        px, py, th = env.robot.px, env.robot.py, env.robot.theta
        v, w = env.desiredVelocity[0], env.desiredVelocity[1]
        dv, dw = DELTAS[a]
        worst = 1e9
        for t in range(1, steps + 1):
            px, py, th, v, w = self._roll(px, py, th, v, w, dv, dw)
            for (hx, hy, hvx, hvy, hr) in humans:
                # constant-velocity human propagation -- NOT ground truth
                fx, fy = hx + hvx * self.dt * t, hy + hvy * self.dt * t
                d = np.hypot(px - fx, py - fy) - (self.r_rob + hr)
                worst = min(worst, d)
            if self.obstacles:
                worst = min(worst, self._obstacle_clearance(env, px, py))
            if worst < -self.margin:
                break               # already unsafe, no need to roll further
        return worst

    # ---- the filter ---------------------------------------------------------------
    def filter(self, env, action):
        """Return a safe action, substituting only if the proposed one is predicted unsafe."""
        a = int(action)
        speak = a >= 9              # acoustic action space: 9-17 mirror 0-8
        base = a % 9

        self.n_checked += 1
        c = self._min_clearance(env, base, self.horizon)
        if c is None or c >= 0.0:
            return action           # nothing in view, or already safe

        # Rank alternatives by clearance, tie-broken by similarity to the policy's choice
        best, best_key = None, None
        for cand in range(9):
            cc = self._min_clearance(env, cand, self.horizon)
            if cc is None:
                continue
            key = (cc, -abs(cand - base))
            if best_key is None or key > best_key:
                best, best_key = cand, key

        if best is None or best == base or best_key[0] <= c:
            return action           # no strictly safer option exists -- do not meddle
        self.n_overridden += 1
        if self.verbose:
            print('  CBF override: %d -> %d (clearance %.3f -> %.3f)'
                  % (base, best, c, best_key[0]))
        return best + 9 if speak else best

    def stats(self):
        r = 0.0 if not self.n_checked else 100.0 * self.n_overridden / self.n_checked
        return self.n_checked, self.n_overridden, r
