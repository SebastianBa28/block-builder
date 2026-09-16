import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque

from threedof.TrajectoryUtils import goto5
from threedof.KinematicChain import KinematicChain
from threedof.config import Q_READY

class P_OR_Q(Enum):
    P = 0
    Q = 1

@dataclass
class TrajectoryState:
    mode: P_OR_Q
    q: np.ndarray = None       # joint angles
    p: np.ndarray = None       # x, y, z
    q_init: np.ndarray = None  # previous joint angles
    min_duration: float = None # secs
    delay_before: float = 0    # secs
    delay_after: float  = 0    # secs
    t_at_start: float   = None    # secs (includes delay_before)
    t_at_end: float     = None    # secs (includes delay_after)

def damped_pinv(J: np.ndarray, gamma: float = 0.001) -> np.ndarray:
    """
    Computes the damped pseudo-inverse of a matrix J using SVD.
    """
    U, S, Vt = np.linalg.svd(J)
    S_plus = np.zeros((len(Vt), len(U.T)))
    for i in range(len(S)):
        if gamma <= 0.0: # No damping
            S_plus[i][i] = 1 / S[i] if S[i] != 0.0 else 0.0
        else:
            if S[i] < gamma: S_plus[i][i] = S[i] / (gamma**2)
            else: S_plus[i][i] = 1 / S[i]
    Jpinv = Vt.T @ S_plus @ U.T
    condition_number = S[0] / S[-1] if S[-1] != 0.0 else 0.0
    return Jpinv, condition_number


class Trajectory:
    def __init__(self, node, q0: np.ndarray, dt: float, clock, start_time, chain=KinematicChain):
        self.node = node
        self.logger = node.get_logger()
        self.trajectory_states: deque[TrajectoryState] = deque([])
        self.logger.info(f"Initial position p0: {chain.fkin(q0)[0]}")
        self.dt = dt
        self.chain = chain
        self.lambda_ = 20.0
        self.q0 = q0
        self.pos_error = np.zeros(3)
        self.clock = clock
        self.start_time = start_time
        
    def get_t(self):
        now = self.clock.now()
        return (now - self.start_time).nanoseconds * 1e-9

    def _ensure_valid_state(self, state: TrajectoryState):
        # TODO: implement this correctly. IMPORTANT: can cause robot breaking
        # ensure that t_at_start from different states are not conflicting
        # if conflicting: will cause jerks
        if state.mode == P_OR_Q.Q and state.q is None:
            print(f"Error: TrajectoryState in Q mode must have a final joint value 'q'.")
            return None
        if state.mode == P_OR_Q.P and state.p is None:
            print(f"Error: TrajectoryState in P mode must have a final position value 'p'.")
            return None
        if state.q is not None and state.p is not None:
            print(f"Warning: TrajectoryState has both 'q' and 'p' defined. Using 'q' for Q mode and 'p' for P mode.")
            if state.mode == P_OR_Q.Q:
                state.p = None
            else:
                state.q = None
        
        # If p or q are missing, compute them using IK or FK
        if state.mode == P_OR_Q.Q and state.p is None:
            state.p = self.chain.fkin(state.q)[0]
        elif state.mode == P_OR_Q.P and state.q is None:
            state.q = self.ikin(state.p)
        
        return state

    def _update_t_start_end(self, t: float):
        if not self.trajectory_states:
            return

        initial_t = self.trajectory_states[0].t_at_start
        running_duration = min(t, initial_t) if initial_t is not None else t
        for s in self.trajectory_states:
            s.t_at_start = running_duration
            s.t_at_end = running_duration + s.delay_before + s.min_duration + s.delay_after
            running_duration = s.t_at_end
        
    def add_state(
        self, 
        state: TrajectoryState,
        prioritize: bool = False,
    ):
        state = self._ensure_valid_state(state)
        if not state:
            return
        
        if prioritize:
            self.trajectory_states.appendleft(state)
        else:
            self.trajectory_states.append(state)
        self._update_t_start_end(t=self.get_t())

    def clear_states(self, q_curr: np.ndarray):
        self.trajectory_states = deque([])
        self.q0 = q_curr
        self.logger.info("Cleared all trajectory states. Holding current position.")

    def add_states(
        self, 
        states: list[TrajectoryState], 
        prioritize: bool = False,
    ):
        iterator = states if not prioritize else reversed(states)
        for state in iterator:
            self.add_state(state, prioritize=prioritize)

    def _pop_completed_states(self, t: float):
        while self.trajectory_states and t >= self.trajectory_states[0].t_at_end:
            popped_state = self.trajectory_states.popleft()
            self.q0 = popped_state.q
            self.logger.info("Completed a trajectory state.")
            
    def _ensure_q_init(self, state: TrajectoryState):
        if state.q_init is None:
            state.q_init = self.q0

    def ikin(
        self, 
        p_desired: np.ndarray, 
        q_guess: np.ndarray = Q_READY,
        max_iter: int = 10000,
        tolerance = 1e-3,
        sim_dt = 0.05
    ) -> np.ndarray:
        # Define Convergence Criteria

        # Initial Guess
        q = q_guess

        (p, R, Jv, Jw) = self.chain.fkin(q)
        e = p_desired  - p

        iter_ = 0
        while np.linalg.norm(e) > tolerance:
            if iter_ > max_iter:
                self.logger.info('IK Solution did not converge')
                return q
            qd = np.linalg.pinv(Jv) @  e
            q = q + qd * sim_dt
            (p, R, Jv, Jw) = self.chain.fkin(q)
            e = p_desired - p
            iter_ += 1
        
        return q

    def get_update(
        self, 
        t: float, 
    ) -> tuple[np.ndarray, np.ndarray]:
        self._pop_completed_states(t)
        
        # If queue is empty, hold the last known stable position (q0)
        if not self.trajectory_states:
            return self.q0, np.zeros_like(self.q0)

        state_curr = self.trajectory_states[0]
        
        # --- PHASE 1: DELAY BEFORE ---
        if t < state_curr.t_at_start + state_curr.delay_before:
            # We are waiting to start. Hold the previous position (q0).
            return self.q0, np.zeros_like(self.q0)

        self._ensure_q_init(state_curr)

        # --- PHASE 2: ACTIVE MOVEMENT ---
        if t <= state_curr.t_at_end - state_curr.delay_after:
            # Compute active trajectory -- q always defined for state
            q, qd, _ = goto5(
                t - state_curr.t_at_start - state_curr.delay_before, 
                state_curr.min_duration, 
                state_curr.q_init, 
                state_curr.q
            )
            if t + self.dt >= state_curr.t_at_end - state_curr.delay_after:
                self.q0 = state_curr.q
            return q, qd

        # --- PHASE 3: DELAY AFTER ---
        else:
            # Hold final position of current state
            return self.q0, np.zeros_like(self.q0)