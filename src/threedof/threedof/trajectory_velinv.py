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
    label: str
    mode: P_OR_Q
    q: np.ndarray = None       # joint angles
    p: np.ndarray = None       # x, y, z
    q_init: np.ndarray = None  # previous joint angles
    p_init: np.ndarray = None  # previous position
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
    def __init__(self, node, q0: np.ndarray, dt: float, chain=KinematicChain,):
        self.node = node
        self.logger = node.get_logger()
        self.trajectory_states: deque[TrajectoryState] = deque([])
        self.logger.info(f"Initial position p0: {chain.fkin(q0)[0]}")
        self.dt = dt
        self.chain = chain
        self.lambda_ = 20.0
        self.q0 = q0
        self.pos_error = np.zeros(3)
        # self._ensure_valid_q0(q0)

    def _ensure_valid_state(self, state: TrajectoryState):
        # TODO: implement this correctly. IMPORTANT: can cause robot breaking
        # ensure that t_at_start from different states are not conflicting
        # if conflicting: will cause jerks
        if state.label in [state.label for state in self.trajectory_states]:
            print(f"Warning: TrajectoryState with label {state.label} already exists. Skipping addition.")
            return None
        if state.mode == P_OR_Q.Q and state.q is None:
            print(f"Error: TrajectoryState {state.label} in Q mode must have a final joint value 'q'.")
            return None
        if state.mode == P_OR_Q.P and state.p is None:
            print(f"Error: TrajectoryState {state.label} in P mode must have a final position value 'p'.")
            return None
        if state.q is not None and state.p is not None:
            print(f"Warning: TrajectoryState {state.label} has both 'q' and 'p' defined. Using 'q' for Q mode and 'p' for P mode.")
            if state.mode == P_OR_Q.Q:
                state.p = None
            else:
                state.q = None
        
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
        t: float = 0
    ):
        state = self._ensure_valid_state(state)
        if not state:
            return
        
        if prioritize:
            self.trajectory_states.appendleft(state)
        else:
            self.trajectory_states.append(state)
        self._update_t_start_end(t=t)

    def clear_states(self, q_curr: np.ndarray):
        self.trajectory_states = deque([])
        self.q0 = q_curr
        self.logger.info("Cleared all trajectory states. Holding current position.")

    def add_states(
        self, 
        states: list[TrajectoryState], 
        prioritize: bool = False,
        t: float = 0
    ):
        iterator = states if not prioritize else reversed(states)
        for state in iterator:
            self.add_state(state, prioritize=prioritize, t=t)

    # def _ensure_valid_q0(self, q_curr: np.ndarray):
    #     if len(self.trajectory_states) == 0:
    #         self.q0 = q_curr
        

    def _pop_completed_states(self, t: float, q_curr: np.ndarray):
        while self.trajectory_states and t >= self.trajectory_states[0].t_at_end:
            self.q0 = q_curr
            self.trajectory_states.popleft()
            self.logger.info("Completed a trajectory state.")
            
    def _ensure_q_p_init(self, state: TrajectoryState, q_curr: np.ndarray):
        if state.q_init is None:
            state.q_init = q_curr
        if state.p_init is None:
            state.p_init = self.chain.fkin(q_curr)[0]

    def velocity_ikin(
        self, 
        state: TrajectoryState, 
        t: float,
        q_curr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Velocity inverse kinematics

        # Reset position error before starting motion
        if t <= state.t_at_start + state.delay_before:
            self.pos_error = np.zeros_like(self.pos_error)

        t_state = t - state.t_at_start - state.delay_before
        state_duration = state.min_duration
        (curr_p, _, Jv, _) = self.chain.fkin(q_curr)
        p, v, a = goto5(t_state, state_duration, state.p_init, state.p)
        Jpinv, _ = damped_pinv(Jv)
        qd = Jpinv @ (v + self.lambda_ * self.pos_error)
        q = q_curr + qd * self.dt
        self.pos_error = p - self.chain.fkin(q)[0] # pd - pc
        return q, qd

    def get_update(
        self, 
        t: float, 
        q_curr: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        self._pop_completed_states(t, q_curr)
        
        # If queue is empty, hold the last known stable position (q0)
        if not self.trajectory_states:
            return self.q0, np.zeros_like(self.q0)

        state_curr = self.trajectory_states[0]
        
        # --- PHASE 1: DELAY BEFORE ---
        if t < state_curr.t_at_start + state_curr.delay_before:
            # We are waiting to start. Hold the previous position (q0).
            return self.q0, np.zeros_like(self.q0)

        self._ensure_q_p_init(state_curr, q_curr)

        # --- PHASE 2: ACTIVE MOVEMENT ---
        if t <= state_curr.t_at_end - state_curr.delay_after:
            # Compute active trajectory
            if state_curr.mode == P_OR_Q.Q:
                q, qd, _ = goto5(
                    t - state_curr.t_at_start - state_curr.delay_before, 
                    state_curr.min_duration, 
                    state_curr.q_init, 
                    state_curr.q
                )
            else:
                q, qd = self.ikin(state_curr, t, q_curr)

            if t + self.dt >= state_curr.t_at_end - state_curr.delay_after:
                self.q0 = q_curr
            return q, qd

        # --- PHASE 3: DELAY AFTER ---
        else:
            # Hold final position of current state
            return self.q0, np.zeros_like(self.q0)