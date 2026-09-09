import numpy as np
from dataclasses import dataclass
from enum import Enum
from demo134.TrajectoryUtils import goto5

@dataclass
class TrajectoryState:
    name: str
    duration: float
    q: np.ndarray
    t_at_start: float = None

class Trajectory:
    def __init__(self, q0: np.ndarray):
        self.trajectory_states = []
        self.q0 = q0
        
    def _ensure_valid_state(self, state: TrajectoryState):
        # TODO: implement this correctly. IMPORTANT: can cause robot breaking
        # ensure that t_at_start from different states are not conflicting
        # if conflicting: will cause jerks
        if state.t_at_start is None:
            total_duration = sum(state.duration for state in self.trajectory_states)
            state.t_at_start = total_duration
        return state
    
    def add_state(self, state: TrajectoryState):
        state = self._ensure_valid_state(state)
        self.trajectory_states.append(state)
        
    def add_states(self, states: list[TrajectoryState]):
        for state in states:
            self.add_state(state)
    
    # def get_state(self, t: float) -> TrajectoryState:
    #     for state in self.trajectory_states:
    #         if t < state.t_at_start + state.duration:
    #             return state
    #     return self.trajectory_states[-1]
    
    def get_update(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.trajectory_states) == 0:
            return [], [], []
        q0 = self.q0.copy()
        for state in self.trajectory_states:
            if state.t_at_start <= t < state.t_at_start + state.duration:
                break
            else:
                q0 = state.q.copy()
        else:
            state = self.trajectory_states[-1]
            return state.q, np.zeros_like(state.q), np.zeros_like(state.q)
        t_state = t - state.t_at_start
        q, qd, qdd = goto5(t_state, state.duration, q0, state.q)
        return q, qd, qdd
