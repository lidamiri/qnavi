from qnavi.agents.base import Agent
from qnavi.agents.double_q import DoubleQLearning
from qnavi.agents.expected_sarsa import ExpectedSarsa
from qnavi.agents.q_learning import QLearning
from qnavi.agents.registry import AGENTS, make_agent
from qnavi.agents.sarsa import Sarsa

__all__ = [
    "AGENTS",
    "Agent",
    "DoubleQLearning",
    "ExpectedSarsa",
    "QLearning",
    "Sarsa",
    "make_agent",
]
