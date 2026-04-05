#!/usr/bin/env python3
"""Smoke test: directly instantiate the environment and run all 3 tasks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "distributed_infra_env"))

from distributed_infra_env.server.environment import DistributedInfraEnvironment
from distributed_infra_env.models import InfraAction, InfraObservation

def test_task(task_id: str):
    print(f"\n--- Testing task: {task_id} ---")
    env = DistributedInfraEnvironment()
    obs = env.reset(seed=42, task=task_id)

    assert isinstance(obs, InfraObservation), f"Expected InfraObservation, got {type(obs)}"
    assert len(obs.cpu_loads) == 8, f"Expected 8 nodes, got {len(obs.cpu_loads)}"
    assert all(0 <= c <= 1 for c in obs.cpu_loads), f"CPU loads out of range: {obs.cpu_loads}"
    assert obs.step == 0
    assert obs.task_hint != ""
    print(f"  Reset OK: {len(obs.cpu_loads)} nodes, task_hint='{obs.task_hint[:50]}...'")

    # Run 10 steps with no_op
    for i in range(10):
        action = InfraAction(action_type="no_op")
        obs = env.step(action)
        assert isinstance(obs, InfraObservation)
        assert obs.step == i + 1

    print(f"  10 no_op steps OK: latency={obs.latency_ms:.1f}ms, failed={obs.failed_nodes}")

    # Test restart_node
    action = InfraAction(action_type="restart_node", target=0)
    obs = env.step(action)
    print(f"  restart_node OK: step={obs.step}")

    # Test reroute_traffic
    action = InfraAction(action_type="reroute_traffic", from_node=0, to_node=1)
    obs = env.step(action)
    print(f"  reroute_traffic OK: step={obs.step}")

    # Test scale_up
    action = InfraAction(action_type="scale_up")
    obs = env.step(action)
    print(f"  scale_up OK: now {len(obs.cpu_loads)} nodes")

    # Test throttle
    action = InfraAction(action_type="throttle", rate=0.5)
    obs = env.step(action)
    print(f"  throttle OK: request_rate={obs.request_rate}")

    # Check state
    state = env.state
    print(f"  State: episode={state.episode_id[:8]}..., steps={state.step_count}, task={state.task_id}")

    # Check reward and done
    assert obs.reward is not None, "Reward should not be None"
    assert isinstance(obs.done, bool), "Done should be bool"

    # Check metadata has task_score
    score = obs.metadata.get("task_score", -1)
    assert 0.0 <= score <= 1.0, f"Task score out of range: {score}"
    print(f"  Task score: {score:.4f}")

    print(f"  PASSED: {task_id}")

if __name__ == "__main__":
    for task in ["traffic_spike", "node_failure", "cascading_failure"]:
        test_task(task)
    print("\n=== ALL SMOKE TESTS PASSED ===")
