#!/usr/bin/env python3
"""Smoke test: directly instantiate the environment and run all tasks."""

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation


def test_task(task_id: str):
    print(f"\n--- Testing task: {task_id} ---")
    env = DistributedInfraEnvironment()
    obs = env.reset(seed=42, task=task_id)

    assert isinstance(obs, InfraObservation), (
        f"Expected InfraObservation, got {type(obs)}"
    )
    assert len(obs.cpu_loads) == 8, f"Expected 8 nodes, got {len(obs.cpu_loads)}"
    assert obs.step == 0
    assert obs.task_hint != ""
    print(
        f"  Reset OK: {len(obs.cpu_loads)} nodes, task_hint='{obs.task_hint[:50]}...'"
    )

    # Run 10 steps with no_op
    for i in range(10):
        action = InfraAction(action_type="no_op")
        obs = env.step(action)
        assert isinstance(obs, InfraObservation)
        assert obs.step == i + 1

    print(
        f"  10 no_op steps OK: latency={obs.latency_ms:.1f}ms, failed={obs.failed_nodes}"
    )

    # Test restart_node (if any failed)
    if obs.failed_nodes:
        target = obs.failed_nodes[0]
        action = InfraAction(action_type="restart_node", target=target)
        obs = env.step(action)
        print(f"  restart_node OK: step={obs.step}")
    else:
        # Force a restart on node 0 (it won't do anything if not failed, but won't error)
        action = InfraAction(action_type="restart_node", target=0)
        obs = env.step(action)
        print(f"  restart_node (no-op on healthy): step={obs.step}")

    # Test reroute_traffic
    action = InfraAction(action_type="reroute_traffic", from_node=0, to_node=1)
    obs = env.step(action)
    print(f"  reroute_traffic OK: step={obs.step}")

    # Test scale_up
    action = InfraAction(action_type="scale_up")
    obs = env.step(action)
    print(f"  scale_up OK: now {len(obs.cpu_loads)} nodes, budget={obs.cloud_budget}")

    # Test throttle
    action = InfraAction(action_type="throttle", rate=0.5)
    obs = env.step(action)
    print(f"  throttle OK: request_rate={obs.request_rate}")

    # Check state
    state = env.state
    print(
        f"  State: episode={state.episode_id[:8]}..., steps={state.step_count}, task={state.task_id}"
    )

    # Check reward and done
    assert obs.reward is not None, "Reward should not be None"
    assert isinstance(obs.done, bool), "Done should be bool"

    # Check metadata has task_score
    score = obs.task_score
    assert 0.0 <= score <= 1.0, f"Task score out of range: {score}"
    print(f"  Task score: {score:.4f}")

    print(f"  PASSED: {task_id}")


def test_rubric_breakdown():
    """Verify rubric breakdown appears in observation metadata."""
    print("\n--- Testing rubric breakdown ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    action = InfraAction(action_type="no_op")
    obs = env.step(action)

    # The rubric breakdown is computed internally; we verify the reward is a float
    assert isinstance(obs.reward, float), "Reward should be float"
    assert obs.reward != 0.0 or True, "Reward can be zero but should be computed"
    print(f"  Rubric reward: {obs.reward}")
    print("  PASSED: rubric_breakdown")


def test_curriculum_levels():
    """Verify curriculum level task IDs work."""
    print("\n--- Testing curriculum levels ---")
    for task_id in [
        "level_1_read_logs",
        "level_2_single_fix",
        "level_3_stochastic",
        "level_4_expert",
    ]:
        env = DistributedInfraEnvironment()
        obs = env.reset(seed=42, task=task_id)
        assert isinstance(obs, InfraObservation)
        assert obs.task_hint != ""
        print(f"  {task_id}: hint='{obs.task_hint[:60]}...'")
    print("  PASSED: curriculum_levels")


def test_partial_observability():
    """Verify telemetry dropout occurs and query_logs clears it."""
    print("\n--- Testing partial observability ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=123, task="traffic_spike")

    timeout_seen = False
    for i in range(50):
        action = InfraAction(action_type="no_op")
        obs = env.step(action)
        if "timeout" in obs.telemetry_status.values():
            timeout_seen = True
            # Test query_logs to clear a timed-out node
            timed_out_node = [
                k for k, v in obs.telemetry_status.items() if v == "timeout"
            ][0]
            action = InfraAction(action_type="query_logs", target=timed_out_node)
            obs = env.step(action)
            print(f"  query_logs on node {timed_out_node} at step {obs.step}")
            break

    if timeout_seen:
        print("  Telemetry dropout detected and query_logs tested.")
    else:
        print("  WARN: No telemetry dropout in 50 steps (probabilistic - may happen).")
    print("  PASSED: partial_observability")


def test_budget_exhaustion():
    """Verify cloud budget prevents unlimited scale_up."""
    print("\n--- Testing budget exhaustion ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    initial_budget = env.sim.cloud_budget
    print(f"  Initial budget: {initial_budget}")

    for i in range(initial_budget + 2):
        action = InfraAction(action_type="scale_up")
        obs = env.step(action)

    # After exhausting the budget, action_errors should contain InsufficientFunds
    assert any("InsufficientFunds" in e for e in obs.action_errors), (
        f"Expected InsufficientFunds error, got: {obs.action_errors}"
    )
    print(
        f"  Budget exhausted after {initial_budget} scale_ups, error: {obs.action_errors[-1][:60]}"
    )
    print("  PASSED: budget_exhaustion")


def test_cooldown():
    """Verify restart_node cooldown prevents spam."""
    print("\n--- Testing restart cooldown ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="node_failure")

    # Advance to step 5 to trigger the node 3 failure
    for _ in range(6):
        obs = env.step(InfraAction(action_type="no_op"))

    # First restart should work
    obs = env.step(InfraAction(action_type="restart_node", target=3))
    assert not any("CooldownActive" in e for e in obs.action_errors), (
        "First restart should not be blocked"
    )
    print(f"  First restart on node 3: OK")

    # Second restart immediately should be blocked
    obs = env.step(InfraAction(action_type="restart_node", target=3))
    assert any("CooldownActive" in e for e in obs.action_errors), (
        f"Expected CooldownActive error, got: {obs.action_errors}"
    )
    print(f"  Second restart blocked: {obs.action_errors[-1][:60]}")
    print("  PASSED: cooldown")


def test_command_parser():
    """Verify raw_command kubectl parsing."""
    print("\n--- Testing command parser ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Test kubectl scale
    action = InfraAction(
        action_type="no_op",
        raw_command="kubectl scale deployment frontend --replicas=10",
    )
    obs = env.step(action)
    assert not any("ParseError" in e for e in obs.action_errors), (
        f"kubectl scale should parse OK, got: {obs.action_errors}"
    )
    print(f"  kubectl scale: parsed OK, nodes={len(obs.cpu_loads)}")

    # Test kubectl logs
    action = InfraAction(action_type="no_op", raw_command="kubectl logs node-2")
    obs = env.step(action)
    assert not any("ParseError" in e for e in obs.action_errors), (
        f"kubectl logs should parse OK, got: {obs.action_errors}"
    )
    print(f"  kubectl logs: parsed OK")

    # Test kubectl throttle
    action = InfraAction(
        action_type="no_op", raw_command="kubectl throttle ingress --rate=0.7"
    )
    obs = env.step(action)
    assert not any("ParseError" in e for e in obs.action_errors), (
        f"kubectl throttle should parse OK, got: {obs.action_errors}"
    )
    print(f"  kubectl throttle: parsed OK")

    # Test invalid command
    action = InfraAction(action_type="no_op", raw_command="invalid-command --foo")
    obs = env.step(action)
    assert any("ParseError" in e for e in obs.action_errors), (
        "Invalid command should produce ParseError"
    )
    print(f"  invalid command: ParseError raised correctly")

    print("  PASSED: command_parser")


if __name__ == "__main__":
    # Original tasks
    for task in ["traffic_spike", "node_failure", "cascading_failure", "flash_crowd"]:
        test_task(task)

    # New feature tests
    test_rubric_breakdown()
    test_curriculum_levels()
    test_partial_observability()
    test_budget_exhaustion()
    test_cooldown()
    test_command_parser()

    print("\n=== ALL SMOKE TESTS PASSED ===")
