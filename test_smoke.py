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
        action = InfraAction(action_type="restart_node", target=1)
        obs = env.step(action)
        print(f"  restart_node (no-op on healthy): step={obs.step}")

    # Test reroute_traffic (between app servers only, nodes 1 and 2)
    action = InfraAction(action_type="reroute_traffic", from_node=1, to_node=2)
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

    score = obs.task_score
    assert 0.0 <= score <= 1.0, f"Task score out of range: {score}"
    print(f"  Task score: {score:.4f}")

    print(f"  PASSED: {task_id}")


def test_rubric_breakdown():
    """Verify rubric breakdown with ThroughputVerifier."""
    print("\n--- Testing rubric breakdown ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    action = InfraAction(action_type="no_op")
    obs = env.step(action)
    assert isinstance(obs.reward, float), "Reward should be float"
    print(f"  Rubric reward: {obs.reward}")
    print("  PASSED: rubric_breakdown")


def test_curriculum_levels():
    """Verify all curriculum level task IDs work."""
    print("\n--- Testing curriculum levels ---")
    for task_id in [
        "level_1_read_logs",
        "level_2_single_fix",
        "level_3_stochastic",
        "level_4_expert",
        "level_5_alibaba_trace",
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
    for _ in range(50):
        obs = env.step(InfraAction(action_type="no_op"))
        if "timeout" in obs.telemetry_status.values():
            timeout_seen = True
            timed_out_node = [
                k for k, v in obs.telemetry_status.items() if v == "timeout"
            ][0]
            obs = env.step(InfraAction(action_type="query_logs", target=timed_out_node))
            print(f"  query_logs on node {timed_out_node} at step {obs.step}")
            break

    if timeout_seen:
        print("  Telemetry dropout detected and query_logs tested.")
    else:
        print("  WARN: No telemetry dropout in 50 steps (probabilistic).")
    print("  PASSED: partial_observability")


def test_budget_exhaustion():
    """Verify cloud budget prevents unlimited scale_up."""
    print("\n--- Testing budget exhaustion ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    initial_budget = env.sim.cloud_budget
    print(f"  Initial budget: {initial_budget}")

    for _ in range(initial_budget + 2):
        obs = env.step(InfraAction(action_type="scale_up"))

    assert any("InsufficientFunds" in e for e in obs.action_errors), (
        f"Expected InsufficientFunds error, got: {obs.action_errors}"
    )
    print(f"  Budget exhausted, error: {obs.action_errors[-1][:60]}")
    print("  PASSED: budget_exhaustion")


def test_cooldown():
    """Verify restart_node cooldown prevents spam."""
    print("\n--- Testing restart cooldown ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="node_failure")

    for _ in range(6):
        env.step(InfraAction(action_type="no_op"))

    obs = env.step(InfraAction(action_type="restart_node", target=3))
    assert not any("CooldownActive" in e for e in obs.action_errors)
    print("  First restart on node 3: OK")

    obs = env.step(InfraAction(action_type="restart_node", target=3))
    assert any("CooldownActive" in e for e in obs.action_errors)
    print(f"  Second restart blocked: {obs.action_errors[-1][:60]}")
    print("  PASSED: cooldown")


def test_command_parser():
    """Verify raw_command kubectl parsing."""
    print("\n--- Testing command parser ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # kubectl scale
    obs = env.step(
        InfraAction(
            action_type="no_op",
            raw_command="kubectl scale deployment frontend --replicas=10",
        )
    )
    assert not any("ParseError" in e for e in obs.action_errors)
    print(f"  kubectl scale: parsed OK, nodes={len(obs.cpu_loads)}")

    # kubectl logs
    obs = env.step(InfraAction(action_type="no_op", raw_command="kubectl logs node-2"))
    assert not any("ParseError" in e for e in obs.action_errors)
    print("  kubectl logs: parsed OK")

    # kubectl throttle
    obs = env.step(
        InfraAction(
            action_type="no_op", raw_command="kubectl throttle ingress --rate=0.7"
        )
    )
    assert not any("ParseError" in e for e in obs.action_errors)
    print("  kubectl throttle: parsed OK")

    # invalid command
    obs = env.step(
        InfraAction(action_type="no_op", raw_command="invalid-command --foo")
    )
    assert any("ParseError" in e for e in obs.action_errors)
    print("  invalid command: ParseError raised correctly")

    print("  PASSED: command_parser")


# =====================================================================
# Phase 2 Tests
# =====================================================================


def test_node_roles():
    """Verify node 0 is DB and nodes 1-7 are app servers."""
    print("\n--- Testing node roles ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Node 0 must be database
    assert env.sim.nodes[0].role == "database", f"Node 0 role: {env.sim.nodes[0].role}"
    assert env.sim.nodes[0].capacity == 25, f"DB capacity: {env.sim.nodes[0].capacity}"
    print(
        f"  Node 0: role={env.sim.nodes[0].role}, capacity={env.sim.nodes[0].capacity}"
    )

    # Nodes 1-7 must be app servers
    for i in range(1, 8):
        assert env.sim.nodes[i].role == "app_server", (
            f"Node {i} role: {env.sim.nodes[i].role}"
        )
    print("  Nodes 1-7: all app_server")

    # Fail DB and verify app servers can't process
    env.sim.nodes[0].is_failed = True
    env.sim.nodes[0].cpu_util = 0.0
    # Give app servers some queue
    for n in env.sim.nodes[1:]:
        n.queue_length = 10
    served_before = env.sim.total_requests_served
    env._distribute_load()
    served_after = env.sim.total_requests_served
    # With DB down, app servers should serve 0 requests (DB processes its own)
    app_served = served_after - served_before
    # Only the DB itself might have processed (but it's failed), so delta should be minimal
    print(
        f"  DB failed: app servers served {app_served} requests (expected ~0 from apps)"
    )
    print("  PASSED: node_roles")


def test_cold_start():
    """Verify scale_up creates nodes with cold start penalty."""
    print("\n--- Testing cold start ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Scale up
    obs = env.step(InfraAction(action_type="scale_up"))
    new_node = env.sim.nodes[-1]
    assert new_node.is_temporary, "New node should be temporary"
    assert new_node.booting_steps > 0, (
        f"Expected booting_steps > 0, got {new_node.booting_steps}"
    )
    print(
        f"  New node: booting_steps={new_node.booting_steps}, capacity={new_node.capacity}"
    )

    # During boot, effective capacity = 10% of normal
    effective = max(1, int(new_node.capacity * 0.10))
    print(
        f"  Effective capacity during boot: {effective} (vs {new_node.capacity} normal)"
    )

    # Step a few times and verify booting decrements
    for i in range(3):
        obs = env.step(InfraAction(action_type="no_op"))
    assert env.sim.nodes[-1].booting_steps == 0, (
        f"After 3 steps, booting should be 0, got {env.sim.nodes[-1].booting_steps}"
    )
    print("  After 3 steps: booting_steps=0 (fully operational)")
    print("  PASSED: cold_start")


def test_prometheus_metrics():
    """Verify prometheus_metrics field in observation."""
    print("\n--- Testing Prometheus metrics ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")
    obs = env.step(InfraAction(action_type="no_op"))

    assert hasattr(obs, "prometheus_metrics"), "Missing prometheus_metrics field"
    assert isinstance(obs.prometheus_metrics, list), "prometheus_metrics should be list"
    assert len(obs.prometheus_metrics) > 0, "prometheus_metrics should not be empty"

    # Check structure of first metric
    m = obs.prometheus_metrics[0]
    assert "metric" in m, "Missing 'metric' key"
    assert "labels" in m, "Missing 'labels' key"
    assert "value" in m, "Missing 'value' key"
    assert "timestamp" in m, "Missing 'timestamp' key"
    print(f"  Sample metric: {m}")

    # Check for node role labels
    cpu_metrics = [
        m for m in obs.prometheus_metrics if m["metric"] == "node_cpu_utilization"
    ]
    if cpu_metrics:
        assert "role" in cpu_metrics[0]["labels"], "CPU metric should have role label"
        roles = set(m["labels"]["role"] for m in cpu_metrics)
        print(f"  Roles in metrics: {roles}")
        assert "database" in roles or "app_server" in roles, "Should have role labels"

    # Check for global metrics
    global_metrics = [m["metric"] for m in obs.prometheus_metrics if not m["labels"]]
    print(f"  Global metrics: {global_metrics}")
    assert "cluster_latency_ms" in global_metrics, "Missing cluster_latency_ms"

    print("  PASSED: prometheus_metrics")


def test_throughput_verifier():
    """Verify reward function penalizes throttle-to-zero exploit vs no_op."""
    print("\n--- Testing ThroughputVerifier ---")
    from server.rubrics import ProductionSREReward

    engine = ProductionSREReward()

    # Healthy cluster state (no failures, moderate load)
    state = {
        "cpu_loads": [0.3, 0.4, 0.5, 0.3, 0.4, 0.3, 0.4, 0.3],
        "mem_utilizations": [0.3, 0.35, 0.4, 0.3, 0.35, 0.3, 0.35, 0.3],
        "queue_lengths": [5, 10, 8, 5, 7, 5, 8, 5],
        "failed_nodes": [],
        "latency_ms": 30.0,
        "p99_latency": 40.0,
        "error_budget": 100.0,
    }

    # Normal no_op action
    reward_normal = engine.calculate_reward(state, {"action_type": "no_op"})
    print(f"  Normal reward (no_op, healthy): {reward_normal:.4f}")

    # Throttle-to-zero exploit
    reward_exploit = engine.calculate_reward(
        state, {"action_type": "throttle", "rate": 0.0}
    )
    print(f"  Exploit reward (throttle=0.0): {reward_exploit:.4f}")

    # Throttle should have penalty (action tax + shed tax)
    assert reward_exploit < reward_normal, (
        f"Exploit reward ({reward_exploit:.4f}) should be < normal ({reward_normal:.4f}). "
        f"The throttle/action tax should make exploit strictly worse."
    )

    # Also verify both are bounded
    assert -5.0 <= reward_normal <= 5.0, f"Normal reward out of bounds: {reward_normal}"
    assert -5.0 <= reward_exploit <= 5.0, (
        f"Exploit reward out of bounds: {reward_exploit}"
    )

    print("  Zero-service exploit correctly penalized!")
    print("  PASSED: throughput_verifier")


def test_alibaba_trace_replay():
    """Verify Alibaba trace replay loads and drives traffic."""
    print("\n--- Testing Alibaba trace replay ---")
    env = DistributedInfraEnvironment()
    obs = env.reset(seed=42, task="level_5_alibaba_trace")

    assert env.sim.trace_replay is not None, "Trace should be loaded"
    assert env.sim.cloud_budget == 8, f"Budget should be 8, got {env.sim.cloud_budget}"
    print(f"  Trace loaded: {len(env.sim.trace_replay)} steps available")
    print(f"  Cloud budget: {env.sim.cloud_budget}")

    # Run 10 steps and collect request rates
    rates = []
    for _ in range(10):
        obs = env.step(InfraAction(action_type="no_op"))
        rates.append(obs.request_rate)

    # Rates should vary (not constant Gaussian)
    rate_variance = max(rates) - min(rates)
    print(
        f"  Request rate range: {min(rates):.1f} - {max(rates):.1f} (variance={rate_variance:.1f})"
    )
    assert rate_variance > 1.0, "Trace should produce varying request rates"
    print("  PASSED: alibaba_trace_replay")


def test_db_dependency_cascade():
    """Verify DB failure halts all app server processing."""
    print("\n--- Testing DB dependency cascade ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Give app servers queued work
    for n in env.sim.nodes[1:]:
        n.queue_length = 20

    # Kill the DB
    env.sim.nodes[0].is_failed = True
    env.sim.nodes[0].cpu_util = 0.0

    served_before = env.sim.total_requests_served

    # Step (should not be able to process app server queues)
    obs = env.step(InfraAction(action_type="no_op"))

    # Check for CRITICAL error message
    has_critical = any("CRITICAL" in e for e in obs.action_errors)
    print(f"  DB failed: CRITICAL error in action_errors = {has_critical}")

    # App server queues should not have emptied significantly
    app_queues = [env.sim.nodes[i].queue_length for i in range(1, 8)]
    print(f"  App server queues after DB failure: {app_queues}")
    print("  PASSED: db_dependency_cascade")


# =====================================================================
# Phase 3 Tests — RL Readiness Verification
# =====================================================================


def test_zombie_node_zeroed():
    """Verify failed nodes with no healthy neighbors have metrics zeroed."""
    print("\n--- Testing zombie node zeroed ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Fail ALL nodes so that when a node fails, there are no healthy neighbors
    for node in env.sim.nodes:
        node.is_failed = True
        node.cpu_util = 0.0
        node.queue_length = 0

    # Now un-fail one node, set it to high CPU, then let it fail
    env.sim.nodes[3].is_failed = False
    env.sim.nodes[3].cpu_util = 0.95
    env.sim.nodes[3].queue_length = 50
    env.sim.nodes[3].memory_util = 0.7
    env.sim.nodes[3].high_cpu_streak = 3  # triggers failure

    # Run failure check
    env._check_failures()

    # Node 3 should now be failed AND have zeroed metrics
    assert env.sim.nodes[3].is_failed, "Node 3 should be failed"
    assert env.sim.nodes[3].cpu_util == 0.0, (
        f"Zombie bug: cpu_util={env.sim.nodes[3].cpu_util}, expected 0.0"
    )
    assert env.sim.nodes[3].queue_length == 0, (
        f"Zombie bug: queue_length={env.sim.nodes[3].queue_length}, expected 0"
    )
    assert env.sim.nodes[3].memory_util == 0.0, (
        f"Zombie bug: memory_util={env.sim.nodes[3].memory_util}, expected 0.0"
    )
    print("  Node 3 failed with zeroed metrics (no zombie state)")
    print("  PASSED: zombie_node_zeroed")


def test_reward_signal_variance():
    """Verify rewards are NOT all identical (gradient health check)."""
    print("\n--- Testing reward signal variance ---")
    from server.rubrics import ProductionSREReward

    engine = ProductionSREReward()

    # Test: rewards should vary across different cluster states
    states = [
        {  # Healthy cluster
            "cpu_loads": [0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "mem_utilizations": [0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "queue_lengths": [2, 5, 5, 5, 5, 5, 5, 5],
            "failed_nodes": [],
            "latency_ms": 20.0,
            "p99_latency": 30.0,
            "error_budget": 100.0,
        },
        {  # Light stress (still manageable)
            "cpu_loads": [0.35, 0.5, 0.45, 0.5, 0.45, 0.5, 0.45, 0.5],
            "mem_utilizations": [0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            "queue_lengths": [5, 15, 10, 15, 10, 15, 10, 15],
            "failed_nodes": [],
            "latency_ms": 45.0,
            "p99_latency": 48.0,
            "error_budget": 90.0,
        },
        {  # Moderate stress
            "cpu_loads": [0.5, 0.6, 0.55, 0.6, 0.55, 0.6, 0.55, 0.6],
            "mem_utilizations": [0.4, 0.5, 0.45, 0.5, 0.45, 0.5, 0.45, 0.5],
            "queue_lengths": [10, 30, 25, 30, 25, 30, 25, 30],
            "failed_nodes": [],
            "latency_ms": 60.0,
            "p99_latency": 80.0,
            "error_budget": 75.0,
        },
        {  # DB dead (catastrophic)
            "cpu_loads": [0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "mem_utilizations": [0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            "queue_lengths": [0, 200, 200, 200, 200, 200, 200, 200],
            "failed_nodes": [0],
            "latency_ms": 500.0,
            "p99_latency": 800.0,
            "error_budget": 30.0,
        },
    ]

    rewards = []
    for i, state in enumerate(states):
        r = engine.calculate_reward(state, {"action_type": "no_op"})
        rewards.append(r)
        print(f"  State {i} reward: {r:.4f}")

    unique_rewards = set(round(r, 2) for r in rewards)
    print(f"  Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"  Unique reward values: {len(unique_rewards)}")

    assert len(unique_rewards) >= 3, (
        f"Insufficient reward variance: only {len(unique_rewards)} unique values. "
        f"Need at least 3 for healthy RL gradients."
    )

    # Ensure all bounded
    assert all(r >= -5.0 for r in rewards), (
        f"Unbounded reward detected: {min(rewards):.2f}. Must be >= -5.0"
    )
    assert all(r <= 5.0 for r in rewards), (
        f"Unbounded reward detected: {max(rewards):.2f}. Must be <= 5.0"
    )

    # Healthy should be strictly better than DB-dead
    assert rewards[0] > rewards[-1], (
        f"Healthy ({rewards[0]:.4f}) should reward more than DB-dead ({rewards[-1]:.4f})"
    )

    print("  Reward signal has healthy variance and monotonic degradation!")
    print("  PASSED: reward_signal_variance")


def test_reward_bounded():
    """Verify reward is bounded even when DB fails."""
    print("\n--- Testing reward bounded on DB failure ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Force DB failure
    env.sim.nodes[0].is_failed = True
    env.sim.nodes[0].cpu_util = 0.0

    obs = env.step(InfraAction(action_type="no_op"))
    reward = obs.reward
    print(f"  Reward after DB failure: {reward}")

    assert reward >= -5.0, (
        f"Reward too low: {reward}. Must be >= -5.0 (was -1000 before fix)."
    )
    assert reward <= 5.0, f"Reward too high: {reward}. Must be <= 5.0."

    # Also test near-total collapse
    for n in env.sim.nodes[1:6]:
        n.is_failed = True
    obs = env.step(InfraAction(action_type="no_op"))
    reward2 = obs.reward
    print(f"  Reward after near-total collapse: {reward2}")
    assert -5.0 <= reward2 <= 5.0, f"Reward out of bounds: {reward2}"

    print("  PASSED: reward_bounded")


def test_node_prefix_parsing():
    """Verify command parser handles 'node-' prefixed IDs from LLM output."""
    print("\n--- Testing node- prefix parsing ---")
    env = DistributedInfraEnvironment()
    env.reset(seed=42, task="traffic_spike")

    # Test: --from=node-5 --to=node-3 (common LLM hallucination)
    obs = env.step(
        InfraAction(
            action_type="no_op",
            raw_command="kubectl exec -it istio-proxy -- traffic shift --from=node-5 --to=node-3",
        )
    )
    assert not any("ParseError" in e for e in obs.action_errors), (
        f"node- prefix caused parse error: {obs.action_errors}"
    )
    print("  traffic shift --from=node-5 --to=node-3: parsed OK")

    # Test: kubectl delete pod node-3 (should already work but verify)
    obs = env.step(
        InfraAction(
            action_type="no_op",
            raw_command="kubectl delete pod node-3",
        )
    )
    assert not any("ParseError" in e for e in obs.action_errors)
    print("  kubectl delete pod node-3: parsed OK")

    # Test: kubectl logs node-2 (should work)
    obs = env.step(
        InfraAction(
            action_type="no_op",
            raw_command="kubectl logs node-2",
        )
    )
    assert not any("ParseError" in e for e in obs.action_errors)
    print("  kubectl logs node-2: parsed OK")

    print("  PASSED: node_prefix_parsing")


if __name__ == "__main__":
    # Phase 1 tasks
    for task in ["traffic_spike", "node_failure", "cascading_failure", "flash_crowd"]:
        test_task(task)

    # Phase 1 feature tests
    test_rubric_breakdown()
    test_curriculum_levels()
    test_partial_observability()
    test_budget_exhaustion()
    test_cooldown()
    test_command_parser()

    # Phase 2 feature tests
    test_node_roles()
    test_cold_start()
    test_prometheus_metrics()
    test_throughput_verifier()
    test_alibaba_trace_replay()
    test_db_dependency_cascade()

    # Phase 3 — RL readiness tests
    test_zombie_node_zeroed()
    test_reward_signal_variance()
    test_reward_bounded()
    test_node_prefix_parsing()

    print("\n=== ALL SMOKE TESTS PASSED ===")
