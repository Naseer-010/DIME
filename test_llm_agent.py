from types import SimpleNamespace

from agents.llm_agent import LLMResearchAgent


def test_llm_agent_normalizes_observation_and_returns_safe_raw_command(monkeypatch):
    captured = {}

    def fake_llm_decide(observation, model_name, mode, api_base, api_key):
        captured["observation"] = observation
        captured["model_name"] = model_name
        captured["mode"] = mode
        captured["api_base"] = api_base
        captured["api_key"] = api_key
        return {"command": "kubectl logs node-2"}, "check telemetry", "raw-output"

    monkeypatch.setattr("agents.llm_agent.llm_decide", fake_llm_decide)
    agent = LLMResearchAgent(
        model_name="test-model",
        mode="endpoint",
        api_base="http://example.test/v1",
        api_key="test-key",
    )
    observation = SimpleNamespace(
        cpu_loads=[0.1] * 8,
        mem_utilizations=[0.2] * 8,
        queue_lengths=[0] * 8,
        failed_nodes=[],
        latency_ms=12.0,
        request_rate=100.0,
        step=0,
        task_hint="keep the service healthy",
        task_score=0.1,
    )

    action = agent.act(observation)

    assert captured["observation"]["cpu_loads"] == [0.1] * 8
    assert captured["model_name"] == "test-model"
    assert captured["mode"] == "endpoint"
    assert captured["api_base"] == "http://example.test/v1"
    assert captured["api_key"] == "test-key"
    assert action.action_type == "no_op"
    assert action.raw_command == "kubectl logs node-2"
    assert agent.last_reasoning == "check telemetry"
    assert agent.last_raw_output == "raw-output"


def test_llm_agent_falls_back_to_no_op_for_non_dict_model_output(monkeypatch):
    def fake_llm_decide(observation, model_name, mode, api_base, api_key):
        return "not an action dict", "", ""

    monkeypatch.setattr("agents.llm_agent.llm_decide", fake_llm_decide)
    agent = LLMResearchAgent(model_name="test-model")

    action = agent.act({"step": 0})

    assert action.action_type == "no_op"
    assert action.raw_command is None
