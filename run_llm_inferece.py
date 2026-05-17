import argparse
import json
import traceback
from typing import Any

# Import the strict referee
from benchmark.evaluation_harness import run_benchmark
from agents.base_agent import BaseAgent
from server.models import InfraAction

# Import the LLM brain from your existing inference file
from inference import llm_decide, build_safe_backend_action

class LLMResearchAgent(BaseAgent):
    """
    A unified agent that bridges the strict evaluation harness
    with the robust LLM prompting/parsing from inference.py.
    """
    def __init__(self, model_name: str, mode: str, api_base: str, api_key: str):
        self.model_name = model_name
        self.mode = mode
        self.api_base = api_base
        self.api_key = api_key
        
    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        """Called by the harness before every episode. Clear any memory here if needed."""
        pass

    def act(self, observation: Any) -> Any:
        # 1. The Harness passes us a raw Python object/dict. 
        # llm_decide expects a dictionary.
        obs_dict = observation if isinstance(observation, dict) else observation.__dict__
        
        # 2. Use inference.py's robust logic to get the action dict and reasoning
        action_dict, reasoning, raw_output = llm_decide(
            observation=obs_dict,
            model_name=self.model_name,
            mode=self.mode,
            api_base=self.api_base,
            api_key=self.api_key
        )
        
        # 3. Use inference.py's safe reconstructor to guarantee no backend crashes
        safe_action_dict = build_safe_backend_action(action_dict)
        
        # 4. Return the typed object the evaluation_harness demands
        return InfraAction(**safe_action_dict)

def main():
    parser = argparse.ArgumentParser(description="Official DIME LLM Evaluator for Research")
    parser.add_argument("--model", required=True, help="e.g., llama3, gpt-4o")
    parser.add_argument("--mode", choices=["local", "endpoint"], default="endpoint")
    parser.add_argument("--api-base", default="http://localhost:11434/v1") # Default to Ollama
    parser.add_argument("--api-key", default="dummy_key")
    parser.add_argument("--split", default="hidden_eval", help="Which task suite to run")
    args = parser.parse_args()

    print(f"Initializing LLM Agent: {args.model} via {args.api_base}...")
    
    # 1. Instantiate our bridged agent
    agent = LLMResearchAgent(
        model_name=args.model,
        mode=args.mode,
        api_base=args.api_base,
        api_key=args.api_key
    )

    print("Handing control to the official Evaluation Harness...")
    
    # 2. Run the strict benchmark. This handles all seeds, looping, and DIME Index math.
    result = run_benchmark(
        agent=agent,
        benchmark_version="DIME-v1.0",
        split=args.split
    )

    print("\n=== BENCHMARK COMPLETE ===")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull reports saved to: {result['run_dir']}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()