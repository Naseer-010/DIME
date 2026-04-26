"""
FastAPI application for the Distributed Infrastructure Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import os

from openenv.core.env_server.http_server import create_app
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation

# --- THE FIX: The Singleton Factory Pattern ---
# 1. Create the environment instance in memory once
_global_env = DistributedInfraEnvironment()
_viz_env = DistributedInfraEnvironment()
_viz_lock = asyncio.Lock()

# 2. Create a "factory function" that returns our active instance
def env_factory():
    return _global_env

# 3. Pass the callable factory function to OpenEnv
app = create_app(
    env_factory,
    InfraAction,
    InfraObservation,
    env_name="distributed_infra_env",
)


#Clear formatted document page for root url
@app.get("/")
def home():
    # Safely locate the home.html file in the same directory as this script
    html_file_path = os.path.join(os.path.dirname(__file__), "home.html")
    
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
        
    return HTMLResponse(content=html_content)


def _parse_action_payload(payload: dict) -> InfraAction:
    """
    Parse a client-sent intervention payload into an InfraAction.

    Supports either structured fields or a raw command string.
    """
    if "command" in payload and payload["command"]:
        return InfraAction(action_type="no_op", raw_command=str(payload["command"]))

    action_type = str(payload.get("action_type", "no_op"))
    kwargs = {
        "action_type": action_type,
        "target": payload.get("target"),
        "from_node": payload.get("from_node"),
        "to_node": payload.get("to_node"),
        "rate": payload.get("rate"),
    }
    return InfraAction(**kwargs)


@app.websocket("/ws/simulation")
async def simulation_socket(websocket: WebSocket):
    """
    Stream live DIME observations for visual frontends.

    Protocol:
    - Server emits a JSON payload roughly every 200ms.
    - Client may send optional intervention JSON:
      {"command":"kubectl throttle ingress --rate=0.3"}
      or structured action fields compatible with InfraAction.
    """
    await websocket.accept()

    # Ensure the visualization environment has an initialized episode.
    async with _viz_lock:
        if _viz_env.sim.step_count == 0 and not _viz_env.sim.nodes:
            _viz_env.reset(task="cascading_failure")

    pending_action: InfraAction | None = None
    pending_command: str | None = None

    try:
        while True:
            # Non-blocking receive so we can preserve a fixed tick-rate stream.
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                incoming = json.loads(raw) if raw else {}
                if isinstance(incoming, dict):
                    pending_action = _parse_action_payload(incoming)
                    pending_command = (
                        str(incoming.get("command"))
                        if incoming.get("command")
                        else incoming.get("action_type")
                    )
            except asyncio.TimeoutError:
                pass

            async with _viz_lock:
                action = pending_action or InfraAction(action_type="no_op")
                pending_action = None

                obs = _viz_env.step(action=action)
                if obs.done:
                    obs = _viz_env.reset(task=_viz_env.sim.task_id or "cascading_failure")

                await websocket.send_json(
                    {
                        "observation": obs.model_dump(),
                        "intervention": pending_command,
                        "last_action_type": _viz_env.sim.last_action_type,
                        "timestamp_ms": int(asyncio.get_event_loop().time() * 1000),
                    }
                )
                pending_command = None

            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, RuntimeError):
        return

def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
