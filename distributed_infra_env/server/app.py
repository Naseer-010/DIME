"""
FastAPI application for the Distributed Infrastructure Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app

    from .environment import DistributedInfraEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from server.environment import DistributedInfraEnvironment

# Use parent-level model imports
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import InfraAction, InfraObservation

app = create_app(
    DistributedInfraEnvironment,
    InfraAction,
    InfraObservation,
    env_name="distributed_infra_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
