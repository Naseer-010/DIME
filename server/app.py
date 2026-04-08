"""
FastAPI application for the Distributed Infrastructure Environment.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app

from server.environment import DistributedInfraEnvironment
from server.models import InfraAction, InfraObservation
from fastapi.responses import HTMLResponse
import os

# --- THE FIX: The Singleton Factory Pattern ---
# 1. Create the environment instance in memory once
_global_env = DistributedInfraEnvironment()

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

def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()