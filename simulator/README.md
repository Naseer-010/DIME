# Simulator Frontend

The simulator is now a mission-control frontend for the real backend environment in this repo.

It no longer runs a separate fake simulation. It drives the same backend logic used by `server/` and renders that state with a Gymnasium + PyGame UI.

## Install

```powershell
cd C:\NewVolume\DIME
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run The Interactive Frontend

```powershell
cd C:\NewVolume\DIME
.\venv\Scripts\Activate.ps1
python -m simulator.run_demo --task traffic_spike
```

Available tasks:

- `traffic_spike`
- `node_failure`
- `cascading_failure`

## Headless Render Check

```powershell
cd C:\NewVolume\DIME
.\venv\Scripts\Activate.ps1
python -c "import simulator, gymnasium as gym; env = gym.make('CascadeGuardMissionControl-v0', render_mode='rgb_array'); obs, info = env.reset(seed=7, options={'task':'traffic_spike'}); frame = env.render(); print(obs['node_metrics'].shape, obs['edge_metrics'].shape, obs['global_metrics'].shape); print(frame.shape, info['task_id'], info['latency_ms']); env.close()"
```

Expected output shape:

```text
(20, 7) (64, 4) (9,)
(900, 1600, 3) traffic_spike 20.0
```

## Controls

- `Space`: start or pause
- `1`: reroute traffic
- `2`: restart a failed node
- `3`: scale out
- `4`: throttle ingress
- `5`: balance queues
- `Tab`: cycle focused node
- `Mouse click`: focus a node
- `M`: toggle reduced motion
- `R`: reset the current task
- `Esc`: exit
