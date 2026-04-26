"use client";

import "reactflow/dist/style.css";

import { memo, useMemo } from "react";
import ReactFlow, {
  Background,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  MarkerType,
  Position,
  getBezierPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "reactflow";
import { motion } from "framer-motion";

import { useSimulationSocket } from "@/lib/useSimulationSocket";

type InfraNodeData = {
  label: string;
  cpu: number;
  queue: number;
  failed: boolean;
  role: "db" | "worker";
};

type TrafficEdgeData = {
  traffic: number;
};

const NODE_POSITIONS: Array<{ x: number; y: number }> = [
  { x: 380, y: 260 },
  { x: 110, y: 70 },
  { x: 270, y: 40 },
  { x: 500, y: 40 },
  { x: 650, y: 90 },
  { x: 660, y: 290 },
  { x: 510, y: 430 },
  { x: 260, y: 430 },
];

function nodeStyle(cpu: number, failed: boolean, role: "db" | "worker") {
  if (failed) {
    return {
      border: "1.8px solid rgba(248, 113, 113, 0.9)",
      boxShadow: "0 0 0 rgba(0,0,0,0)",
      background: "rgba(28, 28, 33, 0.95)",
      color: "rgba(228, 228, 231, 0.9)",
    };
  }

  if (cpu >= 0.85) {
    return {
      border: "1.8px solid rgba(251, 191, 36, 0.95)",
      boxShadow: "0 0 28px rgba(251, 191, 36, 0.38)",
      background: "rgba(39, 30, 10, 0.92)",
      color: "rgba(255, 251, 235, 0.95)",
    };
  }

  return {
    border: role === "db" ? "1.8px solid rgba(125, 211, 252, 0.95)" : "1.6px solid rgba(45, 212, 191, 0.9)",
    boxShadow:
      role === "db"
        ? "0 0 24px rgba(125, 211, 252, 0.25)"
        : "0 0 20px rgba(45, 212, 191, 0.24)",
    background: role === "db" ? "rgba(13, 33, 48, 0.9)" : "rgba(9, 33, 30, 0.9)",
    color: "rgba(240, 253, 250, 0.95)",
  };
}

const InfraNode = memo(function InfraNode({ data }: NodeProps<InfraNodeData>) {
  const style = nodeStyle(data.cpu, data.failed, data.role);

  return (
    <motion.div
      animate={{
        scale: data.failed ? 0.97 : 1,
        opacity: data.failed ? 0.7 : 1,
      }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="w-44 rounded-2xl px-4 py-3 backdrop-blur"
      style={style}
    >
      <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em]">
        <span>{data.label}</span>
        <span>{data.role === "db" ? "DB" : "APP"}</span>
      </div>
      <div className="mt-3 space-y-1.5 font-mono text-xs">
        <p>CPU {Math.max(0, data.cpu * 100).toFixed(0)}%</p>
        <p>Queue {Math.max(0, data.queue)}</p>
      </div>
    </motion.div>
  );
});

function TrafficEdge(props: EdgeProps<TrafficEdgeData>) {
  const [edgePath, labelX, labelY] = getBezierPath(props);
  const traffic = Math.max(0.2, props.data?.traffic ?? 0.2);
  const duration = Number((2.5 / Math.min(2.5, Math.max(0.25, traffic))).toFixed(2));

  return (
    <>
      <BaseEdge path={edgePath} style={{ stroke: "rgba(148, 163, 184, 0.5)", strokeWidth: 1.4 }} />
      <path id={props.id} d={edgePath} fill="none" stroke="transparent" />
      <circle r="2.8" fill="rgba(125, 211, 252, 0.95)">
        <animateMotion dur={`${duration}s`} repeatCount="indefinite" path={edgePath} />
      </circle>
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            pointerEvents: "none",
          }}
          className="rounded bg-black/65 px-1.5 py-0.5 font-mono text-[10px] text-sky-200"
        >
          {(traffic * 100).toFixed(0)}%
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

const nodeTypes: NodeTypes = {
  infra: InfraNode,
};

const edgeTypes = {
  traffic: TrafficEdge,
};

export function ClusterSimulation() {
  const { packet, connected, error, sendIntervention } = useSimulationSocket();

  const observation = packet?.observation;
  const simulationState = useMemo(
    () => ({
      cpu: observation?.cpu_loads ?? [0.2, 0.3, 0.26, 0.4, 0.32, 0.28, 0.35, 0.31],
      queue: observation?.queue_lengths ?? [2, 6, 4, 9, 5, 7, 3, 2],
      failed: new Set(observation?.failed_nodes ?? []),
    }),
    [observation]
  );

  const nodes = useMemo<Node<InfraNodeData>[]>(() => {
    return NODE_POSITIONS.map((position, idx) => ({
      id: `${idx}`,
      type: "infra",
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      data: {
        label: idx === 0 ? "Node 0" : `Node ${idx}`,
        cpu: simulationState.cpu[idx] ?? 0,
        queue: simulationState.queue[idx] ?? 0,
        failed: simulationState.failed.has(idx),
        role: idx === 0 ? "db" : "worker",
      },
      draggable: false,
      selectable: false,
    }));
  }, [simulationState]);

  const edges = useMemo<Edge<TrafficEdgeData>[]>(() => {
    const workerNodes = [1, 2, 3, 4, 5, 6, 7];
    return workerNodes.map((worker) => {
      const workerCpu = Math.max(0.2, simulationState.cpu[worker] ?? 0.2);
      return {
        id: `e-${worker}-0`,
        source: `${worker}`,
        target: "0",
        type: "traffic",
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "rgba(125, 211, 252, 0.7)",
        },
        style: { stroke: "rgba(125, 211, 252, 0.5)", strokeWidth: 1.4 },
        data: { traffic: workerCpu },
      };
    });
  }, [simulationState]);

  return (
    <div className="rounded-2xl border border-zinc-800 bg-[#0e1117] p-4 shadow-[0_0_45px_rgba(56,189,248,0.08)] sm:p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="font-mono text-xs uppercase tracking-[0.18em] text-zinc-400">
          {connected ? "Live simulation stream" : "Reconnecting simulation stream"}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => sendIntervention("kubectl throttle ingress --rate=0.35")}
            className="rounded-md border border-sky-300/40 bg-sky-400/10 px-3 py-1.5 font-mono text-xs text-sky-100 transition hover:bg-sky-400/20"
          >
            Apply Throttle
          </button>
          <button
            type="button"
            onClick={() => sendIntervention("kubectl rollout restart node-3")}
            className="rounded-md border border-amber-300/40 bg-amber-400/10 px-3 py-1.5 font-mono text-xs text-amber-100 transition hover:bg-amber-400/20"
          >
            Restart Node 3
          </button>
        </div>
      </div>

      <div className="h-[30rem] w-full overflow-hidden rounded-xl border border-zinc-800 bg-[radial-gradient(circle_at_30%_20%,rgba(56,189,248,0.08),transparent_35%),radial-gradient(circle_at_80%_80%,rgba(45,212,191,0.08),transparent_35%)]">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          minZoom={0.55}
          maxZoom={1.4}
          fitView
          fitViewOptions={{ padding: 0.08 }}
          proOptions={{ hideAttribution: true }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnDrag={false}
        >
          <Background color="rgba(148, 163, 184, 0.08)" gap={24} />
          <Controls showInteractive={false} className="!bg-black/60 !text-zinc-100" />
        </ReactFlow>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-zinc-300 sm:grid-cols-4">
        <div className="rounded-lg border border-zinc-800 bg-black/35 p-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-500">Step</p>
          <p className="mt-1 text-lg font-semibold text-white">{observation?.step ?? 0}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-black/35 p-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-500">Latency</p>
          <p className="mt-1 text-lg font-semibold text-white">{(observation?.latency_ms ?? 0).toFixed(1)}ms</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-black/35 p-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-500">Request Rate</p>
          <p className="mt-1 text-lg font-semibold text-white">{(observation?.request_rate ?? 0).toFixed(0)} req/s</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-black/35 p-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-zinc-500">Agent Action</p>
          <p className="mt-1 truncate text-sm font-semibold text-white">{packet?.intervention ?? packet?.last_action_type ?? "no_op"}</p>
        </div>
      </div>

      {error ? <p className="mt-3 font-mono text-xs text-rose-300">socket warning: {error}</p> : null}
    </div>
  );
}

export default ClusterSimulation;
