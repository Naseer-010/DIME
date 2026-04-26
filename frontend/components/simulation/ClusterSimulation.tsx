"use client";

import { memo, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";

type NodeRole = "core" | "gateway" | "worker" | "cache";
type NodeStatus = "ok" | "warm" | "hot" | "offline";

type NodeDatum = {
  id: number;
  label: string;
  role: NodeRole;
  cpu: number;
  mem: number;
  queue: number;
  status: NodeStatus;
};

type Snapshot = {
  step: number;
  nodes: NodeDatum[];
  latencyMs: number;
  requestRate: number;
  action: string;
  stability: number;
};

type Point = {
  x: number;
  y: number;
};

const POSITIONS: Point[] = [
  { x: 480, y: 280 },
  { x: 480, y: 92 },
  { x: 650, y: 150 },
  { x: 722, y: 318 },
  { x: 610, y: 466 },
  { x: 350, y: 466 },
  { x: 238, y: 318 },
  { x: 310, y: 150 },
  { x: 480, y: 176 },
];

const NODE_META: Array<Pick<NodeDatum, "id" | "label" | "role">> = [
  { id: 0, label: "control-plane", role: "core" },
  { id: 1, label: "ingress-01", role: "gateway" },
  { id: 2, label: "worker-02", role: "worker" },
  { id: 3, label: "worker-03", role: "worker" },
  { id: 4, label: "cache-04", role: "cache" },
  { id: 5, label: "worker-05", role: "worker" },
  { id: 6, label: "worker-06", role: "worker" },
  { id: 7, label: "ingress-07", role: "gateway" },
  { id: 8, label: "policy-agent", role: "core" },
];

const STATUS_STYLE: Record<NodeStatus, { accent: string; text: string; fill: string; border: string }> = {
  ok: {
    accent: "#2dd4bf",
    text: "#ccfbf1",
    fill: "rgba(8, 20, 24, 0.92)",
    border: "rgba(45, 212, 191, 0.28)",
  },
  warm: {
    accent: "#fbbf24",
    text: "#fde68a",
    fill: "rgba(29, 22, 10, 0.94)",
    border: "rgba(251, 191, 36, 0.32)",
  },
  hot: {
    accent: "#fb7185",
    text: "#fecdd3",
    fill: "rgba(36, 15, 20, 0.94)",
    border: "rgba(251, 113, 133, 0.36)",
  },
  offline: {
    accent: "#71717a",
    text: "#d4d4d8",
    fill: "rgba(16, 16, 20, 0.96)",
    border: "rgba(113, 113, 122, 0.26)",
  },
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function smoothNoise(seed: number) {
  return Math.sin(seed * 12.9898) * 0.5 + Math.sin(seed * 4.1414) * 0.5;
}

function getStatus(cpu: number, offline: boolean): NodeStatus {
  if (offline) return "offline";
  if (cpu >= 0.82) return "hot";
  if (cpu >= 0.62) return "warm";
  return "ok";
}

function generateSnapshot(step: number): Snapshot {
  const t = step * 0.16;
  const phase = step % 96;
  const surge = phase >= 48 && phase <= 74;
  const recovery = phase > 74;

  const nodes = NODE_META.map((node) => {
    const wave = Math.sin(t + node.id * 0.72);
    const pulse = Math.max(0, Math.sin((phase - 42) / 12));
    const offline = surge && (node.id === 3 || node.id === 6) && phase > 62;
    const base = node.role === "core" ? 0.38 : node.role === "gateway" ? 0.42 : node.role === "cache" ? 0.33 : 0.31;
    const stress = surge ? 0.24 + pulse * 0.24 : recovery ? 0.16 : 0.08;
    const cpu = offline ? 0 : clamp(base + wave * 0.1 + stress + smoothNoise(step + node.id) * 0.018, 0.07, 0.96);
    const mem = offline ? 0 : clamp(0.34 + Math.sin(t * 0.65 + node.id) * 0.08 + (surge ? 0.12 : 0), 0.16, 0.9);
    const queue = offline ? 0 : Math.round(clamp(8 + cpu * 28 + (surge ? pulse * 18 : 0), 2, 58));

    return {
      ...node,
      cpu,
      mem,
      queue,
      status: getStatus(cpu, offline),
    };
  });

  const activeNodes = nodes.filter((node) => node.status !== "offline").length;
  const avgCpu = nodes.reduce((sum, node) => sum + node.cpu, 0) / Math.max(activeNodes, 1);

  return {
    step,
    nodes,
    latencyMs: clamp(32 + avgCpu * 96 + (surge ? Math.max(0, Math.sin((phase - 48) / 10)) * 64 : 0), 28, 240),
    requestRate: clamp(148 + Math.sin(t * 0.4) * 32 + (surge ? 170 : recovery ? 82 : 0), 80, 430),
    action: surge ? "traffic_shift + scale_out" : recovery ? "drain_and_recover" : "steady_state",
    stability: clamp((activeNodes / nodes.length) * (1 - Math.max(0, avgCpu - 0.58) * 0.75), 0.42, 0.99),
  };
}

function edgePath(from: Point, to: Point, bend = 0) {
  const midX = (from.x + to.x) / 2;
  const midY = (from.y + to.y) / 2;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const cx = midX + (-dy / length) * bend;
  const cy = midY + (dx / length) * bend;

  return `M ${from.x} ${from.y} Q ${cx} ${cy} ${to.x} ${to.y}`;
}

function Metric({ label, value, tone = "text-zinc-200" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="min-w-0 rounded-lg border border-zinc-800/80 bg-black/25 px-3 py-2">
      <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-zinc-500">{label}</p>
      <p className={`mt-1 truncate font-mono text-sm ${tone}`}>{value}</p>
    </div>
  );
}

function Edge({ from, to, traffic, muted, index }: { from: Point; to: Point; traffic: number; muted: boolean; index: number }) {
  const path = edgePath(from, to, index % 2 === 0 ? 18 : -18);
  const hot = traffic > 0.78;
  const color = muted ? "rgba(113, 113, 122, 0.28)" : hot ? "rgba(251, 191, 36, 0.58)" : "rgba(45, 212, 191, 0.42)";
  const particleColor = hot ? "#fbbf24" : "#67e8f9";
  const duration = clamp(4.1 - traffic * 2.2, 1.45, 3.8);

  return (
    <g>
      <motion.path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={muted ? 0.8 : 1 + traffic * 1.1}
        strokeLinecap="round"
        initial={false}
        animate={{ opacity: muted ? 0.18 : 0.34 + traffic * 0.34 }}
        transition={{ duration: 0.5 }}
      />
      {!muted ? (
        <>
          <circle r={2.4} fill={particleColor} opacity="0.76">
            <animateMotion dur={`${duration}s`} repeatCount="indefinite" path={path} />
          </circle>
          <circle r={1.45} fill={particleColor} opacity="0.42">
            <animateMotion dur={`${duration * 1.25}s`} begin={`${duration * 0.45}s`} repeatCount="indefinite" path={path} />
          </circle>
        </>
      ) : null}
    </g>
  );
}

const NodeCard = memo(function NodeCard({ node, position }: { node: NodeDatum; position: Point }) {
  const style = STATUS_STYLE[node.status];
  const core = node.role === "core";
  const width = core ? 132 : 116;
  const height = core ? 74 : 66;
  const radius = core ? 18 : 15;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - node.cpu);
  const opacity = node.status === "offline" ? 0.52 : 1;

  return (
    <motion.g
      initial={false}
      animate={{ x: position.x, y: position.y, opacity }}
      transition={{ type: "spring", stiffness: 80, damping: 18 }}
    >
      {node.status !== "offline" ? (
        <motion.circle
          r={width / 2}
          fill="none"
          stroke={style.accent}
          strokeWidth="0.7"
          animate={{ opacity: [0.1, 0.2, 0.1], scale: [0.96, 1.08, 0.96] }}
          transition={{ duration: node.status === "hot" ? 1.5 : 3.8, repeat: Infinity, ease: "easeInOut" }}
        />
      ) : null}

      <motion.rect
        x={-width / 2}
        y={-height / 2}
        width={width}
        height={height}
        rx={14}
        fill={style.fill}
        stroke={style.border}
        strokeWidth="1"
        initial={false}
        animate={{
          filter: node.status === "hot" ? "drop-shadow(0 0 16px rgba(251, 113, 133, 0.22))" : "drop-shadow(0 12px 30px rgba(0, 0, 0, 0.24))",
        }}
        transition={{ duration: 0.45 }}
      />

      <circle cx={-width / 2 + 29} cy={-7} r={radius} fill="rgba(255,255,255,0.025)" stroke="rgba(255,255,255,0.08)" strokeWidth="3" />
      <motion.circle
        cx={-width / 2 + 29}
        cy={-7}
        r={radius}
        fill="none"
        stroke={style.accent}
        strokeWidth="3"
        strokeLinecap="round"
        strokeDasharray={circumference}
        initial={false}
        animate={{ strokeDashoffset: dashOffset }}
        transition={{ duration: 0.65, ease: "easeOut" }}
        transform={`rotate(-90 ${-width / 2 + 29} -7)`}
      />
      <text
        x={-width / 2 + 29}
        y={-3}
        textAnchor="middle"
        fill={style.text}
        fontFamily="var(--font-geist-mono), monospace"
        fontSize="10"
        fontWeight="700"
      >
        {node.status === "offline" ? "off" : `${Math.round(node.cpu * 100)}`}
      </text>

      <text x={-width / 2 + 58} y={-14} fill="#f4f4f5" fontFamily="var(--font-geist-mono), monospace" fontSize="10" fontWeight="700">
        {node.label}
      </text>
      <text x={-width / 2 + 58} y={5} fill="#a1a1aa" fontFamily="var(--font-geist-mono), monospace" fontSize="8.5">
        {node.role.toUpperCase()} / {node.status.toUpperCase()}
      </text>
      <text x={-width / 2 + 58} y={23} fill={style.accent} fontFamily="var(--font-geist-mono), monospace" fontSize="8.5">
        MEM {Math.round(node.mem * 100)}%   Q {node.queue}
      </text>
    </motion.g>
  );
});

export function ClusterSimulation() {
  const [snapshot, setSnapshot] = useState(() => generateSnapshot(0));

  useEffect(() => {
    let step = 0;
    const timer = window.setInterval(() => {
      step += 1;
      setSnapshot(generateSnapshot(step));
    }, 700);

    return () => window.clearInterval(timer);
  }, []);

  const summary = useMemo(() => {
    const online = snapshot.nodes.filter((node) => node.status !== "offline").length;
    const avgCpu = snapshot.nodes.reduce((sum, node) => sum + node.cpu, 0) / Math.max(online, 1);

    return {
      online,
      avgCpu,
      hotNodes: snapshot.nodes.filter((node) => node.status === "hot").length,
    };
  }, [snapshot.nodes]);

  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-800/80 bg-[#080b0f] shadow-[0_24px_80px_rgba(0,0,0,0.38)]">
      <div className="flex flex-col gap-3 border-b border-zinc-800/70 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-5">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-300 shadow-[0_0_12px_rgba(110,231,183,0.7)]" />
          <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-emerald-200/80">Simulation</p>
        </div>
        <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-zinc-500">
          step {snapshot.step} / {summary.online} of {snapshot.nodes.length} nodes online
        </p>
      </div>

      <div className="relative min-h-[25rem] overflow-hidden bg-[radial-gradient(circle_at_50%_40%,rgba(45,212,191,0.12),transparent_38%),linear-gradient(180deg,rgba(24,24,27,0.28),rgba(0,0,0,0.08))]">
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.035)_1px,transparent_1px)] bg-[size:40px_40px] opacity-35" />
        <svg viewBox="0 0 960 560" className="relative h-[25rem] w-full sm:h-[31rem]" preserveAspectRatio="xMidYMid meet" aria-label="Animated distributed cluster simulation">
          <defs>
            <radialGradient id="cluster-core-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(45, 212, 191, 0.22)" />
              <stop offset="65%" stopColor="rgba(45, 212, 191, 0.04)" />
              <stop offset="100%" stopColor="rgba(45, 212, 191, 0)" />
            </radialGradient>
          </defs>

          <motion.circle
            cx="480"
            cy="280"
            r="220"
            fill="url(#cluster-core-glow)"
            animate={{ opacity: [0.45, 0.65, 0.45], scale: [0.98, 1.02, 0.98] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
            style={{ transformOrigin: "480px 280px" }}
          />

          <g opacity="0.38">
            <ellipse cx="480" cy="280" rx="310" ry="192" fill="none" stroke="rgba(161,161,170,0.22)" strokeDasharray="6 16" />
            <ellipse cx="480" cy="280" rx="206" ry="128" fill="none" stroke="rgba(45,212,191,0.16)" strokeDasharray="4 14" />
          </g>

          {snapshot.nodes.slice(1, 8).map((node, index) => (
            <Edge
              key={`edge-core-${node.id}`}
              from={POSITIONS[node.id]}
              to={POSITIONS[0]}
              traffic={node.cpu}
              muted={node.status === "offline"}
              index={index}
            />
          ))}
          {[1, 2, 3, 4, 5, 6, 7].map((id, index, ids) => {
            const nextId = ids[(index + 1) % ids.length];
            const node = snapshot.nodes[id];
            const nextNode = snapshot.nodes[nextId];

            return (
              <Edge
                key={`edge-ring-${id}-${nextId}`}
                from={POSITIONS[id]}
                to={POSITIONS[nextId]}
                traffic={(node.cpu + nextNode.cpu) / 2}
                muted={node.status === "offline" || nextNode.status === "offline"}
                index={index + 7}
              />
            );
          })}
          <Edge from={POSITIONS[8]} to={POSITIONS[0]} traffic={snapshot.stability} muted={false} index={18} />

          {snapshot.nodes.map((node) => (
            <NodeCard key={node.id} node={node} position={POSITIONS[node.id]} />
          ))}
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-2 border-t border-zinc-800/70 p-3 sm:grid-cols-5 sm:p-4">
        <Metric label="latency" value={`${snapshot.latencyMs.toFixed(0)}ms`} tone={snapshot.latencyMs > 150 ? "text-amber-200" : "text-sky-200"} />
        <Metric label="request rate" value={`${snapshot.requestRate.toFixed(0)} rps`} tone="text-violet-200" />
        <Metric label="avg cpu" value={`${Math.round(summary.avgCpu * 100)}%`} tone={summary.avgCpu > 0.72 ? "text-rose-200" : "text-emerald-200"} />
        <Metric label="stability" value={`${Math.round(snapshot.stability * 100)}%`} tone="text-emerald-200" />
        <Metric label="action" value={snapshot.action} tone={summary.hotNodes > 0 ? "text-amber-200" : "text-zinc-300"} />
      </div>
    </div>
  );
}

export default ClusterSimulation;
