import fs from "node:fs/promises";
import path from "node:path";

export type TelemetrySnapshot = {
  step: number;
  cpuLoads: number[];
  queueLengths: number[];
  latencyMs: number;
  failedNodes: number[];
  requestRate: number;
};

type MetricsCsvRow = {
  model: string;
  taskId: string;
  step: number;
  actionTaken: string;
  reasoning: string;
  reward: number;
  cumulativeScore: number;
};

type CsvCache = {
  mtimeMs: number;
  snapshots: TelemetrySnapshot[];
};

let cache: CsvCache | null = null;

const DEFAULT_CSV_PATH = path.resolve(process.cwd(), "..", "metrics_Qwen_Qwen3-8B.csv");
const CSV_PATH = process.env.TELEMETRY_CSV_PATH ?? DEFAULT_CSV_PATH;

const HEADER_COLUMNS = [
  "model",
  "task_id",
  "step",
  "action_taken",
  "reasoning",
  "reward",
  "cumulative_score",
  "done",
  "error",
];

export async function getTelemetrySnapshots(): Promise<TelemetrySnapshot[]> {
  const stat = await fs.stat(CSV_PATH);
  if (cache && cache.mtimeMs === stat.mtimeMs) {
    return cache.snapshots;
  }

  const csvText = await fs.readFile(CSV_PATH, "utf-8");
  const rows = parseCsv(csvText);
  const normalizedRows = normalizeRows(rows);

  const snapshots = normalizedRows
    .map((row, index) => toSnapshot(row, index))
    .filter((snapshot): snapshot is TelemetrySnapshot => snapshot !== null);

  cache = { mtimeMs: stat.mtimeMs, snapshots };
  return snapshots;
}

function normalizeRows(rows: string[][]): MetricsCsvRow[] {
  if (rows.length === 0) return [];

  const startAt = isHeaderRow(rows[0]) ? 1 : 0;
  const normalized: MetricsCsvRow[] = [];

  for (let i = startAt; i < rows.length; i += 1) {
    const row = rows[i];
    if (row.length < 7) continue;

    const step = toFiniteNumber(row[2]);
    const reward = toFiniteNumber(row[5]);
    const cumulativeScore = toFiniteNumber(row[6]);

    normalized.push({
      model: row[0] ?? "",
      taskId: row[1] ?? "",
      step: step ?? i - startAt + 1,
      actionTaken: row[3] ?? "",
      reasoning: row[4] ?? "",
      reward: reward ?? 0,
      cumulativeScore: cumulativeScore ?? 0,
    });
  }

  return normalized;
}

function toSnapshot(row: MetricsCsvRow, index: number): TelemetrySnapshot | null {
  const cpuParsed = extractCpuLoads(row.reasoning);
  const failedParsed = extractFailedNodes(row.reasoning);

  const cpuLoads = normalizeCpuLoads(cpuParsed ?? deriveCpuLoads(row, index), row, index);
  const queueLengths = normalizeQueueLengths(
    extractQueueLengths(row.reasoning) ?? deriveQueueLengths(cpuLoads, row, index),
    cpuLoads,
    row,
    index
  );

  const latencyMs =
    clamp(round1(extractLatencyMs(row.reasoning) ?? deriveLatencyMs(cpuLoads, queueLengths, row)), 0, 2000) ||
    0;

  const failedNodes = normalizeFailedNodes(
    failedParsed ?? deriveFailedNodes(cpuLoads, row),
    cpuLoads.length
  );

  const requestRate =
    clampInt(
      Math.round(extractRequestRate(row.reasoning) ?? deriveRequestRate(cpuLoads, queueLengths, row)),
      0,
      100_000
    ) || 0;

  if (cpuLoads.length === 0 || queueLengths.length === 0) {
    return null;
  }

  return {
    step: Number.isFinite(row.step) ? Math.max(0, Math.trunc(row.step)) : index + 1,
    cpuLoads,
    queueLengths,
    latencyMs,
    failedNodes,
    requestRate,
  };
}

function parseCsv(input: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < input.length; i += 1) {
    const char = input[i];

    if (inQuotes) {
      if (char === '"') {
        const next = input[i + 1];
        if (next === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }

    if (char === ",") {
      row.push(field);
      field = "";
      continue;
    }

    if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }

    if (char === "\r") {
      continue;
    }

    field += char;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function isHeaderRow(row: string[]): boolean {
  if (row.length < HEADER_COLUMNS.length) return false;
  return HEADER_COLUMNS.every((column, index) => (row[index] ?? "").trim().toLowerCase() === column);
}

function extractCpuLoads(reasoning: string): number[] | null {
  const arr = extractNumericArrayByKeyword(reasoning, ["cpu_loads", "cpu loads", "cpu"]);
  return arr && arr.length > 0 ? arr : null;
}

function extractQueueLengths(reasoning: string): number[] | null {
  const arr = extractNumericArrayByKeyword(reasoning, ["queue_lengths", "queue lengths", "queue"]);
  return arr && arr.length > 0 ? arr.map((value) => Math.round(value)) : null;
}

function extractFailedNodes(reasoning: string): number[] | null {
  const lower = reasoning.toLowerCase();
  if (lower.includes("failed_nodes is empty") || lower.includes("failed nodes is empty")) {
    return [];
  }

  const raw = extractRawArrayByKeyword(reasoning, ["failed_nodes", "failed nodes"]);
  if (raw === null) return null;

  const nodeMatches = Array.from(raw.matchAll(/(?:node\s*[-_]?\s*)?(\d+)/gi));
  const parsed = nodeMatches
    .map((match) => Number.parseInt(match[1], 10))
    .filter((value) => Number.isFinite(value) && value >= 0);

  return uniqueInts(parsed);
}

function extractLatencyMs(reasoning: string): number | null {
  const direct =
    extractByRegex(reasoning, /\blatency_ms\b[^0-9-]{0,24}(-?\d+(?:\.\d+)?)/i) ??
    extractByRegex(reasoning, /\bcurrent latency\b[^0-9-]{0,24}(-?\d+(?:\.\d+)?)/i) ??
    extractByRegex(reasoning, /\blatency(?:\s+is|:|=)\s*(-?\d+(?:\.\d+)?)/i);

  if (direct !== null) return direct;

  // Last-resort fallback for prose like: "latency ... 87.3ms"
  const msNearLatency = extractByRegex(reasoning, /\blatency\b[\s\S]{0,40}?(-?\d+(?:\.\d+)?)\s*ms\b/i);
  return msNearLatency ?? null;
}

function extractRequestRate(reasoning: string): number | null {
  const direct =
    extractByRegex(reasoning, /\brequest_rate\b(?!_norm)[^0-9-]{0,24}(-?\d+(?:\.\d+)?)/i) ??
    extractByRegex(reasoning, /\brequest rate\b[^0-9-]{0,24}(-?\d+(?:\.\d+)?)/i) ??
    extractByRegex(reasoning, /\bcurrent request rate\b[^0-9-]{0,24}(-?\d+(?:\.\d+)?)/i);
  return direct ?? null;
}

function extractNumericArrayByKeyword(text: string, keywords: string[]): number[] | null {
  const raw = extractRawArrayByKeyword(text, keywords);
  if (raw === null) return null;

  const matches = raw.match(/-?\d+(?:\.\d+)?/g);
  if (!matches) return [];

  const parsed = matches
    .map((token) => Number.parseFloat(token))
    .filter((value) => Number.isFinite(value));

  return parsed;
}

function extractRawArrayByKeyword(text: string, keywords: string[]): string | null {
  const lower = text.toLowerCase();

  for (const keyword of keywords) {
    let fromIndex = 0;

    while (true) {
      const idx = lower.indexOf(keyword.toLowerCase(), fromIndex);
      if (idx === -1) break;

      const open = text.indexOf("[", idx);
      if (open === -1 || open - idx > 220) {
        fromIndex = idx + keyword.length;
        continue;
      }

      const close = text.indexOf("]", open + 1);
      if (close === -1 || close - open > 500) {
        fromIndex = idx + keyword.length;
        continue;
      }

      return text.slice(open + 1, close);
    }
  }

  return null;
}

function extractByRegex(text: string, regex: RegExp): number | null {
  const match = text.match(regex);
  if (!match || !match[1]) return null;
  const value = Number.parseFloat(match[1]);
  return Number.isFinite(value) ? value : null;
}

function normalizeCpuLoads(values: number[], row: MetricsCsvRow, index: number): number[] {
  const nodeCount = 4;
  const result = fillToSize(values, nodeCount, (slot) => deriveCpuSlot(row, index, slot));
  return result.map((value) => round3(clamp(value, 0, 1)));
}

function normalizeQueueLengths(
  values: number[],
  cpuLoads: number[],
  row: MetricsCsvRow,
  index: number
): number[] {
  const nodeCount = cpuLoads.length;
  const result = fillToSize(values, nodeCount, (slot) => deriveQueueSlot(cpuLoads, row, index, slot));
  return result.map((value) => clampInt(Math.round(value), 0, 100_000));
}

function normalizeFailedNodes(values: number[], nodeCount: number): number[] {
  return uniqueInts(values)
    .filter((value) => value >= 0 && value < nodeCount)
    .sort((a, b) => a - b);
}

function deriveCpuLoads(row: MetricsCsvRow, index: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < 4; i += 1) {
    out.push(deriveCpuSlot(row, index, i));
  }
  return out;
}

function deriveCpuSlot(row: MetricsCsvRow, index: number, slot: number): number {
  const h = stableHash(`${row.taskId}|${row.step}|${row.actionTaken}|cpu|${slot}|${index}`);
  const taskBias = taskCpuBias(row.taskId);
  const scoreBias = clamp(row.cumulativeScore, 0, 1) * 0.25;
  const rewardBias = row.reward < 0 ? 0.18 : 0.1;
  const hashed = ((h % 1000) / 1000) * 0.38;
  return clamp(taskBias + scoreBias + rewardBias + hashed, 0.05, 1);
}

function deriveQueueLengths(cpuLoads: number[], row: MetricsCsvRow, index: number): number[] {
  return cpuLoads.map((cpu, slot) => deriveQueueSlot(cpuLoads, row, index, slot, cpu));
}

function deriveQueueSlot(
  cpuLoads: number[],
  row: MetricsCsvRow,
  index: number,
  slot: number,
  knownCpu?: number
): number {
  const cpu = knownCpu ?? cpuLoads[slot] ?? 0;
  const h = stableHash(`${row.taskId}|${row.step}|queue|${slot}|${index}`);
  const taskBoost = taskQueueBias(row.taskId);
  const noise = h % 11;
  return Math.max(0, Math.round(cpu * 80 + taskBoost + noise));
}

function deriveLatencyMs(cpuLoads: number[], queueLengths: number[], row: MetricsCsvRow): number {
  const cpuAvg = average(cpuLoads);
  const queueAvg = average(queueLengths);
  const rewardPenalty = row.reward <= 0 ? 22 : 8;
  return clamp(cpuAvg * 140 + queueAvg * 0.95 + rewardPenalty, 5, 2000);
}

function deriveFailedNodes(cpuLoads: number[], row: MetricsCsvRow): number[] {
  const inferred = cpuLoads
    .map((cpu, idx) => ({ cpu, idx }))
    .filter((entry) => entry.cpu <= 0.02)
    .map((entry) => entry.idx);

  if (inferred.length > 0) return inferred;

  if (row.reward <= -1000) {
    return [0];
  }

  return [];
}

function deriveRequestRate(cpuLoads: number[], queueLengths: number[], row: MetricsCsvRow): number {
  const base = taskRequestBase(row.taskId);
  const cpuFactor = average(cpuLoads) * 480;
  const queueFactor = average(queueLengths) * 3.2;
  return base + cpuFactor + queueFactor;
}

function taskRequestBase(taskId: string): number {
  const task = taskId.toLowerCase();
  if (task.includes("flash")) return 600;
  if (task.includes("traffic")) return 280;
  if (task.includes("cascade")) return 220;
  if (task.includes("node")) return 140;
  return 180;
}

function taskCpuBias(taskId: string): number {
  const task = taskId.toLowerCase();
  if (task.includes("flash")) return 0.58;
  if (task.includes("traffic")) return 0.46;
  if (task.includes("cascade")) return 0.52;
  if (task.includes("node")) return 0.4;
  return 0.42;
}

function taskQueueBias(taskId: string): number {
  const task = taskId.toLowerCase();
  if (task.includes("flash")) return 42;
  if (task.includes("traffic")) return 26;
  if (task.includes("cascade")) return 34;
  if (task.includes("node")) return 20;
  return 24;
}

function fillToSize(values: number[], size: number, deriveAt: (slot: number) => number): number[] {
  const normalized = values
    .filter((value) => Number.isFinite(value))
    .slice(0, size);

  while (normalized.length < size) {
    normalized.push(deriveAt(normalized.length));
  }

  return normalized;
}

function toFiniteNumber(value: string | undefined): number | null {
  if (value === undefined) return null;
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  const total = values.reduce((sum, value) => sum + value, 0);
  return total / values.length;
}

function stableHash(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function uniqueInts(values: number[]): number[] {
  return [...new Set(values.map((value) => Math.trunc(value)))];
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampInt(value: number, min: number, max: number): number {
  return Math.trunc(clamp(value, min, max));
}

function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}
