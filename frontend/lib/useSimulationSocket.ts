"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export type SimulationObservation = {
  cpu_loads: number[];
  queue_lengths: number[];
  failed_nodes: number[];
  latency_ms: number;
  request_rate: number;
  mem_utilizations: number[];
  step: number;
  done: boolean;
  action_errors: string[];
};

export type SimulationPacket = {
  observation: SimulationObservation;
  intervention?: string | null;
  last_action_type?: string;
  timestamp_ms?: number;
};

function resolveWebSocketUrl(): string {
  if (process.env.NEXT_PUBLIC_SIM_WS_URL) {
    return process.env.NEXT_PUBLIC_SIM_WS_URL;
  }

  if (typeof window === "undefined") {
    return "ws://localhost:8000/ws/simulation";
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname;
  return `${protocol}://${host}:8000/ws/simulation`;
}

export function useSimulationSocket() {
  const [packet, setPacket] = useState<SimulationPacket | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);

  const wsUrl = useMemo(() => resolveWebSocketUrl(), []);

  useEffect(() => {
    let mounted = true;

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mounted) return;
        setConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        if (!mounted) return;
        try {
          const parsed = JSON.parse(event.data) as SimulationPacket;
          setPacket(parsed);
        } catch {
          setError("Invalid simulation payload");
        }
      };

      ws.onerror = () => {
        if (!mounted) return;
        setError("Simulation socket error");
      };

      ws.onclose = () => {
        if (!mounted) return;
        setConnected(false);
        reconnectTimerRef.current = window.setTimeout(connect, 1500);
      };
    };

    connect();

    return () => {
      mounted = false;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, [wsUrl]);

  const sendIntervention = (command: string) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({ command }));
  };

  return {
    packet,
    connected,
    error,
    sendIntervention,
  };
}
