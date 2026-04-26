"use client";

import { useEffect, useRef, useState } from "react";
import { FlipWords } from "@/components/ui/FlipWords";
import { FloatingDock } from "@/components/ui/FloatingDock";
import { FocusCards } from "@/components/ui/FocusCards";
import { TextGenerateEffect } from "@/components/ui/TextGenerateEffect";
import { ClusterSimulation } from "@/components/simulation/ClusterSimulation";
import type { DockItem, FocusCard } from "@/components/ui/types";

type TelemetrySnapshot = {
  step: number;
  cpuLoads: number[];
  queueLengths: number[];
  latencyMs: number;
  failedNodes: number[];
  requestRate: number;
};

type TaskCard = {
  name: string;
  difficulty: "Easy" | "Medium" | "Hard" | "Expert";
  description: string;
  score: number;
};

type MetricFigure = {
  src: string;
  title: string;
  caption: string;
};

const DEPLOYMENT_LIVE = false;

const HERO_FLIP_WORDS = ["adaptive", "resilient", "autonomous", "modern"];

const DOCK_ITEMS: DockItem[] = [
  {
    title: "Home",
    href: "#about",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M3 11.5 12 4l9 7.5" />
        <path d="M5.5 10.5V20h13V10.5" />
      </svg>
    ),
  },
  {
    title: "Simulation",
    href: "#simulation",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <circle cx="12" cy="12" r="3.5" />
        <path d="M12 2.5v3M12 18.5v3M21.5 12h-3M5.5 12h-3M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1M18.7 18.7l-2.1-2.1M7.4 7.4 5.3 5.3" />
      </svg>
    ),
  },
  {
    title: "Features",
    href: "#features",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 6h16M4 12h16M4 18h10" />
      </svg>
    ),
  },
  {
    title: "Reward",
    href: "#reward",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M4 18h16M7 18V9m5 9V5m5 13v-7" />
      </svg>
    ),
  },
  {
    title: "Metrics",
    href: "#metrics",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <rect x="3.5" y="3.5" width="17" height="17" rx="2.5" />
        <path d="M7 15l3-3 2 2 5-5" />
      </svg>
    ),
  },
  {
    title: "Try It",
    href: "#try",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="m5 19 14-7L5 5v5l8 2-8 2z" />
      </svg>
    ),
  },
];

const TASKS: TaskCard[] = [
  {
    name: "Task 1 - Traffic Spike Recovery",
    difficulty: "Easy",
    description:
      "Handles a 3x request surge while keeping latency under 50ms without wasting actions.",
    score: 0.09,
  },
  {
    name: "Task 2 - Single Node Failure",
    difficulty: "Medium",
    description:
      "Repairs a failed node under pressure while preserving uptime and minimizing MTTR penalties.",
    score: 0.05,
  },
  {
    name: "Task 3 - Cascading Failure Prevention",
    difficulty: "Hard",
    description:
      "Must proactively reroute load before thermal hotspots trigger chain-collapse behavior.",
    score: 0.31,
  },
  {
    name: "Task 4 - Flash Crowd Meltdown",
    difficulty: "Expert",
    description:
      "A 5x traffic event where survival demands precise throttle and scale timing under scarcity.",
    score: 0,
  },
];

const FEATURE_CARDS: FocusCard[] = [
  {
    title: "Forest Adventure",
    src: "https://images.unsplash.com/photo-1518710843675-2540dd79065c?q=80&w=3387&auto=format&fit=crop",
  },
  {
    title: "Valley of life",
    src: "https://images.unsplash.com/photo-1600271772470-bd22a42787b3?q=80&w=3072&auto=format&fit=crop",
  },
  {
    title: "Sala behta hi jayega",
    src: "https://images.unsplash.com/photo-1505142468610-359e7d316be0?q=80&w=3070&auto=format&fit=crop",
  },
  {
    title: "Camping is for pros",
    src: "https://images.unsplash.com/photo-1486915309851-b0cc1f8a0084?q=80&w=3387&auto=format&fit=crop",
  },
  {
    title: "The road not taken",
    src: "https://images.unsplash.com/photo-1507041957456-9c397ce39c97?q=80&w=3456&auto=format&fit=crop",
  },
  {
    title: "The First Rule",
    src: "https://assets.aceternity.com/the-first-rule.png",
  },
];

const TASK_CARDS: FocusCard[] = [
  {
    title: "Traffic Spike Recovery",
    src: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=3272&auto=format&fit=crop",
  },
  {
    title: "Single Node Failure",
    src: "https://images.unsplash.com/photo-1562813733-b31f71025d54?q=80&w=3270&auto=format&fit=crop",
  },
  {
    title: "Cascading Failure Prevention",
    src: "https://images.unsplash.com/photo-1518773553398-650c184e0bb3?q=80&w=3272&auto=format&fit=crop",
  },
  {
    title: "Flash Crowd Meltdown",
    src: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?q=80&w=3270&auto=format&fit=crop",
  },
];

const METRIC_FIGURES: MetricFigure[] = [
  {
    src: "/metrics/fig1_vanishing_gradient_fix.png",
    title: "Fig 1: Vanishing Gradient Fix",
    caption: "Stabilized training signal and improved optimization behavior.",
  },
  {
    src: "/metrics/fig2_cascade_exploit_fix.png",
    title: "Fig 2: Cascade Exploit Fix",
    caption: "Mitigates exploit dynamics in cascading-failure conditions.",
  },
  {
    src: "/metrics/fig3_cost_latency_coupling.png",
    title: "Fig 3: Cost-Latency Coupling",
    caption: "Shows tradeoff frontier between resource cost and service latency.",
  },
  {
    src: "/metrics/fig4_curiosity_annealing.png",
    title: "Fig 4: Curiosity Annealing",
    caption: "Annealing schedule effect on exploration vs. stability.",
  },
];

function useRevealOnScroll<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.15 }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return { ref, visible };
}

function RevealSection({
  id,
  className,
  children,
}: {
  id: string;
  className?: string;
  children: React.ReactNode;
}) {
  const { ref, visible } = useRevealOnScroll<HTMLElement>();

  return (
    <section
      id={id}
      ref={ref}
      className={`scroll-mt-28 transition-all duration-700 motion-reduce:transform-none motion-reduce:opacity-100 ${
        visible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
      } ${className ?? ""}`}
    >
      {children}
    </section>
  );
}

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="my-16 sm:my-20" aria-hidden="true">
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4">
        <div className="h-px bg-gradient-to-r from-transparent via-zinc-800 to-zinc-700/50" />
        <span className="rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 font-mono text-[11px] tracking-[0.16em] text-zinc-500">
          {label}
        </span>
        <div className="h-px bg-gradient-to-r from-zinc-700/50 via-zinc-800 to-transparent" />
      </div>
    </div>
  );
}

function DifficultyBadge({ difficulty }: { difficulty: TaskCard["difficulty"] }) {
  const tone = {
    Easy: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
    Medium: "border-amber-500/40 bg-amber-500/10 text-amber-300",
    Hard: "border-orange-500/40 bg-orange-500/10 text-orange-300",
    Expert: "border-pink-500/40 bg-pink-500/10 text-pink-300",
  }[difficulty];

  return <span className={`rounded-full border px-2.5 py-1 text-xs font-mono ${tone}`}>[{difficulty}]</span>;
}

function Spotlight() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10 overflow-hidden">
      <div className="absolute left-1/2 top-[-12rem] h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(251,146,60,0.28),transparent_62%)]" />
      <div className="absolute right-[-8rem] top-12 h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle,rgba(236,72,153,0.22),transparent_68%)]" />
      <div className="absolute bottom-[-12rem] left-[-10rem] h-[24rem] w-[24rem] rounded-full bg-[radial-gradient(circle,rgba(249,115,22,0.18),transparent_65%)]" />
    </div>
  );
}

function HeroWordmark() {
  return (
    <div className="mt-5 flex flex-wrap items-end gap-x-3 gap-y-2 text-[20vw] font-black leading-[0.82] tracking-[-0.06em] md:text-[10vw]">
      <span className="bg-gradient-to-b from-white via-orange-100 to-pink-300 bg-clip-text text-transparent [text-shadow:0_0_26px_rgba(251,146,60,0.22)]">
        DIME
      </span>
      <span className="mb-2 rounded-full border border-orange-300/30 bg-zinc-900/70 px-3 py-1 text-[0.2em] font-semibold uppercase tracking-[0.18em] text-orange-200/95 backdrop-blur-sm">
        <FlipWords words={HERO_FLIP_WORDS} />
      </span>
    </div>
  );
}

function AnimatedHeading({ words, className }: { words: string; className?: string }) {
  return (
    <TextGenerateEffect
      words={words}
      className={`mt-4 text-4xl font-black leading-[0.95] tracking-[-0.03em] text-zinc-100 [text-shadow:0_0_24px_rgba(244,114,182,0.18)] sm:text-6xl ${className ?? ""}`}
      duration={900}
      filter={false}
    />
  );
}

function parseTelemetry(payload: unknown): TelemetrySnapshot | null {
  if (!payload || typeof payload !== "object") return null;

  const candidate = payload as Record<string, unknown>;
  const source =
    (candidate.observation as Record<string, unknown> | undefined) ??
    (candidate.data as Record<string, unknown> | undefined) ??
    candidate;

  const cpuLoadsRaw = source.cpu_loads ?? source.cpuLoads;
  const queueLengthsRaw = source.queue_lengths ?? source.queueLengths;
  const failedNodesRaw = source.failed_nodes ?? source.failedNodes;
  const latencyRaw = source.latency_ms ?? source.latencyMs;
  const requestRateRaw = source.request_rate ?? source.requestRate;
  const stepRaw = source.step;

  if (!Array.isArray(cpuLoadsRaw) || !Array.isArray(queueLengthsRaw) || !Array.isArray(failedNodesRaw)) {
    return null;
  }

  const cpuLoads = cpuLoadsRaw.map((value) => Number(value)).filter((value) => Number.isFinite(value));
  const queueLengths = queueLengthsRaw
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  const failedNodes = failedNodesRaw.map((value) => Number(value)).filter((value) => Number.isFinite(value));

  const latencyMs = Number(latencyRaw);
  const requestRate = Number(requestRateRaw);
  const step = Number(stepRaw);

  if (!Number.isFinite(latencyMs) || !Number.isFinite(requestRate) || !Number.isFinite(step)) {
    return null;
  }

  return {
    step,
    cpuLoads,
    queueLengths,
    latencyMs,
    failedNodes,
    requestRate,
  };
}

export default function Home() {
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [telemetryError, setTelemetryError] = useState<string | null>(null);
  const [showScrollCue, setShowScrollCue] = useState(true);

  useEffect(() => {
    let mounted = true;

    const loadTelemetry = async () => {
      try {
        const response = await fetch("/api/telemetry", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`telemetry request failed (${response.status})`);
        }

        const payload: unknown = await response.json();
        const parsed = parseTelemetry(payload);

        if (!parsed) {
          throw new Error("telemetry payload did not match expected schema");
        }

        if (!mounted) return;
        setTelemetry(parsed);
        setTelemetryError(null);
      } catch (error) {
        if (!mounted) return;
        setTelemetryError(error instanceof Error ? error.message : "unable to fetch telemetry");
      }
    };

    loadTelemetry();
    const timer = window.setInterval(loadTelemetry, 2500);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const onScroll = () => {
      if (window.scrollY > 12) {
        setShowScrollCue(false);
      }
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#080808] text-white">
      <FloatingDock items={DOCK_ITEMS} className="bottom-6" mobileClassName="bottom-4" />

      <main className="relative mx-auto max-w-7xl px-5 pb-24 pt-24 sm:px-8 sm:pt-28">
        <section
          id="about"
          className="relative grid min-h-screen scroll-mt-28 items-center gap-12 pb-14 md:grid-cols-12"
        >
          <Spotlight />

          <div className="md:col-span-7">
            <p className="font-mono text-xs tracking-[0.2em] text-orange-300">[ SRE BENCHMARK ]</p>
            <HeroWordmark />
            <p className="mt-4 max-w-2xl text-xl text-zinc-200 sm:text-2xl">
              Distributed Infrastructure Management Environment
            </p>
            <p className="mt-5 max-w-2xl text-base leading-7 text-zinc-400 sm:text-lg">
              A high-fidelity simulated distributed system for training and evaluating LLM agents on
              complex Site Reliability Engineering tasks. Built on the OpenEnv framework.
            </p>
            <a
              href="#try"
              className="mt-8 inline-flex items-center rounded-md border border-orange-400/40 bg-orange-500/10 px-5 py-3 font-mono text-sm text-orange-100 transition-colors hover:border-pink-400/50 hover:bg-pink-500/10"
            >
              Jump to Live Access -&gt;
            </a>
          </div>

          <div className="md:col-span-5 md:pl-6">
            <div className="rounded-xl border border-zinc-700 bg-zinc-950/95 p-5 font-mono text-sm shadow-[0_0_40px_rgba(251,146,60,0.08)]">
              <p className="text-zinc-500">● NODE STATUS  [step: {telemetry?.step ?? "--"}]</p>
              <div className="mt-4 space-y-2">
                <p className="text-zinc-400">
                  cpu_loads      <span className="text-emerald-300">[{telemetry ? telemetry.cpuLoads.join(", ") : "loading"}]</span>
                </p>
                <p className="text-zinc-400">
                  queue_lengths  <span className="text-orange-300">[{telemetry ? telemetry.queueLengths.join(", ") : "loading"}]</span>
                </p>
                <p className="text-zinc-400">
                  latency_ms     <span className="text-pink-300">{telemetry ? `${telemetry.latencyMs.toFixed(1)}ms` : "loading"}</span>
                </p>
                <p className="text-zinc-400">
                  failed_nodes   <span className="text-red-300">[{telemetry ? telemetry.failedNodes.join(", ") : "loading"}]</span>
                </p>
                <p className="text-zinc-400">
                  request_rate   <span className="text-amber-200">{telemetry ? `${telemetry.requestRate.toFixed(0)} req/s` : "loading"}</span>
                </p>
              </div>

              {telemetryError ? (
                <p className="mt-4 text-xs text-pink-300">telemetry warning: {telemetryError}</p>
              ) : null}
            </div>
          </div>

          <div className="md:col-span-12 pt-2">
            {showScrollCue ? (
              <div className="relative mt-4 text-center font-mono text-xs tracking-[0.18em] text-zinc-500 transition-all duration-300">
                ↓ scroll to explore
              </div>
            ) : null}
          </div>
        </section>

        <SectionDivider label="LIVE SYSTEM" />

        <RevealSection id="simulation" className="mt-10">
          <p className="font-mono text-xs tracking-[0.2em] text-sky-300">[ REAL-TIME CLUSTER TOPOLOGY ]</p>
          <AnimatedHeading words="Watch DIME Evolve Step by Step" />
          <TextGenerateEffect
            words="Native React Flow vectors, motion-interpolated node states, and live WebSocket telemetry from the Python simulator."
            className="mt-3 text-sm uppercase tracking-[0.14em] text-zinc-500"
          />
          <div className="mt-10">
            <ClusterSimulation />
          </div>
        </RevealSection>

        <SectionDivider label="DYNAMICS" />

        <RevealSection id="features" className="mt-8">
          <p className="font-mono text-xs tracking-[0.2em] text-orange-300">[ SIMULATION DYNAMICS ]</p>
          <AnimatedHeading words="What Makes DIME Hard" />
          <TextGenerateEffect words="Constraint-driven incidents with compounding latency and brittle recovery windows." className="mt-3 text-sm uppercase tracking-[0.14em] text-zinc-500" />

          <div className="mt-12">
            <FocusCards cards={FEATURE_CARDS} />
          </div>

          <div className="mt-20">
            <AnimatedHeading words="Four Tasks. One Unforgiving Benchmark." className="text-3xl sm:text-5xl" />
            <TextGenerateEffect words="Each task shifts failure modes so policies must adapt instead of memorizing." className="mt-3 text-sm uppercase tracking-[0.14em] text-zinc-500" />
            <div className="mt-9">
              <FocusCards cards={TASK_CARDS} className="lg:grid-cols-2" />
            </div>

            <div className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-2">
              {TASKS.map((task) => (
                <article key={task.name} className="rounded-xl border border-zinc-800 bg-zinc-950/90 p-5">
                  <div className="flex items-start justify-between gap-3">
                    <p className="text-base font-semibold text-zinc-100">{task.name}</p>
                    <DifficultyBadge difficulty={task.difficulty} />
                  </div>
                  <p className="mt-3 text-sm text-zinc-400">{task.description}</p>
                  <div className="mt-4">
                    <div className="flex items-center justify-between font-mono text-xs text-zinc-500">
                      <span>Llama-3.1-8B Baseline</span>
                      <span>{task.score.toFixed(3)}</span>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-zinc-900">
                      <div
                        className={`h-2 rounded-full ${task.score > 0.2 ? "bg-orange-400" : "bg-pink-500"}`}
                        style={{ width: `${Math.max(1, task.score * 100)}%` }}
                      />
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </RevealSection>

        <SectionDivider label="SCORING" />

        <RevealSection id="reward" className="mt-10">
          <p className="font-mono text-xs tracking-[0.2em] text-pink-300">[ REWARD SIGNAL ]</p>
          <AnimatedHeading words="How DIME Scores an Agent" />
          <TextGenerateEffect words="Dense per-step feedback rewards stability, penalizes latency, and discourages noisy interventions." className="mt-3 text-sm uppercase tracking-[0.14em] text-zinc-500" />

          <div className="mt-12 grid grid-cols-1 gap-8 lg:grid-cols-12">
            <div className="lg:col-span-7">
              <p className="max-w-3xl text-zinc-400">
                DIME uses a dense, step-level continuous reward signal, not sparse end-of-episode
                rewards. This means the agent gets feedback every step, making it trainable via RL.
              </p>
              <ul className="mt-6 space-y-4 text-zinc-400">
                <li>
                  <span className="font-mono text-zinc-100">+0.40 x uptime_ratio</span> - Keeps nodes
                  alive. The dominant signal.
                </li>
                <li>
                  <span className="font-mono text-zinc-100">-0.30 x normalized_latency</span> - Penalizes
                  slow responses proportional to severity.
                </li>
                <li>
                  <span className="font-mono text-zinc-100">-0.20 x overload_fraction</span> - Discourages
                  ignoring hot nodes.
                </li>
                <li>
                  <span className="font-mono text-zinc-100">-0.10 x (actions/max_steps)</span> - Action
                  efficiency penalty. Spamming actions is punished.
                </li>
                <li>
                  <span className="font-mono text-zinc-100">+0.50 x cascade_prevented_bonus</span> - The
                  highest bonus. Prevention rewarded over recovery.
                </li>
              </ul>
            </div>

            <div className="lg:col-span-5">
              <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 font-mono text-sm text-orange-200">
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
                  {[
                    "R(t) = + 0.40 x uptime_ratio",
                    "       - 0.30 x normalized_latency",
                    "       - 0.20 x overload_fraction",
                    "       - 0.10 x (actions_taken / max_steps)",
                    "       + 0.50 x cascade_prevented_bonus",
                  ].map((line, idx) => (
                    <div key={`row-${idx}`} className="contents">
                      <span className="text-zinc-600">{String(idx + 1).padStart(2, "0")}</span>
                      <span>{line}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </RevealSection>

        <SectionDivider label="METRICS" />

        <RevealSection id="metrics" className="mt-10">
          <p className="font-mono text-xs tracking-[0.2em] text-orange-300">[ EVALUATION METRICS ]</p>
          <AnimatedHeading words="Benchmark Diagnostics" />
          <TextGenerateEffect words="Core plots from DIME evaluation runs across failure, latency, and exploration regimes." className="mt-3 text-sm uppercase tracking-[0.14em] text-zinc-500" />

          <div className="mt-12 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {METRIC_FIGURES.map((figure) => (
              <article key={figure.src} className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/80">
                <img src={figure.src} alt={figure.title} loading="lazy" className="h-auto w-full object-cover" />
                <div className="p-4">
                  <p className="text-base font-semibold text-white">{figure.title}</p>
                  <p className="mt-2 text-sm text-zinc-400">{figure.caption}</p>
                </div>
              </article>
            ))}
          </div>
        </RevealSection>

        <SectionDivider label="ACCESS" />

        <RevealSection id="try" className="mt-10">
          <p className="font-mono text-xs tracking-[0.2em] text-pink-300">[ LIVE ACCESS ]</p>
          <div className="mt-5 max-w-4xl md:ml-10">
            <AnimatedHeading words="Try Your Hands On DIME" className="text-center md:text-left" />
            <TextGenerateEffect
              words="Use the hosted space for quick validation or run the container locally for controlled experiments."
              className="mt-3 text-center text-sm uppercase tracking-[0.14em] text-zinc-500 md:text-left"
            />
            <p className="mt-5 max-w-3xl text-center text-zinc-400 md:text-left">
              DIME is deployed as a containerized environment. You can interact with the live simulation
              via the Hugging Face Space or pull the Docker image to run it locally.
            </p>

            <div className="mt-8 flex flex-wrap items-center justify-center gap-3 md:justify-start">
              <a
                href="#"
                className="inline-flex min-w-[230px] items-center justify-center rounded-md bg-gradient-to-r from-orange-300 to-pink-300 px-5 py-3 text-sm font-semibold text-black"
              >
                Open on Hugging Face -&gt;
              </a>
              <a
                href="#"
                className="inline-flex items-center justify-center rounded-md border border-zinc-700 px-4 py-3 font-mono text-sm text-zinc-200"
              >
                Pull Docker Image
              </a>
            </div>

            <p className="mt-5 inline-block rounded border border-zinc-800 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-600">
              docker://registry-link-placeholder | hf://space-link-placeholder
            </p>

            <div className="mt-4 flex items-center gap-2 font-mono text-xs text-zinc-500">
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  DEPLOYMENT_LIVE ? "animate-pulse bg-emerald-400" : "bg-amber-400"
                }`}
              />
              {DEPLOYMENT_LIVE ? "Simulation Online" : "Coming Soon"}
            </div>
            <p className="mt-2 text-sm text-zinc-500">Links will be updated when the deployment is live.</p>
          </div>
        </RevealSection>
      </main>

      <footer className="border-t border-zinc-800 bg-zinc-950/35">
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-8 px-5 py-10 text-sm text-zinc-500 sm:px-8 lg:grid-cols-12">
          <div className="lg:col-span-6">
            <p className="font-mono text-zinc-300">DIME</p>
            <p className="mt-3 max-w-2xl leading-7">
              Distributed Infrastructure Management Environment is built for the OpenEnv Hackathon to
              benchmark LLM agents against realistic SRE incident-response dynamics.
            </p>
            <p className="mt-3 font-mono text-xs tracking-wide text-zinc-600">
              Co-organized by Meta, PyTorch and Hugging Face
            </p>
          </div>

          <div className="lg:col-span-3">
            <p className="font-mono text-xs tracking-[0.16em] text-zinc-400">BENCHMARK</p>
            <p className="mt-3 text-zinc-400">Environment: distributed_infra_env</p>
            <p className="mt-2 text-zinc-400">Modeled Tasks: 4 graded incidents</p>
            <p className="mt-2 text-zinc-400">Reward: dense step-level signal</p>
          </div>

          <div className="lg:col-span-3">
            <p className="font-mono text-xs tracking-[0.16em] text-zinc-400">RESOURCES</p>
            <div className="mt-3 flex flex-col gap-2">
              <a href="#" className="font-mono transition-colors hover:text-zinc-300">
                Source Code
              </a>
              <a href="#" className="font-mono transition-colors hover:text-zinc-300">
                Hugging Face Space
              </a>
              <a href="#" className="font-mono transition-colors hover:text-zinc-300">
                Docker Image
              </a>
            </div>
          </div>
        </div>
        <div className="border-t border-zinc-800/80 py-4 text-center font-mono text-xs text-zinc-600">
          © {new Date().getFullYear()} DIME · Incident intelligence for LLM systems
        </div>
      </footer>
    </div>
  );
}
