"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { FlipWords } from "@/components/ui/FlipWords";
import { TextGenerateEffect } from "@/components/ui/TextGenerateEffect";
import { ClusterSimulation } from "@/components/simulation/ClusterSimulation";
import type { DockItem } from "@/components/ui/types";

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
  grading: string;
};

type MetricFigure = {
  src: string;
  title: string;
  caption: string;
};

type DetailCard = {
  title: string;
  detail: string;
};

type FeatureRow = {
  title: string;
  descriptor: string;
  dotClassName: string;
};

type RewardTerm = {
  formula: string;
  label: string;
  weight: number;
  tone: string;
};

const HERO_FLIP_WORDS = ["adaptive", "resilient", "autonomous", "modern"];

const NAV_ITEMS: DockItem[] = [
  {
    title: "About",
    href: "#about",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M3 11.5 12 4l9 7.5" />
        <path d="M5.5 10.5V20h13V10.5" />
      </svg>
    ),
  },
  {
    title: "Metrics",
    href: "#metrics",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M5 19V9" />
        <path d="M12 19V5" />
        <path d="M19 19v-7" />
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
    title: "Diagnostics",
    href: "#diagnostics",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <rect x="3.5" y="3.5" width="17" height="17" rx="2.5" />
        <path d="M7 15l3-3 2 2 5-5" />
      </svg>
    ),
  },
  {
    title: "Try It",
    href: "#try-it",
    icon: ({ className }) => (
      <svg viewBox="0 0 24 24" className={className} fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="m5 19 14-7L5 5v5l8 2-8 2z" />
      </svg>
    ),
  },
];

const OBSERVABILITY_CARDS: DetailCard[] = [
  {
    title: "cpu_loads",
    detail:
      "Per-node CPU utilization in [0.0, 1.0], with -1.0 used when telemetry times out.",
  },
  {
    title: "queue_lengths",
    detail:
      "Pending request counts per node, which expose pressure before outright failure.",
  },
  {
    title: "latency_ms + p99_latency",
    detail:
      "Rolling end-to-end and tail latency so policies must control both average flow and spikes.",
  },
  {
    title: "task_hint + task_score",
    detail:
      "A natural-language objective hint plus a live benchmark score on every step.",
  },
];

const FEATURE_ROWS: FeatureRow[] = [
  {
    title: "Stochastic Traffic",
    descriptor: "Burst-prone Gaussian arrivals",
    dotClassName: "bg-amber-400",
  },
  {
    title: "Non-Linear Latency",
    descriptor: "Penalty grows with CPU²",
    dotClassName: "bg-sky-400",
  },
  {
    title: "Cascading Load Redistribution",
    descriptor: "Mesh neighbor collapse",
    dotClassName: "bg-white",
  },
  {
    title: "Deterministic Failure",
    descriptor: "90% CPU for 3 steps",
    dotClassName: "bg-red-400",
  },
];

const TASKS: TaskCard[] = [
  {
    name: "Traffic Spike Recovery",
    difficulty: "Easy",
    description: "",
    score: 0.09,
    grading: "",
  },
  {
    name: "Single Node Failure",
    difficulty: "Medium",
    description: "",
    score: 0.05,
    grading: "",
  },
  {
    name: "Cascading Failure Prevention",
    difficulty: "Hard",
    description: "",
    score: 0.31,
    grading: "",
  },
  {
    name: "Flash Crowd Meltdown",
    difficulty: "Expert",
    description: "",
    score: 0,
    grading: "",
  },
];

const REWARD_TERMS: RewardTerm[] = [
  { formula: "+0.40 × uptime_ratio", label: "dominant signal", weight: 0.4, tone: "bg-emerald-400" },
  { formula: "−0.30 × normalized_latency", label: "latency penalty", weight: 0.3, tone: "bg-red-400" },
  { formula: "−0.20 × overload_fraction", label: "hot-node penalty", weight: 0.2, tone: "bg-zinc-400" },
  { formula: "−0.10 × actions / max_steps", label: "anti-spam tax", weight: 0.1, tone: "bg-zinc-500" },
  { formula: "+0.50 × cascade_prevented_bonus", label: "prevention bonus", weight: 0.5, tone: "bg-emerald-300" },
];

const QUICKSTART_STEPS = [
  {
    label: "Install Dependencies",
    command: "pip install -r requirements.txt",
  },
  {
    label: "Start API Server",
    command: "uvicorn server.app:app --host 0.0.0.0 --port 8000",
  },
  {
    label: "Run Inference",
    command: "python inference.py",
  },
];

const METRIC_FIGURES: MetricFigure[] = [
  {
    src: "/metrics/fig1_vanishing_gradient_fix.png",
    title: "Fig 1: Vanishing Gradient Fix",
    caption: "Shows the reward-shaping change used to keep latency penalties informative instead of flattening into a dead training signal.",
  },
  {
    src: "/metrics/fig2_cascade_exploit_fix.png",
    title: "Fig 2: Cascade Exploit Fix",
    caption: "Illustrates the protection against oscillation-based reward farming in cascading-failure scenarios.",
  },
  {
    src: "/metrics/fig3_cost_latency_coupling.png",
    title: "Fig 3: Cost-Latency Coupling",
    caption: "Maps how capacity cost and service latency move together when the policy tries to stabilize the cluster efficiently.",
  },
  {
    src: "/metrics/fig4_curiosity_annealing.png",
    title: "Fig 4: Curiosity Annealing",
    caption: "Visualizes how intrinsic exploration pressure decays as the agent shifts from discovery toward stable control.",
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
      className={`landing-section scroll-mt-28 transition-all duration-700 motion-reduce:transform-none motion-reduce:opacity-100 ${
        visible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"
      } ${className ?? ""}`}
    >
      <div className="section-inner">{children}</div>
    </section>
  );
}

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="my-8 sm:my-10" aria-hidden="true">
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
    Easy: "bg-[#22c55e]/10 text-[#22c55e]",
    Medium: "bg-[#3b82f6]/10 text-[#3b82f6]",
    Hard: "bg-[#f97316]/10 text-[#f97316]",
    Expert: "bg-[#ef4444]/10 text-[#ef4444]",
  }[difficulty];

  return <span className={`rounded-full px-2.5 py-1 text-xs font-mono ${tone}`}>{difficulty}</span>;
}

function Spotlight() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10 overflow-x-clip">
      <div className="absolute left-1/2 top-[-12rem] h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(251,146,60,0.28),transparent_62%)]" />
      <div className="absolute right-[-11rem] top-12 h-[26rem] w-[26rem] rounded-full bg-[radial-gradient(circle,rgba(236,72,153,0.16),transparent_70%)]" />
      <div className="absolute bottom-[-12rem] left-[-10rem] h-[24rem] w-[24rem] rounded-full bg-[radial-gradient(circle,rgba(249,115,22,0.18),transparent_65%)]" />
    </div>
  );
}

function HeroWordmark() {
  return (
    <div className="mt-5 flex flex-wrap items-end gap-x-3 gap-y-2 text-[clamp(4.4rem,18vw,12rem)] font-black leading-[0.82] tracking-[-0.05em]">
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
      className={`mt-4 text-3xl font-black leading-[0.98] tracking-[-0.02em] text-zinc-100 [text-shadow:0_0_24px_rgba(244,114,182,0.18)] sm:text-5xl lg:text-6xl ${
        className ?? ""
      }`}
      duration={900}
      filter={false}
    />
  );
}

function TopNavigation() {
  return (
    <header className="fixed inset-x-0 top-4 z-50 px-3 sm:px-5">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-3 rounded-[1.6rem] border border-white/12 bg-black/55 px-3 py-3 shadow-[0_18px_80px_rgba(0,0,0,0.45)] backdrop-blur-2xl">
        <a href="#about" className="shrink-0 rounded-full border border-orange-400/20 bg-orange-500/10 px-3 py-2 font-mono text-xs tracking-[0.22em] text-orange-200">
          DIME
        </a>
        <nav className="hero-nav-scroll flex min-w-0 flex-1 items-center justify-end gap-1 overflow-x-auto">
          {NAV_ITEMS.map((item) => (
            <a
              key={`top-${item.href}-${item.title}`}
              href={item.href}
              className="group whitespace-nowrap rounded-full px-3 py-2 text-xs font-medium text-zinc-300 transition-all duration-300 hover:bg-white/8 hover:text-white"
            >
              <span className="relative">
                {item.title}
                <span className="absolute inset-x-0 -bottom-1 h-px origin-left scale-x-0 bg-gradient-to-r from-orange-300 via-pink-300 to-transparent transition-transform duration-300 group-hover:scale-x-100" />
              </span>
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}

function DetailGrid({ cards, columns = "lg:grid-cols-2" }: { cards: DetailCard[]; columns?: string }) {
  return (
    <div className={`grid grid-cols-1 gap-4 sm:grid-cols-2 ${columns}`}>
      {cards.map((card) => (
        <article
          key={card.title}
          className="group h-full rounded-2xl border border-zinc-800/85 bg-gradient-to-b from-zinc-950 to-zinc-950/70 p-4 shadow-[0_0_30px_rgba(251,146,60,0.04)] transition-all duration-300 hover:-translate-y-1 hover:border-zinc-700 hover:shadow-[0_18px_40px_rgba(0,0,0,0.35)] sm:p-5"
        >
          <div className="flex h-full flex-col gap-2.5">
            <p className="font-mono text-[11px] tracking-[0.16em] text-zinc-500">{card.title}</p>
            <p className="text-sm leading-5 text-zinc-300">{card.detail}</p>
          </div>
        </article>
      ))}
    </div>
  );
}

function FeatureEditorialList() {
  return (
    <div className="mt-12">
      {FEATURE_ROWS.map((feature) => (
        <div key={feature.title} className="mb-7">
          <div className="grid grid-cols-[auto_1fr_auto] items-center gap-4">
            <div className="flex items-center gap-3">
              <span className={`h-2 w-2 rounded-full ${feature.dotClassName}`} />
              <p className="text-2xl font-bold text-white md:text-3xl">{feature.title}</p>
            </div>
            <div className="border-b border-dotted border-zinc-800" />
            <p className="text-right font-mono text-sm text-zinc-500">{feature.descriptor}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function TaskBenchmarkList() {
  const { ref, visible } = useRevealOnScroll<HTMLDivElement>();

  return (
    <div ref={ref} className="mt-10 divide-y divide-zinc-900 border-y border-zinc-900">
      {TASKS.map((task, index) => (
        <article key={task.name} className="flex flex-col gap-4 py-5 md:flex-row md:items-center">
          <div className="flex min-w-0 flex-1 items-center gap-4">
            <span className="w-6 font-mono text-sm text-zinc-600">{String(index + 1).padStart(2, "0")}</span>
            <p className="min-w-0 text-lg font-medium text-white">{task.name}</p>
          </div>
          <div className="flex items-center justify-end gap-4">
            <DifficultyBadge difficulty={task.difficulty} />
            <div className="h-1 w-20 rounded-full bg-zinc-800">
              <div
                className="h-1 rounded-full bg-zinc-400 transition-[width] duration-1000 ease-out"
                style={{
                  width: visible ? `${Math.max(0, task.score * 100)}%` : "0%",
                  transitionDelay: `${index * 140}ms`,
                }}
              />
            </div>
            <span className="min-w-10 text-right font-mono text-xs text-zinc-500">{task.score.toFixed(2)}</span>
          </div>
        </article>
      ))}
    </div>
  );
}

function RewardWeightMeter() {
  return (
    <div className="space-y-8">
      {REWARD_TERMS.map((term) => (
        <div key={term.formula}>
          <p className="font-mono text-sm text-zinc-200">{term.formula}</p>
          <div className="mt-3 h-0.5 w-full max-w-xs bg-zinc-900">
            <div className={`h-0.5 rounded-full ${term.tone}`} style={{ width: `${(term.weight / 0.5) * 100}%` }} />
          </div>
          <p className="mt-2 font-mono text-xs text-zinc-500">{term.label}</p>
        </div>
      ))}
    </div>
  );
}

function useAnimatedTelemetry(): TelemetrySnapshot {
  const [snapshot, setSnapshot] = useState<TelemetrySnapshot>({
    step: 0,
    cpuLoads: [0.3, 0.25, 0.4, 0.35, 0.28, 0.32, 0.38, 0.22],
    queueLengths: [4, 6, 3, 8, 5, 7, 2, 4],
    latencyMs: 32.5,
    failedNodes: [],
    requestRate: 142,
  });

  useEffect(() => {
    let step = 0;
    const timer = setInterval(() => {
      step += 1;
      const t = step * 0.15;
      setSnapshot({
        step,
        cpuLoads: Array.from({ length: 8 }, (_, i) => {
          const load = Math.max(0.05, Math.min(0.95, 0.35 + 0.25 * Math.sin(t + i * 0.8) + (Math.random() - 0.5) * 0.06));
          return Number(load.toFixed(2));
        }),
        queueLengths: Array.from({ length: 8 }, (_, i) =>
          Math.max(0, Math.round(12 + 10 * Math.sin(t * 0.7 + i) + Math.random() * 3))
        ),
        latencyMs: Math.max(8, 28 + 16 * Math.sin(t * 0.5) + Math.random() * 6),
        failedNodes: step % 40 > 30 && step % 40 < 36 ? [Math.floor(Math.random() * 7) + 1] : [],
        requestRate: Math.max(60, 150 + 60 * Math.sin(t * 0.3) + Math.random() * 12),
      });
    }, 600);
    return () => clearInterval(timer);
  }, []);

  return snapshot;
}

function ScrollProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const value = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      setProgress(value);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="fixed inset-x-0 top-0 z-[70] h-[2px] bg-transparent">
      <div
        className="h-full bg-gradient-to-r from-orange-300 via-rose-300 to-amber-200 transition-[width] duration-200"
        style={{ width: `${progress * 100}%` }}
      />
    </div>
  );
}

export default function Home() {
  const telemetry = useAnimatedTelemetry();
  const [showScrollCue, setShowScrollCue] = useState(true);

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
      <ScrollProgress />
      <TopNavigation />

      <main className="relative pb-24">
        <section
          id="about"
          className="relative min-h-screen scroll-mt-28 overflow-x-clip pt-28 sm:pt-32"
        >
          <div className="section-inner">
            <div className="relative grid min-h-screen items-center gap-8 md:grid-cols-12">
              <Spotlight />

              <div className="md:col-span-7">
                <p className="font-mono text-xs tracking-[0.2em] text-orange-300">[ SRE BENCHMARK ]</p>
                <HeroWordmark />
                <p className="mt-4 max-w-2xl text-xl text-zinc-200 sm:text-2xl">Distributed infra benchmark for autonomous SRE agents.</p>
                <a
                  href="#try"
                  className="mt-8 inline-flex items-center rounded-md border border-orange-400/40 bg-orange-500/10 px-5 py-3 font-mono text-sm text-orange-100 transition-colors hover:border-pink-400/50 hover:bg-pink-500/10"
                >
                  Jump to Live Access -&gt;
                </a>

                <div className="mt-12 max-w-[460px] rounded-xl border border-zinc-700 bg-zinc-950/95 p-4 font-mono text-xs shadow-[0_0_40px_rgba(251,146,60,0.08)]">
                  <p className="text-emerald-300/80">● LIVE BASELINES</p>
                  <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px] sm:text-xs">
                    <div className="flex items-center gap-2">
                      <span className="text-zinc-600">#1</span>
                      <span className="text-white">Qwen3-8B</span>
                      <span className="text-emerald-300">0.31</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-zinc-600">#2</span>
                      <span className="text-white">Llama-3.1-8B</span>
                      <span className="text-amber-300">0.09</span>
                    </div>
                    <a href="#try-it" className="text-emerald-300/80 transition-colors hover:text-emerald-200">
                      submit your agent -&gt;
                    </a>
                  </div>
                </div>
              </div>

              <div className="md:col-span-5 md:pl-6">
                <div className="rounded-xl border border-zinc-800 bg-zinc-950/95 p-5 font-mono text-sm shadow-[0_0_40px_rgba(251,146,60,0.08)]">
                  <p className="text-zinc-500">● NODE STATUS  [step: {telemetry?.step ?? "--"}]</p>
                  <div className="mt-4 space-y-2">
                    <p className="grid grid-cols-[auto_1fr] gap-3 text-zinc-400">
                      <span>cpu_loads</span>
                      <span className="min-w-0 break-words text-emerald-300">[{telemetry ? telemetry.cpuLoads.join(", ") : "loading"}]</span>
                    </p>
                    <p className="grid grid-cols-[auto_1fr] gap-3 text-zinc-400">
                      <span>queue_lengths</span>
                      <span className="min-w-0 break-words text-orange-300">[{telemetry ? telemetry.queueLengths.join(", ") : "loading"}]</span>
                    </p>
                    <p className="grid grid-cols-[auto_1fr] gap-3 text-zinc-400">
                      <span>latency_ms</span>
                      <span className="text-pink-300">{telemetry ? `${telemetry.latencyMs.toFixed(1)}ms` : "loading"}</span>
                    </p>
                    <p className="grid grid-cols-[auto_1fr] gap-3 text-zinc-400">
                      <span>failed_nodes</span>
                      <span className="min-w-0 break-words text-red-300">[{telemetry ? telemetry.failedNodes.join(", ") : "loading"}]</span>
                    </p>
                    <p className="grid grid-cols-[auto_1fr] gap-3 text-zinc-400">
                      <span>request_rate</span>
                      <span className="text-amber-200">{telemetry ? `${telemetry.requestRate.toFixed(0)} req/s` : "loading"}</span>
                    </p>
                  </div>

                </div>
              </div>

              <div className="md:col-span-12 pt-2">
                {showScrollCue ? (
                  <div className="relative mt-4 text-center font-mono text-xs tracking-[0.18em] text-zinc-500 transition-all duration-300">
                    ↓ scroll to explore
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </section>

        <SectionDivider label="OBSERVABILITY" />

        <RevealSection id="metrics">
          <p className="font-mono text-xs tracking-[0.2em] text-emerald-300">[ AGENT INPUT METRICS ]</p>
          <AnimatedHeading words="The Signals an Agent Sees" />
          <TextGenerateEffect
            words="Observation fields come directly from the environment state, task curriculum, and partial-observability sandbox."
            className="mt-3 text-xs uppercase tracking-[0.14em] text-zinc-500 sm:text-sm"
          />
          <div className="mt-8 sm:mt-10">
            <DetailGrid cards={OBSERVABILITY_CARDS} />
          </div>
        </RevealSection>

        <SectionDivider label="LIVE SYSTEM" />

        <RevealSection id="simulation">
          <p className="font-mono text-xs tracking-[0.2em] text-sky-300">[ REAL-TIME CLUSTER TOPOLOGY ]</p>
          <AnimatedHeading words="Watch DIME Evolve Step by Step" />
          <TextGenerateEffect
            words="Native vector topology, motion-interpolated node states, and live-style telemetry from the Python simulator."
            className="mt-3 text-xs uppercase tracking-[0.14em] text-zinc-500 sm:text-sm"
          />
          <div className="mt-10">
            <ClusterSimulation />
          </div>
        </RevealSection>

        <SectionDivider label="DYNAMICS" />

        <RevealSection id="features">
          <p className="font-mono text-xs tracking-[0.2em] text-orange-300">[ SIMULATION DYNAMICS ]</p>
          <AnimatedHeading words="What Makes DIME Hard" />
          <FeatureEditorialList />

          <div className="mt-14 sm:mt-16">
            <AnimatedHeading words="Four Tasks. One Unforgiving Benchmark." className="text-3xl sm:text-5xl" />
            <TaskBenchmarkList />
          </div>
        </RevealSection>

        <SectionDivider label="SCORING" />

        <RevealSection id="reward">
          <p className="font-mono text-xs tracking-[0.2em] text-pink-300">[ REWARD SIGNAL ]</p>
          <AnimatedHeading words="How DIME Scores an Agent" />

          <div className="mt-12 grid grid-cols-1 gap-8 lg:grid-cols-12">
            <div className="lg:col-span-7">
              <RewardWeightMeter />
            </div>

            <div className="lg:col-span-5">
              <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 font-mono text-sm text-orange-200">
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2">
                  {[
                    "step reward in [-5.0, +5.0]",
                    "db failure => -5.0 terminal penalty",
                    ">=80% failed nodes => -4.0 penalty",
                    "rubric logs: format, stability, latency",
                    "rubric logs: cascade, efficiency, throughput",
                    "task graders compute live benchmark score",
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

        <RevealSection id="diagnostics">
          <p className="font-mono text-xs tracking-[0.2em] text-orange-300">[ BENCHMARK DIAGNOSTICS ]</p>
          <AnimatedHeading words="Benchmark Diagnostics" />
          <TextGenerateEffect
            words="Core plots from the repo show how DIME handles reward shaping, exploit resistance, cost-latency coupling, and exploration schedules."
            className="mt-3 text-xs uppercase tracking-[0.14em] text-zinc-500 sm:text-sm"
          />

          <div className="mt-12 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {METRIC_FIGURES.map((figure) => (
              <article key={figure.src} className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/80">
                <Image
                  src={figure.src}
                  alt={figure.title}
                  width={1100}
                  height={680}
                  loading="lazy"
                  sizes="(min-width: 1024px) 50vw, 100vw"
                  className="h-auto w-full object-cover"
                />
                <div className="p-4">
                  <p className="text-base font-semibold text-white">{figure.title}</p>
                  <p className="mt-2 text-sm text-zinc-400">{figure.caption}</p>
                </div>
              </article>
            ))}
          </div>
        </RevealSection>

        <SectionDivider label="INSTALL" />

        <RevealSection id="try-it">
          <p className="font-mono text-xs tracking-[0.2em] text-emerald-300">[ QUICKSTART ]</p>
          <AnimatedHeading words="Install + Run DIME" className="text-3xl sm:text-5xl" />
          <p className="mt-5 font-mono text-xs text-zinc-500">
            install deps  →  start server  →  run evaluation
          </p>
          <div className="mt-8 grid grid-cols-1 gap-3 md:grid-cols-3">
            {QUICKSTART_STEPS.map((step, index) => (
              <article
                key={step.label}
                className="rounded-xl border border-zinc-800 bg-zinc-950/75 p-4 shadow-[0_8px_30px_rgba(0,0,0,0.28)] transition-colors duration-300 hover:border-zinc-700"
              >
                <p className="font-mono text-[11px] tracking-[0.14em] text-zinc-500">
                  STEP {String(index + 1).padStart(2, "0")}
                </p>
                <p className="mt-2 text-sm font-semibold text-zinc-100">{step.label}</p>
                <div className="mt-3 rounded-md border border-zinc-800 bg-black/50 px-3 py-2 font-mono text-xs text-zinc-300">
                  {step.command}
                </div>
              </article>
            ))}
          </div>
        </RevealSection>

        <SectionDivider label="ACCESS" />

        <RevealSection id="try">
          <p className="font-mono text-xs tracking-[0.2em] text-pink-300">[ LIVE ACCESS ]</p>
          <div className="mt-5 max-w-4xl md:ml-10">
            <AnimatedHeading words="Try DIME" className="text-center md:text-left" />

            <div className="mt-8 flex flex-wrap items-center justify-center gap-3 md:justify-start">
              <button
                type="button"
                disabled
                className="inline-flex min-w-[230px] cursor-not-allowed items-center justify-center rounded-md bg-gradient-to-r from-orange-300 to-pink-300 px-5 py-3 text-sm font-semibold text-black opacity-80"
              >
                Hugging Face Space
              </button>
              <a
                href="#try-it"
                className="inline-flex min-w-[180px] items-center justify-center rounded-md border border-zinc-700 px-5 py-3 font-mono text-sm text-zinc-200 transition-colors hover:border-zinc-500 hover:text-white"
              >
                Run via Docker
              </a>
            </div>

            <div className="mt-6 inline-flex whitespace-pre-wrap rounded-xl border border-zinc-800 bg-zinc-950 px-4 py-3 font-mono text-xs text-zinc-400">
              docker build -t distributed-infra-env .{"\n"}docker run -p 8000:8000 distributed-infra-env
            </div>
          </div>
        </RevealSection>
      </main>

      <div className="mx-auto max-w-7xl px-5 sm:px-8">
        <div className="border-t border-zinc-800" />
      </div>

      <footer className="bg-zinc-950/35">
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-8 px-5 py-10 text-sm text-zinc-500 sm:px-8 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <p className="font-mono text-zinc-300">DIME</p>
            <p className="mt-3 max-w-2xl leading-7">
              Distributed Infrastructure Management Environment was built for the Meta Hackathon in
              Bangalore 2026 under Meta and OpenEnv, with Hugging Face as the deployment surface.
            </p>
          </div>

          <div className="lg:col-span-5">
            <p className="font-mono text-xs tracking-[0.16em] text-zinc-400">RESOURCES</p>
            <div className="mt-3 flex flex-col gap-2">
              <a href="#about" className="font-mono transition-colors hover:text-zinc-300">
                About DIME
              </a>
              <a href="#reward" className="font-mono transition-colors hover:text-zinc-300">
                Scoring Model
              </a>
              <a href="#try" className="font-mono transition-colors hover:text-zinc-300">
                Hugging Face Access
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
