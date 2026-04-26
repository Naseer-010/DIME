"use client";

import React from "react";

type BackgroundBeamsProps = {
  className?: string;
};

const BEAMS = [
  { left: "10%", delay: "0s", duration: "22s" },
  { left: "24%", delay: "4s", duration: "24s" },
  { left: "38%", delay: "8s", duration: "20s" },
  { left: "52%", delay: "12s", duration: "25s" },
  { left: "66%", delay: "16s", duration: "21s" },
  { left: "80%", delay: "20s", duration: "23s" },
];

const SPARKLES = [
  { left: "14%", top: "22%", delay: "2s", duration: "8s" },
  { left: "36%", top: "36%", delay: "4.5s", duration: "10s" },
  { left: "52%", top: "28%", delay: "6.5s", duration: "9s" },
  { left: "74%", top: "34%", delay: "3.5s", duration: "11s" },
  { left: "82%", top: "20%", delay: "5.5s", duration: "8.5s" },
];

export function BackgroundBeams({ className }: BackgroundBeamsProps) {
  return (
    <div
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 overflow-hidden ${className ?? ""}`}
    >
      <div className="absolute inset-y-0 left-0 w-24 bg-gradient-to-r from-[#080808] via-[#080808]/70 to-transparent" />
      <div className="absolute inset-y-0 right-0 w-24 bg-gradient-to-l from-[#080808] via-[#080808]/70 to-transparent" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_26%,rgba(255,255,255,0.12),transparent_56%)]" />
      <div className="absolute inset-x-0 top-0 h-2/3 bg-gradient-to-b from-orange-200/[0.08] via-transparent to-transparent" />

      {BEAMS.map((beam) => (
        <span
          key={`beam-${beam.left}`}
          className="absolute top-[-20%] h-[140%] w-[2px] bg-gradient-to-b from-transparent via-orange-100/70 to-transparent"
          style={{
            left: beam.left,
            boxShadow: "0 0 12px rgba(251, 146, 60, 0.25)",
            animationName: "heroBeamDrift",
            animationDuration: beam.duration,
            animationDelay: beam.delay,
            animationIterationCount: "infinite",
            animationTimingFunction: "linear",
          }}
        />
      ))}

      {SPARKLES.map((sparkle) => (
        <span
          key={`sparkle-${sparkle.left}-${sparkle.top}`}
          className="absolute h-1.5 w-1.5 rounded-full bg-orange-100/90"
          style={{
            left: sparkle.left,
            top: sparkle.top,
            boxShadow: "0 0 18px rgba(251, 146, 60, 0.65)",
            animationName: "heroSparklePulse",
            animationDuration: sparkle.duration,
            animationDelay: sparkle.delay,
            animationIterationCount: "infinite",
            animationTimingFunction: "ease-in-out",
          }}
        />
      ))}
    </div>
  );
}

export type { BackgroundBeamsProps };
export default BackgroundBeams;
