"use client";

import { cn } from "./cn";

type TextHoverEffectProps = {
  text: string;
  className?: string;
  duration?: number;
};

export function TextHoverEffect({ text, className, duration = 500 }: TextHoverEffectProps) {
  return (
    <div
      className={cn("group relative inline-block select-none", className)}
      style={{
        // Keep animation timing configurable without external dependencies.
        ["--th-duration" as string]: `${duration}ms`,
      }}
    >
      <span className="relative z-0 text-transparent [text-shadow:0_0_0_#ffffff] [-webkit-text-stroke:1px_rgba(255,255,255,0.5)]">
        {text}
      </span>
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-10 overflow-hidden text-white [clip-path:inset(0_100%_0_0)] transition-[clip-path] duration-[var(--th-duration)] ease-out group-hover:[clip-path:inset(0_0%_0_0)]"
      >
        {text}
      </span>
      <style jsx>{`
        .group:hover {
          filter: none;
        }
        .group * {
          opacity: 1;
        }
      `}</style>
    </div>
  );
}

export type { TextHoverEffectProps };
export default TextHoverEffect;
