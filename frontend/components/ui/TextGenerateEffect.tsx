"use client";

import { useEffect, useMemo, useState } from "react";
import { cn } from "./cn";

type TextGenerateEffectProps = {
  words: string;
  className?: string;
  duration?: number;
  filter?: boolean;
};

export function TextGenerateEffect({
  words,
  className,
  duration = 700,
  filter = true,
}: TextGenerateEffectProps) {
  const tokens = useMemo(() => words.trim().split(/\s+/).filter(Boolean), [words]);
  const [visibleCount, setVisibleCount] = useState(0);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduceMotion(media.matches);
    const timer = window.setTimeout(sync, 0);
    media.addEventListener("change", sync);
    return () => {
      window.clearTimeout(timer);
      media.removeEventListener("change", sync);
    };
  }, []);

  useEffect(() => {
    if (!tokens.length || reduceMotion) return;

    const perWordDelay = Math.max(35, Math.floor(duration / tokens.length));
    const timer = window.setInterval(() => {
      setVisibleCount((current) => {
        if (current >= tokens.length) {
          window.clearInterval(timer);
          return current;
        }
        return current + 1;
      });
    }, perWordDelay);

    return () => window.clearInterval(timer);
  }, [tokens, duration, reduceMotion]);

  const renderedCount = reduceMotion ? tokens.length : visibleCount;

  return (
    <p className={cn("flex flex-wrap text-zinc-100", className)}>
      {tokens.map((word, index) => {
        const visible = index < renderedCount;
        return (
          <span
            key={`${word}-${index}`}
            className={cn(
              "mr-1.5 inline-block transition-all duration-300",
              visible
                ? "translate-y-0 opacity-100 blur-none"
                : cn("translate-y-1 opacity-0", filter ? "blur-[2px]" : "blur-none")
            )}
          >
            {word}
          </span>
        );
      })}
    </p>
  );
}

export type { TextGenerateEffectProps };
export default TextGenerateEffect;
