"use client";

import { useEffect, useMemo, useState } from "react";
import { cn } from "./cn";

type FlipWordsProps = {
  words: readonly string[];
  className?: string;
  duration?: number;
};

export function FlipWords({ words, className, duration = 2500 }: FlipWordsProps) {
  const normalizedWords = useMemo(
    () => words.map((word) => word.trim()).filter(Boolean),
    [words]
  );
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (normalizedWords.length <= 1) return;

    const interval = window.setInterval(() => {
      setActiveIndex((current) => (current + 1) % normalizedWords.length);
    }, Math.max(400, duration));

    return () => window.clearInterval(interval);
  }, [duration, normalizedWords]);

  if (!normalizedWords.length) {
    return null;
  }

  const visibleIndex = activeIndex % normalizedWords.length;

  return (
    <span className={cn("inline-flex items-center", className)}>
      <span aria-live="polite" aria-atomic="true" className="sr-only">
        {normalizedWords[visibleIndex]}
      </span>
      <span aria-hidden="true" className="relative inline-flex h-[1.2em] overflow-hidden">
        <span
          className="flex flex-col transition-transform duration-500 ease-out will-change-transform"
          style={{ transform: `translateY(-${visibleIndex * 100}%)` }}
        >
          {normalizedWords.map((word, index) => (
            <span
              key={`${word}-${index}`}
              className={cn(
                "block leading-[1.2] transition-opacity duration-300",
                index === visibleIndex ? "opacity-100" : "opacity-35"
              )}
            >
              {word}
            </span>
          ))}
        </span>
      </span>
    </span>
  );
}

export type { FlipWordsProps };
export default FlipWords;
