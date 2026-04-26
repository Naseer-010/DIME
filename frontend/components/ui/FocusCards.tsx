"use client";

import Image from "next/image";
import { useState } from "react";
import { cn } from "./cn";
import type { FocusCard } from "./types";

type FocusCardsProps = {
  cards: FocusCard[];
  className?: string;
};

export function FocusCards({ cards, className }: FocusCardsProps) {
  const [focusedIndex, setFocusedIndex] = useState<number | null>(null);

  return (
    <div
      className={cn("grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3", className)}
      onMouseLeave={() => setFocusedIndex(null)}
    >
      {cards.map((card, index) => {
        const focused = focusedIndex === index;
        const blurred = focusedIndex !== null && !focused;

        return (
          <button
            type="button"
            key={`${card.title}-${card.src}-${index}`}
            onMouseEnter={() => setFocusedIndex(index)}
            onFocus={() => setFocusedIndex(index)}
            onBlur={() => setFocusedIndex(null)}
            className={cn(
              "group relative h-64 overflow-hidden rounded-2xl border border-white/10 text-left transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70",
              focused ? "scale-[1.02] border-white/30 shadow-2xl" : "scale-100",
              blurred ? "opacity-60" : "opacity-100"
            )}
          >
            <Image
              src={card.src}
              alt={card.title}
              fill
              sizes="(min-width: 1024px) 33vw, (min-width: 640px) 50vw, 100vw"
              className={cn(
                "h-full w-full object-cover transition-transform duration-500",
                focused ? "scale-110" : "scale-100"
              )}
              loading="lazy"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/25 to-transparent" />
            <p className="absolute bottom-4 left-4 right-4 text-sm font-semibold text-white sm:text-base">
              {card.title}
            </p>
          </button>
        );
      })}
    </div>
  );
}

export type { FocusCardsProps };
export default FocusCards;
