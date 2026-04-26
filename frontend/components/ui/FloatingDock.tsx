"use client";

import type { ReactNode } from "react";
import { cn } from "./cn";
import type { DockItem } from "./types";

type FloatingDockProps = {
  items: DockItem[];
  className?: string;
  mobileClassName?: string;
};

function renderIcon(icon: DockItem["icon"], className: string) {
  if (typeof icon === "function") {
    return icon({ className });
  }

  return <span className={className}>{icon as ReactNode}</span>;
}

export function FloatingDock({ items, className, mobileClassName }: FloatingDockProps) {
  return (
    <>
      <div
        className={cn(
          "fixed bottom-6 left-1/2 z-50 hidden -translate-x-1/2 items-center gap-1 rounded-2xl border border-white/15 bg-black/70 p-2 shadow-2xl backdrop-blur-xl sm:flex",
          className
        )}
      >
        {items.map((item) => (
          <a
            key={`${item.href}-${item.title}`}
            href={item.href}
            className="group relative inline-flex h-11 w-11 items-center justify-center rounded-xl text-white/80 transition-colors hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/60"
            aria-label={item.title}
            title={item.title}
          >
            {renderIcon(item.icon, "h-5 w-5")}
            <span className="pointer-events-none absolute -top-9 left-1/2 -translate-x-1/2 rounded-md border border-white/10 bg-black px-2 py-1 text-[11px] text-white opacity-0 transition-opacity group-hover:opacity-100">
              {item.title}
            </span>
          </a>
        ))}
      </div>

      <div
        className={cn(
          "fixed inset-x-4 bottom-4 z-50 flex items-center justify-around rounded-2xl border border-white/15 bg-black/75 px-3 py-2 shadow-2xl backdrop-blur-xl sm:hidden",
          mobileClassName
        )}
      >
        {items.map((item) => (
          <a
            key={`mobile-${item.href}-${item.title}`}
            href={item.href}
            className="inline-flex min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-lg px-1 py-1 text-white/80 transition-colors hover:bg-white/10 hover:text-white"
          >
            {renderIcon(item.icon, "h-5 w-5")}
            <span className="max-w-full truncate text-[10px] leading-none">{item.title}</span>
          </a>
        ))}
      </div>
    </>
  );
}

export type { FloatingDockProps };
export default FloatingDock;
