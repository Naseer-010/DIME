"use client";

import { useEffect, useRef } from "react";
import { cn } from "./cn";

type SpotlightProps = {
  className?: string;
  fill?: string;
};

export function Spotlight({ className, fill = "255,255,255" }: SpotlightProps) {
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = rootRef.current;
    if (!node) return;

    let frame = 0;
    let t = 0;
    let pointerActive = false;

    const setPos = (x: number, y: number) => {
      node.style.setProperty("--spot-x", `${x}%`);
      node.style.setProperty("--spot-y", `${y}%`);
    };

    setPos(50, 35);

    const onMove = (event: PointerEvent) => {
      const rect = node.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      pointerActive = true;
      const x = ((event.clientX - rect.left) / rect.width) * 100;
      const y = ((event.clientY - rect.top) / rect.height) * 100;
      setPos(Math.min(100, Math.max(0, x)), Math.min(100, Math.max(0, y)));
    };

    const animate = () => {
      if (!pointerActive) {
        t += 0.008;
        const x = 50 + Math.sin(t * 0.9) * 14;
        const y = 38 + Math.cos(t * 0.7) * 10;
        setPos(x, y);
      }
      frame = window.requestAnimationFrame(animate);
    };

    const resetPointer = () => {
      pointerActive = false;
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerleave", resetPointer);
    window.addEventListener("blur", resetPointer);
    frame = window.requestAnimationFrame(animate);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", resetPointer);
      window.removeEventListener("blur", resetPointer);
    };
  }, []);

  return (
    <div
      ref={rootRef}
      aria-hidden="true"
      className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)}
      style={{
        backgroundImage: `radial-gradient(42rem 42rem at var(--spot-x,50%) var(--spot-y,35%), rgba(${fill},0.18), rgba(${fill},0.07) 35%, rgba(${fill},0.02) 55%, transparent 70%)`,
      }}
    />
  );
}

export type { SpotlightProps };
export default Spotlight;
