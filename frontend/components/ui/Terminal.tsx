"use client";

import { useEffect, useMemo, useState } from "react";
import { cn } from "./cn";

type TerminalProps = {
  commands: string[];
  outputs: string[];
  typingSpeed?: number;
  delayBetweenCommands?: number;
  className?: string;
};

type RenderedLine = {
  type: "command" | "output";
  text: string;
};

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

export function Terminal({
  commands,
  outputs,
  typingSpeed = 26,
  delayBetweenCommands = 450,
  className,
}: TerminalProps) {
  const [lines, setLines] = useState<RenderedLine[]>([]);
  const [activeLine, setActiveLine] = useState<RenderedLine | null>(null);

  const script = useMemo(() => {
    const merged: RenderedLine[] = [];
    const max = Math.max(commands.length, outputs.length);

    for (let i = 0; i < max; i += 1) {
      if (commands[i]) merged.push({ type: "command", text: commands[i] });
      if (outputs[i]) merged.push({ type: "output", text: outputs[i] });
    }

    return merged;
  }, [commands, outputs]);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      setLines([]);
      setActiveLine(null);

      for (let i = 0; i < script.length; i += 1) {
        const line = script[i];
        const value = line.text ?? "";

        for (let c = 0; c <= value.length; c += 1) {
          if (cancelled) return;
          setActiveLine({ type: line.type, text: value.slice(0, c) });
          await sleep(typingSpeed);
        }

        if (cancelled) return;
        setLines((prev) => [...prev, line]);
        setActiveLine(null);
        await sleep(delayBetweenCommands);
      }
    };

    run();

    return () => {
      cancelled = true;
    };
  }, [script, typingSpeed, delayBetweenCommands]);

  const renderLine = (line: RenderedLine, key: string, active = false) => {
    const isCommand = line.type === "command";
    return (
      <div
        key={key}
        className={cn(
          "whitespace-pre-wrap break-words font-mono text-sm leading-relaxed",
          isCommand ? "text-emerald-300" : "text-zinc-200"
        )}
      >
        {isCommand ? "$ " : ""}
        {line.text}
        {active ? <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-white/90 align-middle" /> : null}
      </div>
    );
  };

  return (
    <div
      className={cn(
        "rounded-xl border border-white/10 bg-[#070707] p-4 shadow-xl",
        className
      )}
    >
      <div className="mb-3 flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full bg-red-400/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-yellow-300/80" />
        <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
      </div>

      <div className="space-y-1">
        {lines.map((line, index) => renderLine(line, `line-${index}`))}
        {activeLine ? renderLine(activeLine, "active", true) : null}
      </div>
    </div>
  );
}

export type { TerminalProps };
export default Terminal;
