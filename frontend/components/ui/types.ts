import type { ReactNode } from "react";

export type DockItem = {
  title: string;
  href: string;
  icon: ReactNode | ((props: { className?: string }) => ReactNode);
};

export type FocusCard = {
  title: string;
  src: string;
};
