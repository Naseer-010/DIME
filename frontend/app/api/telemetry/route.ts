import { NextResponse } from "next/server";

import { getTelemetrySnapshots } from "@/lib/telemetry";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const snapshots = await getTelemetrySnapshots();
    return NextResponse.json(snapshots);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown telemetry error";
    return NextResponse.json(
      {
        error: "Failed to load telemetry",
        details: message,
      },
      { status: 500 }
    );
  }
}
