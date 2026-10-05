
import { Badge, Group, Tooltip } from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import { invoke } from "@tauri-apps/api/core";
import {
  isPermissionGranted,
  requestPermission,
  sendNotification,
} from "@tauri-apps/plugin-notification";
import { useEffect, useRef, useState } from "react";

type Health = {
  mode: "paper" | "conservative" | "balanced" | "growth" | "liquid";
  degraded: boolean;
  degraded_reasons: string[];
  model_health_score: number;
  model_fallback_active: boolean;
};

type Inspector = {
  alerts: Array<{
    id: string;
    severity: string;
    code: string;
    message: string;
  }>;
};

export function AlgoFrameStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [inspector, setInspector] = useState<Inspector | null>(null);
  const [notificationsEnabled] = useLocalStorage<boolean>({
    key: "algoframe-notifications-enabled",
    defaultValue: true,
  });
  const seenAlerts = useRef<Set<string>>(new Set());

  useEffect(() => {
    let mounted = true;

    const refresh = async () => {
      try {
        const [nextHealth, nextInspector] = await Promise.all([
          invoke<Health>("algoframe_product_health"),
          invoke<Inspector>("learning_get_inspector"),
        ]);

        if (!mounted) return;

        setHealth(nextHealth);
        setInspector(nextInspector);
      } catch {
        // The global status indicator must never break the app shell.
      }
    };

    refresh();
    const timer = window.setInterval(refresh, 10_000);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!notificationsEnabled || !inspector) return;

    const important = inspector.alerts.filter(
      (alert) =>
        alert.severity === "critical" ||
        alert.severity === "warning",
    );

    const fresh = important.filter((alert) => {
      if (seenAlerts.current.has(alert.id)) return false;
      seenAlerts.current.add(alert.id);
      return true;
    });

    if (fresh.length === 0) return;

    const notify = async () => {
      let granted = await isPermissionGranted();

      if (!granted) {
        granted = (await requestPermission()) === "granted";
      }

      if (!granted) return;

      for (const alert of fresh.slice(0, 3)) {
        sendNotification({
          title: `AlgoFrame · ${alert.code}`,
          body: alert.message,
        });
      }
    };

    notify().catch(() => undefined);
  }, [inspector, notificationsEnabled]);

  if (!health) {
    return (
      <Badge variant="light" color="gray">
        AlgoFrame
      </Badge>
    );
  }

  const modeColor =
    health.mode === "paper"
      ? "yellow"
      : health.degraded
        ? "red"
        : health.model_fallback_active
          ? "orange"
          : "green";

  const tooltip = [
    `Mode: ${health.mode}`,
    `Model health: ${(health.model_health_score * 100).toFixed(0)}%`,
    ...health.degraded_reasons,
  ].join("\n");

  return (
    <Tooltip label={tooltip} multiline>
      <Group gap={6}>
        <Badge color={modeColor} variant="filled">
          {health.mode === "paper" ? "PAPER MODE" : health.mode.toUpperCase()}
        </Badge>
        <Badge
          variant="light"
          color={health.degraded ? "red" : "green"}
        >
          {(health.model_health_score * 100).toFixed(0)}%
        </Badge>
      </Group>
    </Tooltip>
  );
}
