
import api from "@api/index";
import {
  Alert,
  Badge,
  Button,
  Card,
  Container,
  Divider,
  Grid,
  Group,
  Modal,
  NumberFormatter,
  NumberInput,
  Progress,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Table,
  Tabs,
  Text,
  TextInput,
  Title,
  Tooltip,
} from "@mantine/core";
import { useLocalStorage } from "@mantine/hooks";
import { invoke } from "@tauri-apps/api/core";
import {
  ArcElement,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip as ChartTooltip,
} from "chart.js";
import { Doughnut, Line } from "react-chartjs-2";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Filler,
  ChartTooltip,
  Legend,
);

type ProductHealth = {
  checked_at: string;
  mode: "paper" | "conservative" | "balanced" | "growth" | "liquid";
  database_integrity: string;
  database_size_bytes: number;
  snapshot_count: number;
  decision_count: number;
  outcome_count: number;
  alert_count: number;
  latest_snapshot_at?: string | null;
  latest_decision_at?: string | null;
  snapshot_age_minutes?: number | null;
  model_health_score: number;
  model_fallback_active: boolean;
  model_healthy: boolean;
  degraded: boolean;
  degraded_reasons: string[];
};

type ProductBackup = {
  name: string;
  path: string;
  size_bytes: number;
  created_at: string;
};

type ProductProfile = {
  name: string;
  config: any;
  updated_at: string;
};

type ReliabilityConfig = {
  enabled: boolean;
  circuit_breaker_enabled: boolean;
  minimum_model_health: number;
  minimum_data_quality: number;
  maximum_anomaly_score: number;
  execution_failure_window_minutes: number;
  maximum_execution_failures: number;
  maximum_actions_per_minute: number;
  maximum_daily_realized_loss: number;
  prepared_timeout_minutes: number;
  automatic_recovery_minutes: number;
  release_gate_enabled: boolean;
  release_gate_minimum_samples: number;
  release_gate_minimum_reward_uplift_pct: number;
  release_gate_max_failure_delta: number;
  release_gate_max_drawdown: number;
  release_gate_minimum_walk_forward_stability: number;
  release_gate_require_no_leakage: boolean;
  release_gate_minimum_safe_stress_fraction: number;
};

type ReliabilityStatus = {
  config: ReliabilityConfig;
  breaker: {
    tripped: boolean;
    reasons: string[];
    manual_trip: boolean;
    recent_failures: number;
    recent_actions: number;
    unknown_executions: number;
    daily_realized_loss: number;
    checked_at: string;
  };
  recent_executions: Array<{
    id: string;
    decision_id: string;
    item_key: string;
    wfm_id: string;
    side: string;
    operation: string;
    target_price: number;
    target_quantity: number;
    previous_price: number;
    state: string;
    attempt_count: number;
    error: string;
    created_at: string;
    updated_at: string;
  }>;
  pending_executions: number;
  lifecycle_events: number;
  last_release_gate?: any | null;
};

type FakeMarketSuite = {
  passed: boolean;
  passed_count: number;
  total_count: number;
  generated_at: string;
  scenarios: Array<{
    name: string;
    passed: boolean;
    final_state: string;
    notes: string[];
  }>;
};

type Inspector = {
  model: string;
  mode: ProductHealth["mode"];
  health: {
    score: number;
    healthy: boolean;
    fallback_active: boolean;
    reasons: string[];
    recent_reward: number;
    baseline_reward: number;
    recent_failure_rate: number;
    recent_prediction_mae: number;
    calibration_error: number;
    drawdown: number;
  };
  metrics: {
    sample_count: number;
    average_reward: number;
    failure_rate: number;
    prediction_mae: number;
    calibration_error: number;
    profit_per_hour: number;
  };
  portfolio: {
    total_capital_budget: number;
    reserved_capital: number;
    allocated_capital: number;
    high_volatility_capital: number;
    experimental_capital: number;
    item_exposure: Record<string, number>;
    family_exposure: Record<string, number>;
    category_exposure: Record<string, number>;
    estimated_daily_interactions: number;
  };
  alerts: Array<{
    id: string;
    severity: string;
    code: string;
    message: string;
    item_key?: string | null;
    created_at: string;
  }>;
  arbitrage_opportunities: Array<any>;
  top_items: Array<any>;
  active_events: Array<any>;
  decisions_total: number;
  decisions_completed: number;
  decisions_failed: number;
  decisions_open: number;
  snapshots_recorded: number;
  transactions_learned: number;
  config: any;
  advanced?: any;
};

const fmtPlat = (value: unknown, digits = 0) => {
  const number = Number(value ?? 0);
  return `${number.toFixed(digits)}p`;
};

const pct = (value: unknown, digits = 0) =>
  `${(Number(value ?? 0) * 100).toFixed(digits)}%`;

const fmtNumber = (value: unknown, digits = 2) =>
  Number(value ?? 0).toFixed(digits);

const fmtDate = (value: unknown) => {
  if (!value) return "—";
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
};

const pretty = (value: unknown) =>
  String(value ?? "")
    .split("_")
    .join(" ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const severityColor = (severity: string) => {
  switch (severity) {
    case "critical":
      return "red";
    case "warning":
      return "yellow";
    default:
      return "blue";
  }
};

const statusColor = (status: string) => {
  if (status.includes("complete") || status === "filled") return "green";
  if (status.includes("reject") || status.includes("expire") || status.includes("cancel"))
    return "red";
  if (status.includes("paper")) return "yellow";
  return "blue";
};

function MetricCard({
  label,
  value,
  detail,
  progress,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  progress?: number;
}) {
  return (
    <Card withBorder>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={700} size="xl">
        {value}
      </Text>
      {progress != null && (
        <Progress mt="xs" value={Math.max(0, Math.min(100, progress))} />
      )}
      {detail && (
        <Text size="xs" c="dimmed" mt={4}>
          {detail}
        </Text>
      )}
    </Card>
  );
}

function ExplanationModal({
  decision,
  onClose,
  onReplay,
}: {
  decision: any | null;
  onClose: () => void;
  onReplay: (decision: any) => void;
}) {
  const explanation = decision?.explanation ?? {};

  return (
    <Modal
      opened={Boolean(decision)}
      onClose={onClose}
      title={decision ? `${decision.item_name} · Decision` : "Decision"}
      size="xl"
    >
      {decision && (
        <Stack>
          <Group>
            <Badge color={decision.side === "buy" ? "blue" : "grape"}>
              {pretty(decision.side)}
            </Badge>
            <Badge color={statusColor(String(decision.status))}>
              {pretty(decision.status)}
            </Badge>
            <Badge variant="light">{pretty(decision.regime)}</Badge>
            <Badge variant="light">{pretty(decision.policy_name)}</Badge>
          </Group>

          <Title order={4}>
            {explanation.headline || `${pretty(decision.chosen_action)} at ${decision.price}p`}
          </Title>

          <SimpleGrid cols={{ base: 2, md: 4 }}>
            <MetricCard label="Price" value={fmtPlat(decision.price)} />
            <MetricCard label="Quantity" value={decision.quantity ?? 0} />
            <MetricCard
              label="Expected profit"
              value={fmtPlat(decision.predicted_profit, 2)}
            />
            <MetricCard
              label="1h fill"
              value={pct(decision.predicted_fill?.fill_1h)}
            />
          </SimpleGrid>

          <Grid>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Card withBorder>
                <Text fw={600}>Why AlgoFrame liked it</Text>
                <Stack gap={4} mt="xs">
                  {(explanation.positive_factors ?? []).length === 0 && (
                    <Text size="sm" c="dimmed">
                      No positive factors were recorded.
                    </Text>
                  )}
                  {(explanation.positive_factors ?? []).map(
                    (factor: string, index: number) => (
                      <Text size="sm" key={`${factor}:${index}`}>
                        + {factor}
                      </Text>
                    ),
                  )}
                </Stack>
              </Card>
            </Grid.Col>

            <Grid.Col span={{ base: 12, md: 6 }}>
              <Card withBorder>
                <Text fw={600}>Risks / blockers</Text>
                <Stack gap={4} mt="xs">
                  {(explanation.negative_factors ?? []).length === 0 &&
                    (explanation.guardrails ?? []).length === 0 && (
                      <Text size="sm" c="dimmed">
                        No blockers were recorded.
                      </Text>
                    )}
                  {(explanation.negative_factors ?? []).map(
                    (factor: string, index: number) => (
                      <Text size="sm" key={`${factor}:${index}`} c="orange">
                        − {factor}
                      </Text>
                    ),
                  )}
                  {(explanation.guardrails ?? []).map(
                    (factor: string, index: number) => (
                      <Text size="sm" key={`${factor}:${index}`} c="red">
                        Guardrail: {factor}
                      </Text>
                    ),
                  )}
                </Stack>
              </Card>
            </Grid.Col>
          </Grid>

          <Card withBorder>
            <Text fw={600}>Confidence</Text>
            <Stack gap={4} mt="xs">
              {(explanation.confidence_notes ?? []).map(
                (note: string, index: number) => (
                  <Text size="sm" key={`${note}:${index}`}>
                    {note}
                  </Text>
                ),
              )}
              <Text size="sm">
                Chosen propensity: {pct(decision.chosen_propensity, 1)}
              </Text>
              <Text size="sm">
                Capital exposed: {fmtPlat(decision.capital, 1)}
              </Text>
              <Text size="sm">
                Expected reward: {fmtNumber(decision.predicted_reward, 4)}
              </Text>
            </Stack>
          </Card>

          <Group justify="space-between">
            <Text size="xs" c="dimmed">
              {fmtDate(decision.created_at)} · {decision.id}
            </Text>
            <Button variant="light" onClick={() => onReplay(decision)}>
              Counterfactual replay
            </Button>
          </Group>
        </Stack>
      )}
    </Modal>
  );
}

export default function AlgoFrameHubPage() {
  const navigate = useNavigate();
  const { data: summary, refetch: refetchSummary } = api.dashboard.summary();

  const [inspector, setInspector] = useState<Inspector | null>(null);
  const [health, setHealth] = useState<ProductHealth | null>(null);
  const [decisions, setDecisions] = useState<any[]>([]);
  const [backups, setBackups] = useState<ProductBackup[]>([]);
  const [profiles, setProfiles] = useState<ProductProfile[]>([]);
  const [selectedDecision, setSelectedDecision] = useState<any | null>(null);
  const [replay, setReplay] = useState<any | null>(null);

  const [search, setSearch] = useState("");
  const [sideFilter, setSideFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [profileName, setProfileName] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [reliability, setReliability] = useState<ReliabilityStatus | null>(null);
  const [fakeMarketSuite, setFakeMarketSuite] = useState<FakeMarketSuite | null>(null);
  const [releaseGate, setReleaseGate] = useState<any | null>(null);
  const [manualFreezeReason, setManualFreezeReason] = useState("manual operator safety freeze");

  const [notificationsEnabled, setNotificationsEnabled] =
    useLocalStorage<boolean>({
      key: "algoframe-notifications-enabled",
      defaultValue: true,
    });

  const refresh = async () => {
    try {
      const [
        nextInspector,
        nextHealth,
        nextDecisions,
        nextBackups,
        nextProfiles,
        nextReliability,
      ] = await Promise.all([
        invoke<Inspector>("learning_get_inspector"),
        invoke<ProductHealth>("algoframe_product_health"),
        invoke<any[]>("learning_get_decisions", { limit: 1000 }),
        invoke<ProductBackup[]>("algoframe_product_list_backups"),
        invoke<ProductProfile[]>("algoframe_product_list_profiles"),
        invoke<ReliabilityStatus>("reliability_get_status"),
      ]);

      setInspector(nextInspector);
      setHealth(nextHealth);
      setDecisions(nextDecisions);
      setBackups(nextBackups);
      setProfiles(nextProfiles);
      setReliability(nextReliability);
      setMessage("");
      refetchSummary();
    } catch (error) {
      setMessage(String(error));
    }
  };

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 10_000);
    return () => window.clearInterval(timer);
  }, []);

  const completedDecisions = useMemo(
    () =>
      decisions
        .filter(
          (decision) =>
            decision.actual_profit != null &&
            (decision.status === "completed" ||
              decision.status === "paper_completed"),
        )
        .sort(
          (a, b) =>
            new Date(a.created_at).getTime() -
            new Date(b.created_at).getTime(),
        ),
    [decisions],
  );

  const profitChart = useMemo(() => {
    const rows = completedDecisions.slice(-120);
    let cumulative = 0;

    return {
      labels: rows.map((row) =>
        new Date(row.created_at).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
        }),
      ),
      datasets: [
        {
          label: "Cumulative realized profit",
          data: rows.map((row) => {
            cumulative += Number(row.actual_profit ?? 0) * Number(row.quantity ?? 1);
            return cumulative;
          }),
          tension: 0.25,
          fill: true,
        },
      ],
    };
  }, [completedDecisions]);

  const predictionChart = useMemo(() => {
    const rows = completedDecisions.slice(-80);

    return {
      labels: rows.map((row) => row.item_name),
      datasets: [
        {
          label: "Predicted",
          data: rows.map((row) => Number(row.predicted_profit ?? 0)),
          tension: 0.25,
        },
        {
          label: "Actual",
          data: rows.map((row) => Number(row.actual_profit ?? 0)),
          tension: 0.25,
        },
      ],
    };
  }, [completedDecisions]);

  const categoryExposureChart = useMemo(() => {
    const exposure = inspector?.portfolio.category_exposure ?? {};
    const entries = Object.entries(exposure)
      .filter(([, value]) => Number(value) > 0)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .slice(0, 12);

    return {
      labels: entries.map(([key]) => pretty(key)),
      datasets: [
        {
          label: "Capital exposure",
          data: entries.map(([, value]) => Number(value)),
        },
      ],
    };
  }, [inspector]);

  const opportunities = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();

    return decisions
      .slice()
      .sort(
        (a, b) =>
          new Date(b.updated_at ?? b.created_at).getTime() -
          new Date(a.updated_at ?? a.created_at).getTime(),
      )
      .filter((decision) => {
        if (
          normalizedSearch &&
          !String(decision.item_name ?? "")
            .toLowerCase()
            .includes(normalizedSearch)
        ) {
          return false;
        }

        if (sideFilter !== "all" && decision.side !== sideFilter) return false;
        if (statusFilter !== "all" && decision.status !== statusFilter)
          return false;

        return true;
      })
      .slice(0, 300);
  }, [decisions, search, sideFilter, statusFilter]);

  const activity = useMemo(() => {
    const rows: Array<any> = [];

    for (const decision of decisions.slice(-250)) {
      rows.push({
        id: `decision:${decision.id}`,
        type: "decision",
        time: decision.updated_at ?? decision.created_at,
        title: decision.item_name,
        subtitle: `${pretty(decision.side)} · ${pretty(decision.chosen_action)} · ${fmtPlat(decision.price)} × ${decision.quantity}`,
        status: decision.status,
        decision,
      });
    }

    for (const alert of inspector?.alerts ?? []) {
      rows.push({
        id: `alert:${alert.id}`,
        type: "alert",
        time: alert.created_at,
        title: alert.code,
        subtitle: alert.message,
        status: alert.severity,
        alert,
      });
    }

    for (const transaction of (summary?.resent_transactions ?? []) as any[]) {
      rows.push({
        id: `tx:${transaction.id ?? Math.random()}`,
        type: "transaction",
        time: transaction.updated_at ?? transaction.created_at,
        title:
          transaction.item_name ??
          transaction.properties?.item_name ??
          "Transaction",
        subtitle: `${pretty(transaction.transaction_type ?? "trade")} · ${fmtPlat(transaction.price)} × ${transaction.quantity ?? 1}`,
        status: transaction.transaction_type ?? "trade",
        transaction,
      });
    }

    return rows
      .filter((row) => row.time)
      .sort(
        (a, b) =>
          new Date(b.time).getTime() - new Date(a.time).getTime(),
      )
      .slice(0, 300);
  }, [decisions, inspector, summary]);

  const bestOpportunities = useMemo(
    () =>
      decisions
        .filter(
          (decision) =>
            decision.status === "open" ||
            decision.status === "paper_open" ||
            decision.status === "partial",
        )
        .sort(
          (a, b) =>
            Number(b.predicted_reward ?? 0) -
            Number(a.predicted_reward ?? 0),
        )
        .slice(0, 8),
    [decisions],
  );

  const thought = useMemo(() => {
    if (!inspector || !health) return "Collecting enough information to summarize the market.";

    if (health.degraded) {
      return `AlgoFrame is operating defensively: ${health.degraded_reasons[0] ?? "health checks are degraded"}.`;
    }

    if (inspector.mode === "paper") {
      return `Paper mode is active. AlgoFrame is learning from ${inspector.snapshots_recorded.toLocaleString()} market snapshots without changing live WFM orders.`;
    }

    if (bestOpportunities.length > 0) {
      const best = bestOpportunities[0];
      return `${best.item_name} currently has the strongest recorded opportunity: ${fmtPlat(best.predicted_profit, 1)} expected profit with ${pct(best.predicted_fill?.fill_1h)} estimated 1h fill probability.`;
    }

    return `Model health is ${pct(inspector.health.score)}. No currently open decision dominates the portfolio.`;
  }, [inspector, health, bestOpportunities]);

  const realizedProfit = useMemo(
    () =>
      completedDecisions.reduce(
        (sum, decision) =>
          sum +
          Number(decision.actual_profit ?? 0) *
            Number(decision.quantity ?? 1),
        0,
      ),
    [completedDecisions],
  );

  const openExpectedProfit = useMemo(
    () =>
      decisions
        .filter(
          (decision) =>
            decision.status === "open" ||
            decision.status === "paper_open" ||
            decision.status === "partial",
        )
        .reduce(
          (sum, decision) =>
            sum +
            Number(decision.predicted_profit ?? 0) *
              Number(decision.quantity ?? 1),
          0,
        ),
    [decisions],
  );

  const setMode = async (mode: ProductHealth["mode"]) => {
    setBusy(true);
    try {
      await invoke("learning_set_mode", { mode });
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const createBackup = async () => {
    setBusy(true);
    try {
      const backup = await invoke<ProductBackup>(
        "algoframe_product_backup_database",
      );
      setMessage(`Database backup created: ${backup.name}`);
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const vacuumDatabase = async () => {
    if (
      !window.confirm(
        "Compact the AlgoFrame SQLite database now? The app may pause briefly.",
      )
    ) {
      return;
    }

    setBusy(true);
    try {
      await invoke("algoframe_product_vacuum_database");
      setMessage("Database compacted successfully.");
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const saveProfile = async () => {
    if (!profileName.trim() || !inspector) return;

    setBusy(true);
    try {
      await invoke("algoframe_product_save_profile", {
        name: profileName.trim(),
        config: inspector.config,
      });
      setProfileName("");
      setMessage("Profile saved.");
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const applyProfile = async (name: string) => {
    setBusy(true);
    try {
      await invoke("algoframe_product_apply_profile", { name });
      setMessage(`Applied profile: ${name}`);
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const deleteProfile = async (name: string) => {
    if (!window.confirm(`Delete profile "${name}"?`)) return;

    setBusy(true);
    try {
      await invoke("algoframe_product_delete_profile", { name });
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const runReplay = async (decision: any) => {
    setBusy(true);
    try {
      const result = await invoke<any>("learning_counterfactual_replay", {
        decisionId: decision.id,
      });
      setReplay(result);
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const updateReliabilityConfig = async (
    patch: Partial<ReliabilityConfig>,
  ) => {
    if (!reliability) return;

    setBusy(true);
    try {
      const config = {
        ...reliability.config,
        ...patch,
      };

      await invoke<ReliabilityConfig>("reliability_update_config", {
        config,
      });
      setMessage("Reliability configuration updated.");
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const tripCircuit = async () => {
    setBusy(true);
    try {
      await invoke("reliability_trip_circuit", {
        reason: manualFreezeReason.trim() || "manual operator safety freeze",
      });
      setMessage("Live automation circuit breaker tripped.");
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const clearCircuit = async () => {
    setBusy(true);
    try {
      await invoke("reliability_clear_circuit");
      setMessage("Manual circuit breaker cleared. Automatic safety conditions still apply.");
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const runFakeMarketSuite = async () => {
    setBusy(true);
    try {
      const result = await invoke<FakeMarketSuite>(
        "reliability_run_fake_market_suite",
      );
      setFakeMarketSuite(result);
      setMessage(
        result.passed
          ? `Fake-market E2E suite passed ${result.passed_count}/${result.total_count}.`
          : `Fake-market E2E suite failed ${result.total_count - result.passed_count} scenario(s).`,
      );
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const resolveUnknownExecution = async (
    executionId: string,
    resolution: "applied" | "retry" | "cancelled",
  ) => {
    const labels = {
      applied: "mark this execution as already applied",
      retry: "clear this execution so a future retry is allowed",
      cancelled: "mark this execution as cancelled/not applied",
    };

    if (
      !window.confirm(
        `Confirm: ${labels[resolution]}? Only use this after checking the current WFM order state.`,
      )
    ) {
      return;
    }

    setBusy(true);
    try {
      await invoke("reliability_resolve_execution", {
        executionId,
        resolution,
      });
      setMessage(`Unknown execution resolved as ${resolution}.`);
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const runReleaseGate = async () => {
    setBusy(true);
    try {
      const result = await invoke<any>("reliability_run_release_gate", {
        candidatePolicy: null,
      });
      setReleaseGate(result);
      setMessage(
        result.passed
          ? "Replay-based release gate passed."
          : `Release gate blocked promotion: ${(result.reasons ?? []).join("; ")}`,
      );
      await refresh();
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusy(false);
    }
  };

  const modeColor =
    inspector?.mode === "paper"
      ? "yellow"
      : health?.degraded
        ? "red"
        : inspector?.health.fallback_active
          ? "orange"
          : "green";

  return (
    <Container fluid p="md">
      <Stack gap="md">
        <Group justify="space-between" align="flex-start">
          <div>
            <Group gap="sm">
              <Title order={2}>AlgoFrame</Title>
              <Badge color={modeColor} size="lg">
                {inspector?.mode === "paper"
                  ? "PAPER MODE"
                  : pretty(inspector?.mode ?? "loading")}
              </Badge>
              {health?.degraded && <Badge color="red">DEGRADED</Badge>}
              {reliability?.breaker.tripped && (
                <Badge color="red">EXECUTION FROZEN</Badge>
              )}
            </Group>
            <Text c="dimmed">
              Trading command center, audit trail, portfolio and model health.
            </Text>
          </div>

          <Group>
            <Button variant="light" onClick={() => navigate("/learning")}>
              Learning Lab
            </Button>
            <Button
              variant="light"
              onClick={() => navigate("/legacy-dashboard")}
            >
              Legacy Dashboard
            </Button>
            <Button onClick={refresh} loading={busy}>
              Refresh
            </Button>
          </Group>
        </Group>

        {message && (
          <Alert
            color={
              message.toLowerCase().includes("error") ||
              message.toLowerCase().includes("failed")
                ? "red"
                : "blue"
            }
            title="AlgoFrame"
            withCloseButton
            onClose={() => setMessage("")}
          >
            {message}
          </Alert>
        )}

        {health?.degraded && (
          <Alert color="red" title="AlgoFrame is in a degraded state">
            <Stack gap={2}>
              {health.degraded_reasons.map((reason) => (
                <Text key={reason} size="sm">
                  {reason}
                </Text>
              ))}
            </Stack>
          </Alert>
        )}

        <SimpleGrid cols={{ base: 2, md: 4, xl: 8 }}>
          <MetricCard
            label="Today"
            value={fmtPlat(summary?.today?.summary?.total_profit)}
            detail="Realized transaction profit"
          />
          <MetricCard
            label="All-time transactions"
            value={summary?.total?.total_transactions ?? 0}
            detail={`${summary?.total?.sale_count ?? 0} sales`}
          />
          <MetricCard
            label="Model health"
            value={pct(inspector?.health.score)}
            progress={(inspector?.health.score ?? 0) * 100}
          />
          <MetricCard
            label="Allocated capital"
            value={fmtPlat(inspector?.portfolio.allocated_capital)}
            detail={`Reserve ${fmtPlat(inspector?.portfolio.reserved_capital)}`}
          />
          <MetricCard
            label="Realized learner P&L"
            value={fmtPlat(realizedProfit, 1)}
            detail={`${completedDecisions.length} completed decisions`}
          />
          <MetricCard
            label="Open expected P&L"
            value={fmtPlat(openExpectedProfit, 1)}
            detail={`${inspector?.decisions_open ?? 0} open`}
          />
          <MetricCard
            label="Profit/hour"
            value={fmtPlat(inspector?.metrics.profit_per_hour, 2)}
            detail="Learning estimate"
          />
          <MetricCard
            label="Snapshots"
            value={(health?.snapshot_count ?? 0).toLocaleString()}
            detail={
              health?.snapshot_age_minutes != null
                ? `${health.snapshot_age_minutes.toFixed(0)}m since latest`
                : "Waiting for market data"
            }
          />
        </SimpleGrid>

        <Card withBorder>
          <Group justify="space-between" align="flex-start">
            <div>
              <Text size="xs" c="dimmed">
                ALGFRAME THINKS
              </Text>
              <Text fw={600}>{thought}</Text>
            </div>

            <Group>
              {(["paper", "conservative", "balanced", "growth", "liquid"] as const).map(
                (mode) => (
                  <Button
                    key={mode}
                    size="compact-sm"
                    variant={inspector?.mode === mode ? "filled" : "light"}
                    color={mode === "paper" ? "yellow" : undefined}
                    onClick={() => setMode(mode)}
                    loading={busy}
                  >
                    {pretty(mode)}
                  </Button>
                ),
              )}
            </Group>
          </Group>
        </Card>

        <Tabs defaultValue="overview">
          <Tabs.List>
            <Tabs.Tab value="overview">Overview</Tabs.Tab>
            <Tabs.Tab value="activity">Activity</Tabs.Tab>
            <Tabs.Tab value="opportunities">Opportunities</Tabs.Tab>
            <Tabs.Tab value="portfolio">Portfolio</Tabs.Tab>
            <Tabs.Tab value="reliability">Reliability</Tabs.Tab>
            <Tabs.Tab value="health">Health & Settings</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="overview" pt="md">
            <Stack gap="md">
              <Grid>
                <Grid.Col span={{ base: 12, lg: 8 }}>
                  <Card withBorder>
                    <Group justify="space-between">
                      <Title order={4}>Realized learner profit</Title>
                      <Badge variant="light">
                        {completedDecisions.length} completed
                      </Badge>
                    </Group>
                    {completedDecisions.length > 1 ? (
                      <Line
                        data={profitChart}
                        options={{
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: { legend: { display: true } },
                        }}
                        height={260}
                      />
                    ) : (
                      <Text c="dimmed" mt="md">
                        Complete more paper/live decisions to build this chart.
                      </Text>
                    )}
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, lg: 4 }}>
                  <Card withBorder h="100%">
                    <Title order={4}>Portfolio allocation</Title>
                    {Object.keys(
                      inspector?.portfolio.category_exposure ?? {},
                    ).length > 0 ? (
                      <Doughnut
                        data={categoryExposureChart}
                        options={{
                          responsive: true,
                          plugins: { legend: { position: "bottom" } },
                        }}
                      />
                    ) : (
                      <Text c="dimmed" mt="md">
                        No category exposure recorded yet.
                      </Text>
                    )}
                  </Card>
                </Grid.Col>
              </Grid>

              <Grid>
                <Grid.Col span={{ base: 12, lg: 7 }}>
                  <Card withBorder>
                    <Title order={4}>Best current opportunities</Title>
                    <ScrollArea h={320} mt="sm">
                      <Table striped highlightOnHover>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th>Item</Table.Th>
                            <Table.Th>Side</Table.Th>
                            <Table.Th>Price</Table.Th>
                            <Table.Th>Profit</Table.Th>
                            <Table.Th>1h fill</Table.Th>
                            <Table.Th>Reward</Table.Th>
                            <Table.Th />
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                          {bestOpportunities.map((decision) => (
                            <Table.Tr key={decision.id}>
                              <Table.Td>{decision.item_name}</Table.Td>
                              <Table.Td>{pretty(decision.side)}</Table.Td>
                              <Table.Td>{fmtPlat(decision.price)}</Table.Td>
                              <Table.Td>
                                {fmtPlat(decision.predicted_profit, 1)}
                              </Table.Td>
                              <Table.Td>
                                {pct(decision.predicted_fill?.fill_1h)}
                              </Table.Td>
                              <Table.Td>
                                {fmtNumber(decision.predicted_reward, 4)}
                              </Table.Td>
                              <Table.Td>
                                <Button
                                  size="compact-xs"
                                  variant="light"
                                  onClick={() => setSelectedDecision(decision)}
                                >
                                  Why?
                                </Button>
                              </Table.Td>
                            </Table.Tr>
                          ))}
                          {bestOpportunities.length === 0 && (
                            <Table.Tr>
                              <Table.Td colSpan={7}>
                                <Text c="dimmed">No open opportunities yet.</Text>
                              </Table.Td>
                            </Table.Tr>
                          )}
                        </Table.Tbody>
                      </Table>
                    </ScrollArea>
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, lg: 5 }}>
                  <Card withBorder>
                    <Title order={4}>Alerts requiring attention</Title>
                    <Stack mt="sm" gap="xs">
                      {(inspector?.alerts ?? []).slice(0, 8).map((alert) => (
                        <Alert
                          key={alert.id}
                          color={severityColor(alert.severity)}
                          title={alert.code}
                        >
                          {alert.message}
                        </Alert>
                      ))}
                      {(inspector?.alerts ?? []).length === 0 && (
                        <Text c="dimmed">No learning alerts.</Text>
                      )}
                    </Stack>
                  </Card>
                </Grid.Col>
              </Grid>

              <Card withBorder>
                <Title order={4}>Predicted vs actual profit</Title>
                {completedDecisions.length > 2 ? (
                  <Line
                    data={predictionChart}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        x: {
                          ticks: { display: false },
                        },
                      },
                    }}
                    height={240}
                  />
                ) : (
                  <Text c="dimmed" mt="md">
                    Not enough completed decisions yet.
                  </Text>
                )}
              </Card>
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value="activity" pt="md">
            <Card withBorder>
              <Group justify="space-between">
                <div>
                  <Title order={4}>Activity & audit timeline</Title>
                  <Text size="sm" c="dimmed">
                    Decisions, model alerts and completed transactions in one place.
                  </Text>
                </div>
                <Badge variant="light">{activity.length} recent events</Badge>
              </Group>

              <ScrollArea h={620} mt="md">
                <Stack gap="xs">
                  {activity.map((row) => (
                    <Card
                      key={row.id}
                      withBorder
                      padding="sm"
                      onClick={() => {
                        if (row.decision) setSelectedDecision(row.decision);
                      }}
                      style={{
                        cursor: row.decision ? "pointer" : "default",
                      }}
                    >
                      <Group justify="space-between" align="flex-start">
                        <div>
                          <Group gap="xs">
                            <Badge
                              variant="light"
                              color={
                                row.type === "alert"
                                  ? severityColor(row.status)
                                  : row.type === "transaction"
                                    ? "green"
                                    : statusColor(String(row.status))
                              }
                            >
                              {pretty(row.type)}
                            </Badge>
                            <Text fw={600}>{row.title}</Text>
                          </Group>
                          <Text size="sm" c="dimmed" mt={4}>
                            {row.subtitle}
                          </Text>
                        </div>
                        <Text size="xs" c="dimmed">
                          {fmtDate(row.time)}
                        </Text>
                      </Group>
                    </Card>
                  ))}
                </Stack>
              </ScrollArea>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="opportunities" pt="md">
            <Card withBorder>
              <Group justify="space-between" align="flex-end">
                <div>
                  <Title order={4}>Opportunity explorer</Title>
                  <Text size="sm" c="dimmed">
                    Includes opportunities AlgoFrame rejected, so the policy is inspectable.
                  </Text>
                </div>

                <Group>
                  <TextInput
                    placeholder="Search item"
                    value={search}
                    onChange={(event) => setSearch(event.currentTarget.value)}
                  />
                  <Select
                    value={sideFilter}
                    onChange={(value) => setSideFilter(value ?? "all")}
                    data={[
                      { value: "all", label: "All sides" },
                      { value: "buy", label: "Buy" },
                      { value: "sell", label: "Sell" },
                    ]}
                  />
                  <Select
                    value={statusFilter}
                    onChange={(value) => setStatusFilter(value ?? "all")}
                    data={[
                      { value: "all", label: "All statuses" },
                      { value: "open", label: "Open" },
                      { value: "paper_open", label: "Paper open" },
                      { value: "completed", label: "Completed" },
                      { value: "paper_completed", label: "Paper completed" },
                      { value: "rejected", label: "Rejected" },
                      { value: "expired", label: "Expired" },
                    ]}
                  />
                </Group>
              </Group>

              <ScrollArea h={650} mt="md">
                <Table striped highlightOnHover stickyHeader>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Item</Table.Th>
                      <Table.Th>Category</Table.Th>
                      <Table.Th>Side</Table.Th>
                      <Table.Th>Status</Table.Th>
                      <Table.Th>Action</Table.Th>
                      <Table.Th>Price × Qty</Table.Th>
                      <Table.Th>Expected</Table.Th>
                      <Table.Th>1h fill</Table.Th>
                      <Table.Th>Risk</Table.Th>
                      <Table.Th />
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {opportunities.map((decision) => (
                      <Table.Tr key={decision.id}>
                        <Table.Td>{decision.item_name}</Table.Td>
                        <Table.Td>{pretty(decision.category)}</Table.Td>
                        <Table.Td>{pretty(decision.side)}</Table.Td>
                        <Table.Td>
                          <Badge
                            size="sm"
                            color={statusColor(String(decision.status))}
                          >
                            {pretty(decision.status)}
                          </Badge>
                        </Table.Td>
                        <Table.Td>{pretty(decision.chosen_action)}</Table.Td>
                        <Table.Td>
                          {fmtPlat(decision.price)} × {decision.quantity}
                        </Table.Td>
                        <Table.Td>
                          {fmtPlat(decision.predicted_profit, 1)}
                        </Table.Td>
                        <Table.Td>
                          {pct(decision.predicted_fill?.fill_1h)}
                        </Table.Td>
                        <Table.Td>
                          <Tooltip
                            label={`Volatility ${fmtNumber(decision.features?.volatility)} · anomaly ${pct(decision.features?.anomaly?.score)}`}
                          >
                            <Badge
                              variant="light"
                              color={
                                Number(decision.features?.anomaly?.score ?? 0) > 0.5
                                  ? "red"
                                  : Number(decision.features?.volatility ?? 0) > 0.5
                                    ? "yellow"
                                    : "green"
                              }
                            >
                              {Number(decision.features?.anomaly?.score ?? 0) > 0.5
                                ? "High"
                                : Number(decision.features?.volatility ?? 0) > 0.5
                                  ? "Medium"
                                  : "Low"}
                            </Badge>
                          </Tooltip>
                        </Table.Td>
                        <Table.Td>
                          <Button
                            size="compact-xs"
                            variant="light"
                            onClick={() => setSelectedDecision(decision)}
                          >
                            Why?
                          </Button>
                        </Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="portfolio" pt="md">
            <Stack gap="md">
              <SimpleGrid cols={{ base: 2, md: 5 }}>
                <MetricCard
                  label="Capital budget"
                  value={fmtPlat(inspector?.portfolio.total_capital_budget)}
                />
                <MetricCard
                  label="Allocated"
                  value={fmtPlat(inspector?.portfolio.allocated_capital)}
                />
                <MetricCard
                  label="Reserved"
                  value={fmtPlat(inspector?.portfolio.reserved_capital)}
                />
                <MetricCard
                  label="High volatility"
                  value={fmtPlat(inspector?.portfolio.high_volatility_capital)}
                />
                <MetricCard
                  label="Experimental"
                  value={fmtPlat(inspector?.portfolio.experimental_capital)}
                />
              </SimpleGrid>

              <Grid>
                <Grid.Col span={{ base: 12, md: 5 }}>
                  <Card withBorder>
                    <Title order={4}>Category allocation</Title>
                    {Object.keys(
                      inspector?.portfolio.category_exposure ?? {},
                    ).length > 0 ? (
                      <Doughnut
                        data={categoryExposureChart}
                        options={{
                          responsive: true,
                          plugins: { legend: { position: "bottom" } },
                        }}
                      />
                    ) : (
                      <Text c="dimmed" mt="md">
                        No allocated capital yet.
                      </Text>
                    )}
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, md: 7 }}>
                  <Card withBorder>
                    <Title order={4}>Learned item performance</Title>
                    <ScrollArea h={360} mt="sm">
                      <Table striped>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th>Item</Table.Th>
                            <Table.Th>Trades</Table.Th>
                            <Table.Th>Confidence</Table.Th>
                            <Table.Th>Avg profit</Table.Th>
                            <Table.Th>Buy fill</Table.Th>
                            <Table.Th>Sell fill</Table.Th>
                            <Table.Th>Open units</Table.Th>
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                          {(inspector?.top_items ?? []).map((item: any) => (
                            <Table.Tr key={String(item.item_key)}>
                              <Table.Td>{String(item.item_key)}</Table.Td>
                              <Table.Td>{Number(item.trades ?? 0)}</Table.Td>
                              <Table.Td>
                                {pct(Number(item.confidence ?? 0))}
                              </Table.Td>
                              <Table.Td>
                                {fmtPlat(Number(item.avg_profit ?? 0), 1)}
                              </Table.Td>
                              <Table.Td>
                                {fmtNumber(item.avg_buy_fill_hours)}h
                              </Table.Td>
                              <Table.Td>
                                {fmtNumber(item.avg_sell_fill_hours)}h
                              </Table.Td>
                              <Table.Td>
                                {Number(item.open_units ?? 0)}
                              </Table.Td>
                            </Table.Tr>
                          ))}
                        </Table.Tbody>
                      </Table>
                    </ScrollArea>
                  </Card>
                </Grid.Col>
              </Grid>

              <Card withBorder>
                <Title order={4}>Arbitrage & conversion opportunities</Title>
                <Table mt="sm" striped>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Type</Table.Th>
                      <Table.Th>Family</Table.Th>
                      <Table.Th>Cost</Table.Th>
                      <Table.Th>Revenue</Table.Th>
                      <Table.Th>Profit</Table.Th>
                      <Table.Th>Confidence</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {(inspector?.arbitrage_opportunities ?? [])
                      .slice(0, 30)
                      .map((opportunity: any) => (
                        <Table.Tr key={opportunity.id}>
                          <Table.Td>{pretty(opportunity.kind)}</Table.Td>
                          <Table.Td>{opportunity.family}</Table.Td>
                          <Table.Td>{fmtPlat(opportunity.cost, 1)}</Table.Td>
                          <Table.Td>
                            {fmtPlat(opportunity.expected_revenue, 1)}
                          </Table.Td>
                          <Table.Td>
                            {fmtPlat(opportunity.expected_profit, 1)}
                          </Table.Td>
                          <Table.Td>{pct(opportunity.confidence)}</Table.Td>
                        </Table.Tr>
                      ))}
                  </Table.Tbody>
                </Table>
              </Card>
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value="reliability" pt="md">
            <Stack gap="md">
              <Grid>
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <MetricCard
                    label="Circuit breaker"
                    value={
                      reliability?.breaker.tripped ? "TRIPPED" : "Armed / Clear"
                    }
                    detail={
                      reliability?.breaker.manual_trip
                        ? "Manual freeze is active"
                        : `${reliability?.breaker.recent_failures ?? 0} recent failures`
                    }
                    progress={
                      reliability?.breaker.tripped ? 100 : 0
                    }
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <MetricCard
                    label="Uncertain executions"
                    value={reliability?.breaker.unknown_executions ?? 0}
                    detail={`${reliability?.pending_executions ?? 0} pending reconciliation`}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <MetricCard
                    label="Lifecycle events"
                    value={reliability?.lifecycle_events ?? 0}
                    detail="Authoritative execution-state transitions"
                  />
                </Grid.Col>
              </Grid>

              {reliability?.breaker.tripped && (
                <Alert color="red" title="Live automation is frozen">
                  <Stack gap={4}>
                    {(reliability.breaker.reasons ?? []).map((reason) => (
                      <Text size="sm" key={reason}>
                        {reason}
                      </Text>
                    ))}
                  </Stack>
                </Alert>
              )}

              <Grid>
                <Grid.Col span={{ base: 12, lg: 6 }}>
                  <Card withBorder>
                    <Title order={4}>Circuit breaker controls</Title>
                    <Text size="sm" c="dimmed">
                      A prepared remote mutation is journaled before WFM is touched.
                      Unknown remote state freezes further live execution instead of
                      risking duplicate orders.
                    </Text>

                    <TextInput
                      mt="md"
                      label="Manual freeze reason"
                      value={manualFreezeReason}
                      onChange={(event) =>
                        setManualFreezeReason(event.currentTarget.value)
                      }
                    />

                    <Group mt="sm">
                      <Button color="red" onClick={tripCircuit} loading={busy}>
                        Freeze live execution
                      </Button>
                      <Button
                        variant="light"
                        onClick={clearCircuit}
                        loading={busy}
                      >
                        Clear manual freeze
                      </Button>
                    </Group>

                    <Divider my="md" />

                    <SimpleGrid cols={2}>
                      <Text size="sm">
                        Recent actions:{" "}
                        <b>{reliability?.breaker.recent_actions ?? 0}</b>
                      </Text>
                      <Text size="sm">
                        Recent failures:{" "}
                        <b>{reliability?.breaker.recent_failures ?? 0}</b>
                      </Text>
                      <Text size="sm">
                        Daily realized loss:{" "}
                        <b>{fmtPlat(reliability?.breaker.daily_realized_loss, 1)}</b>
                      </Text>
                      <Text size="sm">
                        Unknown remote state:{" "}
                        <b>{reliability?.breaker.unknown_executions ?? 0}</b>
                      </Text>
                    </SimpleGrid>
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, lg: 6 }}>
                  <Card withBorder>
                    <Title order={4}>Deterministic fake-market E2E suite</Title>
                    <Text size="sm" c="dimmed">
                      Exercises buy→fill→sell, partial fills, rejection, crash-after-dispatch,
                      and paper-only execution through the reliability state machine.
                    </Text>

                    <Group mt="md">
                      <Button onClick={runFakeMarketSuite} loading={busy}>
                        Run fake-market suite
                      </Button>
                      {fakeMarketSuite && (
                        <Badge color={fakeMarketSuite.passed ? "green" : "red"}>
                          {fakeMarketSuite.passed_count}/{fakeMarketSuite.total_count} passed
                        </Badge>
                      )}
                    </Group>

                    {fakeMarketSuite && (
                      <Stack mt="md" gap="xs">
                        {fakeMarketSuite.scenarios.map((scenario) => (
                          <Card withBorder padding="xs" key={scenario.name}>
                            <Group justify="space-between">
                              <Text size="sm">{scenario.name}</Text>
                              <Badge color={scenario.passed ? "green" : "red"}>
                                {scenario.passed ? "PASS" : "FAIL"}
                              </Badge>
                            </Group>
                            <Text size="xs" c="dimmed">
                              Final state: {pretty(scenario.final_state)}
                            </Text>
                          </Card>
                        ))}
                      </Stack>
                    )}
                  </Card>
                </Grid.Col>
              </Grid>

              <Card withBorder>
                <Group justify="space-between" align="flex-start">
                  <div>
                    <Title order={4}>Replay-based release gate</Title>
                    <Text size="sm" c="dimmed">
                      Candidate policies must beat the champion on offline reward while
                      staying inside failure, drawdown, walk-forward, leakage and stress-test limits.
                    </Text>
                  </div>
                  <Button onClick={runReleaseGate} loading={busy}>
                    Evaluate challenger
                  </Button>
                </Group>

                {(releaseGate ?? reliability?.last_release_gate) && (
                  <Grid mt="md">
                    <Grid.Col span={{ base: 12, md: 3 }}>
                      <MetricCard
                        label="Gate result"
                        value={
                          (releaseGate ?? reliability?.last_release_gate)?.passed
                            ? "PASS"
                            : "BLOCK"
                        }
                      />
                    </Grid.Col>
                    <Grid.Col span={{ base: 12, md: 3 }}>
                      <MetricCard
                        label="Reward uplift"
                        value={pct(
                          (releaseGate ?? reliability?.last_release_gate)
                            ?.reward_uplift_pct,
                          1,
                        )}
                      />
                    </Grid.Col>
                    <Grid.Col span={{ base: 12, md: 3 }}>
                      <MetricCard
                        label="Failure delta"
                        value={pct(
                          (releaseGate ?? reliability?.last_release_gate)
                            ?.failure_delta,
                          1,
                        )}
                      />
                    </Grid.Col>
                    <Grid.Col span={{ base: 12, md: 3 }}>
                      <MetricCard
                        label="Walk-forward stability"
                        value={pct(
                          (releaseGate ?? reliability?.last_release_gate)
                            ?.walk_forward_stability,
                          0,
                        )}
                      />
                    </Grid.Col>
                  </Grid>
                )}

                {(releaseGate ?? reliability?.last_release_gate)?.reasons?.length > 0 && (
                  <Alert mt="md" color="yellow" title="Promotion blockers">
                    {(releaseGate ?? reliability?.last_release_gate).reasons.join("; ")}
                  </Alert>
                )}
              </Card>

              <Card withBorder>
                <Group justify="space-between" align="flex-start">
                  <div>
                    <Title order={4}>Safety thresholds</Title>
                    <Text size="sm" c="dimmed">
                      These limits are hard execution controls. They do not change
                      the learner's predictions; they decide when live execution
                      must stop.
                    </Text>
                  </div>
                  <Switch
                    label="Circuit breaker enabled"
                    checked={reliability?.config.circuit_breaker_enabled ?? true}
                    onChange={(event) =>
                      updateReliabilityConfig({
                        circuit_breaker_enabled: event.currentTarget.checked,
                      })
                    }
                  />
                </Group>

                <SimpleGrid cols={{ base: 2, md: 4 }} mt="md">
                  <NumberInput
                    label="Minimum model health %"
                    value={(reliability?.config.minimum_model_health ?? 0.45) * 100}
                    min={0}
                    max={100}
                    decimalScale={0}
                    onChange={(value) =>
                      updateReliabilityConfig({
                        minimum_model_health: Number(value) / 100,
                      })
                    }
                  />
                  <NumberInput
                    label="Max anomaly %"
                    value={(reliability?.config.maximum_anomaly_score ?? 0.85) * 100}
                    min={0}
                    max={100}
                    decimalScale={0}
                    onChange={(value) =>
                      updateReliabilityConfig({
                        maximum_anomaly_score: Number(value) / 100,
                      })
                    }
                  />
                  <NumberInput
                    label="Failures before freeze"
                    value={reliability?.config.maximum_execution_failures ?? 3}
                    min={1}
                    max={100}
                    onChange={(value) =>
                      updateReliabilityConfig({
                        maximum_execution_failures: Number(value),
                      })
                    }
                  />
                  <NumberInput
                    label="Max actions / minute"
                    value={reliability?.config.maximum_actions_per_minute ?? 12}
                    min={1}
                    max={120}
                    onChange={(value) =>
                      updateReliabilityConfig({
                        maximum_actions_per_minute: Number(value),
                      })
                    }
                  />
                  <NumberInput
                    label="Daily realized loss limit"
                    value={reliability?.config.maximum_daily_realized_loss ?? 250}
                    min={0}
                    suffix="p"
                    onChange={(value) =>
                      updateReliabilityConfig({
                        maximum_daily_realized_loss: Number(value),
                      })
                    }
                  />
                  <NumberInput
                    label="Prepared timeout"
                    value={reliability?.config.prepared_timeout_minutes ?? 5}
                    min={1}
                    max={120}
                    suffix=" min"
                    onChange={(value) =>
                      updateReliabilityConfig({
                        prepared_timeout_minutes: Number(value),
                      })
                    }
                  />
                  <NumberInput
                    label="Release gate samples"
                    value={reliability?.config.release_gate_minimum_samples ?? 50}
                    min={10}
                    max={100000}
                    onChange={(value) =>
                      updateReliabilityConfig({
                        release_gate_minimum_samples: Number(value),
                      })
                    }
                  />
                  <NumberInput
                    label="Required reward uplift %"
                    value={
                      (reliability?.config.release_gate_minimum_reward_uplift_pct ??
                        0.03) * 100
                    }
                    min={-50}
                    max={200}
                    decimalScale={1}
                    onChange={(value) =>
                      updateReliabilityConfig({
                        release_gate_minimum_reward_uplift_pct:
                          Number(value) / 100,
                      })
                    }
                  />
                </SimpleGrid>
              </Card>

              <Card withBorder>
                <Title order={4}>Crash-safe execution journal</Title>
                <Text size="sm" c="dimmed">
                  `Prepared` is persisted before a remote mutation. On restart,
                  unresolved executions are reconciled against the WFM order cache.
                </Text>

                <ScrollArea h={420} mt="sm">
                  <Table striped highlightOnHover stickyHeader>
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Updated</Table.Th>
                        <Table.Th>Item key</Table.Th>
                        <Table.Th>Side</Table.Th>
                        <Table.Th>Operation</Table.Th>
                        <Table.Th>Target</Table.Th>
                        <Table.Th>State</Table.Th>
                        <Table.Th>Attempts</Table.Th>
                        <Table.Th>Error</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {(reliability?.recent_executions ?? []).map((entry) => (
                        <Table.Tr key={entry.id}>
                          <Table.Td>{fmtDate(entry.updated_at)}</Table.Td>
                          <Table.Td>{entry.item_key}</Table.Td>
                          <Table.Td>{pretty(entry.side)}</Table.Td>
                          <Table.Td>{pretty(entry.operation)}</Table.Td>
                          <Table.Td>
                            {fmtPlat(entry.target_price)} × {entry.target_quantity}
                          </Table.Td>
                          <Table.Td>
                            <Badge
                              color={
                                entry.state === "applied" ||
                                entry.state === "reconciled"
                                  ? "green"
                                  : entry.state === "unknown" ||
                                      entry.state === "failed"
                                    ? "red"
                                    : "blue"
                              }
                            >
                              {pretty(entry.state)}
                            </Badge>
                          </Table.Td>
                          <Table.Td>{entry.attempt_count}</Table.Td>
                          <Table.Td>
                            <Stack gap={4}>
                              <Text size="xs" c={entry.error ? "red" : "dimmed"}>
                                {entry.error || "—"}
                              </Text>
                              {entry.state === "unknown" && (
                                <Group gap={4}>
                                  <Button
                                    size="compact-xs"
                                    variant="light"
                                    color="green"
                                    onClick={() =>
                                      resolveUnknownExecution(entry.id, "applied")
                                    }
                                  >
                                    Verified applied
                                  </Button>
                                  <Button
                                    size="compact-xs"
                                    variant="light"
                                    color="yellow"
                                    onClick={() =>
                                      resolveUnknownExecution(entry.id, "retry")
                                    }
                                  >
                                    Safe to retry
                                  </Button>
                                  <Button
                                    size="compact-xs"
                                    variant="light"
                                    color="red"
                                    onClick={() =>
                                      resolveUnknownExecution(entry.id, "cancelled")
                                    }
                                  >
                                    Not applied
                                  </Button>
                                </Group>
                              )}
                            </Stack>
                          </Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                </ScrollArea>
              </Card>
            </Stack>
          </Tabs.Panel>

          <Tabs.Panel value="health" pt="md">
            <Stack gap="md">
              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Card withBorder>
                    <Title order={4}>Runtime & database health</Title>
                    <Stack mt="sm" gap={6}>
                      <Group justify="space-between">
                        <Text size="sm">SQLite integrity</Text>
                        <Badge
                          color={
                            health?.database_integrity === "ok"
                              ? "green"
                              : "red"
                          }
                        >
                          {health?.database_integrity ?? "Checking"}
                        </Badge>
                      </Group>
                      <Group justify="space-between">
                        <Text size="sm">Database size</Text>
                        <Text size="sm">
                          <NumberFormatter
                            value={(health?.database_size_bytes ?? 0) / 1024 / 1024}
                            decimalScale={1}
                            suffix=" MB"
                          />
                        </Text>
                      </Group>
                      <Group justify="space-between">
                        <Text size="sm">Snapshots</Text>
                        <Text size="sm">{health?.snapshot_count ?? 0}</Text>
                      </Group>
                      <Group justify="space-between">
                        <Text size="sm">Decisions</Text>
                        <Text size="sm">{health?.decision_count ?? 0}</Text>
                      </Group>
                      <Group justify="space-between">
                        <Text size="sm">Outcomes</Text>
                        <Text size="sm">{health?.outcome_count ?? 0}</Text>
                      </Group>
                      <Group justify="space-between">
                        <Text size="sm">Latest snapshot</Text>
                        <Text size="sm">
                          {fmtDate(health?.latest_snapshot_at)}
                        </Text>
                      </Group>
                      <Group mt="sm">
                        <Button
                          variant="light"
                          onClick={createBackup}
                          loading={busy}
                        >
                          Backup database now
                        </Button>
                        <Button
                          variant="light"
                          onClick={vacuumDatabase}
                          loading={busy}
                        >
                          Compact database
                        </Button>
                      </Group>
                    </Stack>
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Card withBorder>
                    <Title order={4}>Model health</Title>
                    <Progress
                      mt="sm"
                      value={(inspector?.health.score ?? 0) * 100}
                      color={
                        inspector?.health.fallback_active ? "orange" : "green"
                      }
                    />
                    <Stack gap={6} mt="sm">
                      <Text size="sm">
                        Score: {pct(inspector?.health.score, 1)}
                      </Text>
                      <Text size="sm">
                        Prediction MAE:{" "}
                        {fmtPlat(inspector?.health.recent_prediction_mae, 2)}
                      </Text>
                      <Text size="sm">
                        Calibration error:{" "}
                        {pct(inspector?.health.calibration_error, 1)}
                      </Text>
                      <Text size="sm">
                        Failure rate:{" "}
                        {pct(inspector?.health.recent_failure_rate, 1)}
                      </Text>
                      <Text size="sm">
                        Drawdown: {pct(inspector?.health.drawdown, 1)}
                      </Text>
                      {(inspector?.health.reasons ?? []).map((reason) => (
                        <Text key={reason} size="sm" c="orange">
                          {reason}
                        </Text>
                      ))}
                    </Stack>
                  </Card>
                </Grid.Col>
              </Grid>

              <Grid>
                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Card withBorder>
                    <Title order={4}>Configuration profiles</Title>
                    <Text size="sm" c="dimmed">
                      Save a complete AlgoFrame learning/risk configuration and restore it later.
                    </Text>

                    <Group mt="sm" align="flex-end">
                      <TextInput
                        label="Profile name"
                        placeholder="Main Balanced"
                        value={profileName}
                        onChange={(event) =>
                          setProfileName(event.currentTarget.value)
                        }
                        style={{ flex: 1 }}
                      />
                      <Button
                        onClick={saveProfile}
                        disabled={!profileName.trim() || !inspector}
                        loading={busy}
                      >
                        Save current
                      </Button>
                    </Group>

                    <Divider my="md" />

                    <Stack gap="xs">
                      {profiles.map((profile) => (
                        <Card key={profile.name} withBorder padding="sm">
                          <Group justify="space-between">
                            <div>
                              <Text fw={600}>{profile.name}</Text>
                              <Text size="xs" c="dimmed">
                                {pretty(profile.config.mode)} · revision{" "}
                                {profile.config.settings_revision} ·{" "}
                                {fmtDate(profile.updated_at)}
                              </Text>
                            </div>
                            <Group>
                              <Button
                                size="compact-xs"
                                variant="light"
                                onClick={() => applyProfile(profile.name)}
                              >
                                Apply
                              </Button>
                              <Button
                                size="compact-xs"
                                variant="subtle"
                                color="red"
                                onClick={() => deleteProfile(profile.name)}
                              >
                                Delete
                              </Button>
                            </Group>
                          </Group>
                        </Card>
                      ))}
                      {profiles.length === 0 && (
                        <Text c="dimmed" size="sm">
                          No saved profiles.
                        </Text>
                      )}
                    </Stack>
                  </Card>
                </Grid.Col>

                <Grid.Col span={{ base: 12, md: 6 }}>
                  <Card withBorder>
                    <Title order={4}>Backups & notifications</Title>
                    <Switch
                      mt="sm"
                      label="Desktop notifications for important AlgoFrame alerts"
                      checked={notificationsEnabled}
                      onChange={(event) =>
                        setNotificationsEnabled(event.currentTarget.checked)
                      }
                    />

                    <Divider my="md" />

                    <Text fw={600}>Recent database backups</Text>
                    <ScrollArea h={230} mt="xs">
                      <Stack gap="xs">
                        {backups.slice(0, 20).map((backup) => (
                          <Card key={backup.path} withBorder padding="xs">
                            <Group justify="space-between">
                              <div>
                                <Text size="sm">{backup.name}</Text>
                                <Text size="xs" c="dimmed">
                                  {fmtDate(backup.created_at)}
                                </Text>
                              </div>
                              <Text size="xs" c="dimmed">
                                {(backup.size_bytes / 1024 / 1024).toFixed(1)} MB
                              </Text>
                            </Group>
                          </Card>
                        ))}
                        {backups.length === 0 && (
                          <Text c="dimmed" size="sm">
                            No product backups yet.
                          </Text>
                        )}
                      </Stack>
                    </ScrollArea>
                  </Card>
                </Grid.Col>
              </Grid>

              <Card withBorder>
                <Group justify="space-between">
                  <div>
                    <Title order={4}>Advanced controls</Title>
                    <Text size="sm" c="dimmed">
                      Full ML, risk, forgetting, replay and export controls remain in the Learning Lab.
                    </Text>
                  </div>
                  <Button onClick={() => navigate("/learning")}>
                    Open Learning Lab
                  </Button>
                </Group>
              </Card>
            </Stack>
          </Tabs.Panel>
        </Tabs>
      </Stack>

      <ExplanationModal
        decision={selectedDecision}
        onClose={() => setSelectedDecision(null)}
        onReplay={runReplay}
      />

      <Modal
        opened={Boolean(replay)}
        onClose={() => setReplay(null)}
        title="Counterfactual replay"
        size="xl"
      >
        {replay && (
          <Stack>
            <Text fw={600}>
              {replay.decision?.item_name} · {pretty(replay.decision?.side)}
            </Text>
            <Text size="sm" c="dimmed">
              The alternatives below replay the recorded decision against future snapshots from the same item.
            </Text>

            <Table striped>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Action</Table.Th>
                  <Table.Th>Price</Table.Th>
                  <Table.Th>Filled</Table.Th>
                  <Table.Th>Fill time</Table.Th>
                  <Table.Th>Reward</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {(replay.alternatives ?? []).map(
                  (alternative: any, index: number) => (
                    <Table.Tr key={`${alternative.action}:${index}`}>
                      <Table.Td>{pretty(alternative.action)}</Table.Td>
                      <Table.Td>{fmtPlat(alternative.price)}</Table.Td>
                      <Table.Td>
                        {alternative.filled ? "Yes" : "No"}
                      </Table.Td>
                      <Table.Td>
                        {alternative.fill_hours == null
                          ? "—"
                          : `${fmtNumber(alternative.fill_hours)}h`}
                      </Table.Td>
                      <Table.Td>
                        {fmtNumber(alternative.reward, 4)}
                      </Table.Td>
                    </Table.Tr>
                  ),
                )}
              </Table.Tbody>
            </Table>
          </Stack>
        )}
      </Modal>
    </Container>
  );
}
