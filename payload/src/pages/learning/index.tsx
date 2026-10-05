
import { invoke } from "@tauri-apps/api/core";
import {
  Alert,
  Badge,
  Button,
  Card,
  Code,
  Divider,
  Grid,
  Group,
  NumberInput,
  Progress,
  ScrollArea,
  Select,
  Stack,
  Switch,
  Table,
  Tabs,
  Text,
  TextInput,
  Title,
} from "@mantine/core";
import { useCallback, useEffect, useMemo, useState } from "react";

type OperatingMode = "paper" | "conservative" | "balanced" | "growth" | "liquid";

type UltimateConfig = {
  enabled: boolean;
  mode: OperatingMode;
  cash_reserve_pct: number;
  max_item_exposure_pct: number;
  max_category_exposure_pct: number;
  max_family_exposure_pct: number;
  max_high_volatility_exposure_pct: number;
  experimental_budget_pct: number;
  max_single_decision_pct: number;
  max_trade_quantity: number;
  max_daily_trade_interactions: number;
  human_minutes_per_trade: number;
  human_time_value_plat_per_hour: number;
  min_reprice_minutes: number;
  stable_reprice_minutes: number;
  reprice_min_reward_gain_pct: number;
  minimum_data_quality: number;
  minimum_model_health: number;
  max_drawdown_pct: number;
  max_inventory_pressure: number;
  recency_half_life_days: number;
  snapshot_full_days: number;
  snapshot_five_min_days: number;
  snapshot_thirty_min_days: number;
  snapshot_hourly_days: number;
  max_price_search_steps: number;
  max_quantity_search: number;
  propensity_samples: number;
  nearest_neighbor_count: number;
  champion_policy: string;
  challenger_policy: string;
  challenger_fraction: number;
  minimum_promotion_samples: number;
  promotion_margin_pct: number;
  maximum_challenger_failure_delta: number;
  automatic_tuning: boolean;
  automatic_promotion: boolean;
  automatic_rollback: boolean;
  shadow_mode: boolean;
  market_recording: boolean;
  event_intelligence: boolean;
  arbitrage_engine: boolean;
  continuous_price_optimization: boolean;
  quantity_optimization: boolean;
  survival_models: boolean;
  hierarchical_learning: boolean;
  anomaly_detection: boolean;
  regime_detection: boolean;
  opportunity_forecasting: boolean;

  distributional_predictions: boolean;
  cvar_risk_optimization: boolean;
  conformal_intervals: boolean;
  uncertainty_decomposition: boolean;
  learned_market_states: boolean;
  book_persistence_model: boolean;
  causal_evaluation: boolean;
  active_learning: boolean;
  pareto_portfolio: boolean;
  walk_forward_validation: boolean;
  stress_testing: boolean;
  monte_carlo_portfolio: boolean;
  automatic_ablation: boolean;
  leakage_detection: boolean;
  nonlinear_expert: boolean;
  mixture_of_experts: boolean;

  cvar_alpha: number;
  downside_risk_weight: number;
  epistemic_risk_weight: number;
  aleatoric_risk_weight: number;
  active_learning_budget_pct: number;

  settings_revision: number;
};

type Inspector = {
  model: string;
  feature_schema_version: number;
  reward_version: number;
  policy_version: number;
  updated_at?: string | null;
  mode: OperatingMode;
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
    drift_score: number;
    drawdown: number;
  };
  transactions_learned: number;
  snapshots_recorded: number;
  decisions_total: number;
  decisions_completed: number;
  decisions_failed: number;
  decisions_open: number;
  metrics: {
    sample_count: number;
    average_reward: number;
    median_reward: number;
    failure_rate: number;
    prediction_mae: number;
    fill_brier_score: number;
    calibration_error: number;
    ips_reward: number;
    snips_reward: number;
    doubly_robust_reward: number;
    max_drawdown: number;
    profit_per_hour: number;
  };
  profit_attribution: Record<string, number>;
  feature_importance: Array<{
    feature: string;
    correlation: number;
    importance: number;
    sample_count: number;
  }>;
  arbitrage_opportunities: Array<{
    id: string;
    kind: string;
    family: string;
    cost: number;
    expected_revenue: number;
    expected_profit: number;
    expected_hours: number;
    score: number;
    confidence: number;
  }>;
  alerts: Array<{
    id: string;
    severity: string;
    code: string;
    message: string;
    created_at: string;
  }>;
  active_events: Array<{
    id: string;
    kind: string;
    title: string;
    impact: number;
    confidence: number;
    starts_at: string;
    ends_at?: string | null;
    source: string;
  }>;
  shadow_evaluations: Array<{
    policy_name: string;
    metrics: {
      sample_count: number;
      average_reward: number;
      failure_rate: number;
      ips_reward: number;
      doubly_robust_reward: number;
    };
    confidence_low: number;
    confidence_high: number;
    promotable: boolean;
  }>;
  top_items: Array<Record<string, unknown>>;
  portfolio: {
    total_capital_budget: number;
    reserved_capital: number;
    allocated_capital: number;
    high_volatility_capital: number;
    experimental_capital: number;
    estimated_daily_interactions: number;
  };
  advanced?: any;
  config: UltimateConfig;
};

const pct = (value: number | undefined) => `${((value ?? 0) * 100).toFixed(1)}%`;
const num = (value: number | undefined, digits = 2) => (value ?? 0).toFixed(digits);

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string | number;
  detail?: string;
}) {
  return (
    <Card withBorder>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={700} size="xl">
        {value}
      </Text>
      {detail && (
        <Text size="xs" c="dimmed">
          {detail}
        </Text>
      )}
    </Card>
  );
}

export default function LearningPage() {
  const [inspector, setInspector] = useState<Inspector | null>(null);
  const [config, setConfig] = useState<UltimateConfig | null>(null);
  const [decisions, setDecisions] = useState<any[]>([]);
  const [evaluation, setEvaluation] = useState<any[]>([]);
  const [replay, setReplay] = useState<any | null>(null);
  const [forgetItemKey, setForgetItemKey] = useState("");
  const [forgetCategoryName, setForgetCategoryName] = useState("");
  const [forgetBeforeDate, setForgetBeforeDate] = useState("");
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [nextInspector, nextConfig] = await Promise.all([
        invoke<Inspector>("learning_get_inspector"),
        invoke<UltimateConfig>("learning_get_config"),
      ]);
      setInspector(nextInspector);
      setConfig(nextConfig);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const saveConfig = async (next: UltimateConfig) => {
    setBusy(true);
    try {
      const saved = await invoke<UltimateConfig>("learning_update_config", {
        config: next,
      });
      setConfig(saved);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const setMode = async (mode: OperatingMode) => {
    setBusy(true);
    try {
      const saved = await invoke<UltimateConfig>("learning_set_mode", { mode });
      setConfig(saved);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const loadDecisions = async () => {
    setBusy(true);
    try {
      const result = await invoke<any[]>("learning_get_decisions", { limit: 250 });
      setDecisions(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const runEvaluation = async () => {
    setBusy(true);
    try {
      const result = await invoke<any[]>("learning_run_offline_evaluation");
      setEvaluation(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const replayDecision = async (decisionId: string) => {
    setBusy(true);
    try {
      const result = await invoke<any>("learning_counterfactual_replay", {
        decisionId,
      });
      setReplay(result);
      setError("");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const forgetItem = async (itemKey: string) => {
    if (!window.confirm(`Forget all learning for ${itemKey}?`)) return;
    setBusy(true);
    try {
      await invoke("learning_forget_item", { itemKey });
      await refresh();
      await loadDecisions();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const forgetCategory = async (category: string) => {
    if (!category.trim()) return;
    if (!window.confirm(`Forget all learning and snapshots for category "${category}"?`)) return;

    setBusy(true);
    try {
      await invoke("learning_forget_category", { category: category.trim() });
      setForgetCategoryName("");
      await refresh();
      await loadDecisions();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const forgetBefore = async (date: string) => {
    if (!date.trim()) return;
    if (!window.confirm(`Forget learning data before ${date}?`)) return;

    const parsed = new Date(`${date}T00:00:00`);
    if (Number.isNaN(parsed.getTime())) {
      setError("Invalid forget-before date.");
      return;
    }

    setBusy(true);
    try {
      await invoke("learning_forget_before", {
        beforeRfc3339: parsed.toISOString(),
      });
      setForgetBeforeDate("");
      await refresh();
      await loadDecisions();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const exportDataset = async () => {
    setBusy(true);
    try {
      const result = await invoke<any>("learning_export_dataset");
      setError(`Exported ${result.decision_count} decisions and ${result.snapshot_count} snapshots to ${result.path}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const resetLearning = async (keepSnapshots: boolean) => {
    if (!window.confirm("Reset AlgoFrame learning data? This cannot be undone.")) return;
    setBusy(true);
    try {
      await invoke("learning_reset", { keepSnapshots });
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const healthColor = inspector?.health.healthy ? "green" : "red";

  const sortedImportance = useMemo(
    () => [...(inspector?.feature_importance ?? [])].slice(0, 20),
    [inspector],
  );

  return (
    <Stack p="md" gap="md">
      <Group justify="space-between">
        <div>
          <Title order={2}>AlgoFrame Learning</Title>
          <Text c="dimmed">
            Online ML, market recorder, risk engine, shadow policies and model health.
          </Text>
        </div>
        <Group>
          <Badge color={healthColor} size="lg">
            {inspector?.health.fallback_active
              ? "Conservative fallback"
              : inspector?.health.healthy
                ? "Healthy"
                : "Unhealthy"}
          </Badge>
          <Badge variant="light">{inspector?.model ?? "ultimate_v4"}</Badge>
          <Button variant="light" onClick={refresh} loading={busy}>
            Refresh
          </Button>
        </Group>
      </Group>

      {error && (
        <Alert color={error.startsWith("Exported") ? "green" : "red"} title="Learning status">
          {error}
        </Alert>
      )}

      <Grid>
        <Grid.Col span={{ base: 6, md: 2 }}>
          <MetricCard label="Transactions learned" value={inspector?.transactions_learned ?? 0} />
        </Grid.Col>
        <Grid.Col span={{ base: 6, md: 2 }}>
          <MetricCard label="Market snapshots" value={inspector?.snapshots_recorded ?? 0} />
        </Grid.Col>
        <Grid.Col span={{ base: 6, md: 2 }}>
          <MetricCard
            label="Completed decisions"
            value={inspector?.decisions_completed ?? 0}
            detail={`${inspector?.decisions_open ?? 0} open`}
          />
        </Grid.Col>
        <Grid.Col span={{ base: 6, md: 2 }}>
          <MetricCard label="Prediction MAE" value={`${num(inspector?.metrics.prediction_mae)}p`} />
        </Grid.Col>
        <Grid.Col span={{ base: 6, md: 2 }}>
          <MetricCard label="Profit/hour" value={`${num(inspector?.metrics.profit_per_hour)}p`} />
        </Grid.Col>
        <Grid.Col span={{ base: 6, md: 2 }}>
          <MetricCard label="Model health" value={pct(inspector?.health.score)} />
        </Grid.Col>
      </Grid>

      <Tabs defaultValue="overview">
        <Tabs.List>
          <Tabs.Tab value="overview">Overview</Tabs.Tab>
          <Tabs.Tab value="items">Items</Tabs.Tab>
          <Tabs.Tab value="policy">Policies</Tabs.Tab>
          <Tabs.Tab value="arbitrage">Arbitrage</Tabs.Tab>
          <Tabs.Tab value="features">Features</Tabs.Tab>
          <Tabs.Tab value="advanced">Advanced Intelligence</Tabs.Tab>
          <Tabs.Tab value="risk">Risk & Config</Tabs.Tab>
          <Tabs.Tab value="events">Events</Tabs.Tab>
          <Tabs.Tab value="decisions" onClick={loadDecisions}>
            Decisions
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="overview" pt="md">
          <Grid>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Card withBorder>
                <Title order={4}>Model health</Title>
                <Progress mt="sm" value={(inspector?.health.score ?? 0) * 100} color={healthColor} />
                <Stack mt="sm" gap={4}>
                  <Text size="sm">Recent reward: {num(inspector?.health.recent_reward, 4)}</Text>
                  <Text size="sm">Baseline reward: {num(inspector?.health.baseline_reward, 4)}</Text>
                  <Text size="sm">Recent failure rate: {pct(inspector?.health.recent_failure_rate)}</Text>
                  <Text size="sm">Calibration error: {pct(inspector?.health.calibration_error)}</Text>
                  <Text size="sm">Feature/data drift: {pct(inspector?.health.drift_score)}</Text>
                  <Text size="sm">Drawdown: {pct(inspector?.health.drawdown)}</Text>
                  {(inspector?.health.reasons ?? []).map((reason) => (
                    <Text size="sm" c="orange" key={reason}>
                      {reason}
                    </Text>
                  ))}
                </Stack>
              </Card>
            </Grid.Col>
            <Grid.Col span={{ base: 12, md: 6 }}>
              <Card withBorder>
                <Title order={4}>Portfolio</Title>
                <Stack mt="sm" gap={4}>
                  <Text size="sm">Budget: {num(inspector?.portfolio.total_capital_budget)}p</Text>
                  <Text size="sm">Allocated: {num(inspector?.portfolio.allocated_capital)}p</Text>
                  <Text size="sm">Reserved: {num(inspector?.portfolio.reserved_capital)}p</Text>
                  <Text size="sm">
                    Experimental: {num(inspector?.portfolio.experimental_capital)}p
                  </Text>
                  <Text size="sm">
                    High volatility: {num(inspector?.portfolio.high_volatility_capital)}p
                  </Text>
                  <Text size="sm">
                    Est. daily interactions: {num(inspector?.portfolio.estimated_daily_interactions, 0)}
                  </Text>
                </Stack>
              </Card>
            </Grid.Col>
          </Grid>

          <Card withBorder mt="md">
            <Title order={4}>Recent alerts</Title>
            <Stack mt="sm" gap="xs">
              {(inspector?.alerts ?? []).slice(0, 10).map((alert) => (
                <Alert
                  key={alert.id}
                  color={
                    alert.severity === "critical"
                      ? "red"
                      : alert.severity === "warning"
                        ? "yellow"
                        : "blue"
                  }
                  title={`${alert.code} · ${new Date(alert.created_at).toLocaleString()}`}
                >
                  {alert.message}
                </Alert>
              ))}
              {(inspector?.alerts ?? []).length === 0 && (
                <Text c="dimmed" size="sm">
                  No learning alerts.
                </Text>
              )}
            </Stack>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="items" pt="md">
          <Card withBorder>
            <ScrollArea h={520}>
              <Table striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Item</Table.Th>
                    <Table.Th>Trades</Table.Th>
                    <Table.Th>Confidence</Table.Th>
                    <Table.Th>Avg profit</Table.Th>
                    <Table.Th>Buy fill</Table.Th>
                    <Table.Th>Sell fill</Table.Th>
                    <Table.Th>Sell-through</Table.Th>
                    <Table.Th>MAE</Table.Th>
                    <Table.Th>Open</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(inspector?.top_items ?? []).map((item: any) => (
                    <Table.Tr key={String(item.item_key)}>
                      <Table.Td>{String(item.item_key)}</Table.Td>
                      <Table.Td>{Number(item.trades ?? 0)}</Table.Td>
                      <Table.Td>{pct(Number(item.confidence ?? 0))}</Table.Td>
                      <Table.Td>{num(Number(item.avg_profit ?? 0))}p</Table.Td>
                      <Table.Td>{num(Number(item.avg_buy_fill_hours ?? 0))}h</Table.Td>
                      <Table.Td>{num(Number(item.avg_sell_fill_hours ?? 0))}h</Table.Td>
                      <Table.Td>{pct(Number(item.sell_through ?? 0))}</Table.Td>
                      <Table.Td>{num(Number(item.prediction_mae ?? 0))}p</Table.Td>
                      <Table.Td>{Number(item.open_units ?? 0)}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="policy" pt="md">
          <Group mb="md">
            <Button onClick={runEvaluation} loading={busy}>
              Run offline policy evaluation
            </Button>
          </Group>
          <Grid>
            {(evaluation.length ? evaluation : inspector?.shadow_evaluations ?? []).map((entry: any) => (
              <Grid.Col span={{ base: 12, md: 6 }} key={entry.policy_name}>
                <Card withBorder>
                  <Title order={4}>{entry.policy_name}</Title>
                  <Text size="sm">Samples: {entry.metrics.sample_count}</Text>
                  <Text size="sm">Reward: {num(entry.metrics.average_reward, 4)}</Text>
                  <Text size="sm">IPS: {num(entry.metrics.ips_reward, 4)}</Text>
                  <Text size="sm">
                    Doubly robust: {num(entry.metrics.doubly_robust_reward, 4)}
                  </Text>
                  <Text size="sm">Failure rate: {pct(entry.metrics.failure_rate)}</Text>
                  <Text size="sm">
                    CI: [{num(entry.confidence_low, 4)}, {num(entry.confidence_high, 4)}]
                  </Text>
                </Card>
              </Grid.Col>
            ))}
          </Grid>
        </Tabs.Panel>

        <Tabs.Panel value="arbitrage" pt="md">
          <Card withBorder>
            <Table striped>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Type</Table.Th>
                  <Table.Th>Family</Table.Th>
                  <Table.Th>Cost</Table.Th>
                  <Table.Th>Revenue</Table.Th>
                  <Table.Th>Profit</Table.Th>
                  <Table.Th>Score</Table.Th>
                  <Table.Th>Confidence</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {(inspector?.arbitrage_opportunities ?? []).map((op) => (
                  <Table.Tr key={op.id}>
                    <Table.Td>{op.kind}</Table.Td>
                    <Table.Td>{op.family}</Table.Td>
                    <Table.Td>{num(op.cost)}p</Table.Td>
                    <Table.Td>{num(op.expected_revenue)}p</Table.Td>
                    <Table.Td>{num(op.expected_profit)}p</Table.Td>
                    <Table.Td>{pct(op.score)}</Table.Td>
                    <Table.Td>{pct(op.confidence)}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="features" pt="md">
          <Card withBorder>
            <Title order={4}>Automatic feature importance</Title>
            <Table mt="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Feature</Table.Th>
                  <Table.Th>Importance</Table.Th>
                  <Table.Th>Correlation</Table.Th>
                  <Table.Th>Samples</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {sortedImportance.map((feature) => (
                  <Table.Tr key={feature.feature}>
                    <Table.Td>{feature.feature}</Table.Td>
                    <Table.Td>{pct(feature.importance)}</Table.Td>
                    <Table.Td>{num(feature.correlation, 3)}</Table.Td>
                    <Table.Td>{feature.sample_count}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="advanced" pt="md">
          <Stack gap="md">
            <Grid>
              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card withBorder>
                  <Title order={4}>Distributional profit model</Title>
                  <Stack mt="sm" gap={4}>
                    <Text size="sm">
                      P10 / P50 / P90:{" "}
                      {num(inspector?.advanced?.distribution?.p10)}p /{" "}
                      {num(inspector?.advanced?.distribution?.p50)}p /{" "}
                      {num(inspector?.advanced?.distribution?.p90)}p
                    </Text>
                    <Text size="sm">
                      Loss probability:{" "}
                      {pct(inspector?.advanced?.distribution?.loss_probability)}
                    </Text>
                    <Text size="sm">
                      CVaR 10%: {num(inspector?.advanced?.distribution?.cvar_10)}p
                    </Text>
                    <Text size="sm">
                      80% conformal interval:{" "}
                      {num(inspector?.advanced?.distribution?.conformal_low_80)}p →{" "}
                      {num(inspector?.advanced?.distribution?.conformal_high_80)}p
                    </Text>
                    <Text size="sm">
                      Aleatoric uncertainty:{" "}
                      {pct(inspector?.advanced?.distribution?.aleatoric_uncertainty)}
                    </Text>
                    <Text size="sm">
                      Epistemic uncertainty:{" "}
                      {pct(inspector?.advanced?.distribution?.epistemic_uncertainty)}
                    </Text>
                  </Stack>
                </Card>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card withBorder>
                  <Title order={4}>Monte Carlo portfolio</Title>
                  <Stack mt="sm" gap={4}>
                    <Text size="sm">
                      Simulations: {inspector?.advanced?.monte_carlo?.simulations ?? 0}
                    </Text>
                    <Text size="sm">
                      Expected/day: {num(inspector?.advanced?.monte_carlo?.expected_daily_profit)}p
                    </Text>
                    <Text size="sm">
                      P05 / P50 / P95:{" "}
                      {num(inspector?.advanced?.monte_carlo?.p05_daily_profit)}p /{" "}
                      {num(inspector?.advanced?.monte_carlo?.p50_daily_profit)}p /{" "}
                      {num(inspector?.advanced?.monte_carlo?.p95_daily_profit)}p
                    </Text>
                    <Text size="sm">
                      P(loss): {pct(inspector?.advanced?.monte_carlo?.probability_of_loss)}
                    </Text>
                    <Text size="sm">
                      P(drawdown &gt;10%):{" "}
                      {pct(inspector?.advanced?.monte_carlo?.probability_drawdown_gt_10pct)}
                    </Text>
                  </Stack>
                </Card>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 4 }}>
                <Card withBorder>
                  <Title order={4}>Learning velocity</Title>
                  <Stack mt="sm" gap={4}>
                    <Text size="sm">
                      MAE: {num(inspector?.advanced?.learning_velocity?.early_prediction_mae)}p →{" "}
                      {num(inspector?.advanced?.learning_velocity?.recent_prediction_mae)}p
                    </Text>
                    <Text size="sm">
                      MAE improvement:{" "}
                      {num(inspector?.advanced?.learning_velocity?.mae_improvement_pct, 1)}%
                    </Text>
                    <Text size="sm">
                      Reward: {num(inspector?.advanced?.learning_velocity?.early_reward, 4)} →{" "}
                      {num(inspector?.advanced?.learning_velocity?.recent_reward, 4)}
                    </Text>
                    <Text size="sm">
                      Reward improvement:{" "}
                      {num(inspector?.advanced?.learning_velocity?.reward_improvement_pct, 1)}%
                    </Text>
                  </Stack>
                </Card>
              </Grid.Col>
            </Grid>

            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card withBorder>
                  <Title order={4}>Validation health</Title>
                  <Stack mt="sm" gap={4}>
                    <Text size="sm">
                      Leakage check:{" "}
                      {inspector?.advanced?.leakage?.healthy ? "Healthy" : "Warning"} ·{" "}
                      {inspector?.advanced?.leakage?.suspicious_decisions ?? 0} suspicious
                    </Text>
                    <Text size="sm">
                      Walk-forward reward:{" "}
                      {num(inspector?.advanced?.walk_forward?.average_test_reward, 4)}
                    </Text>
                    <Text size="sm">
                      Walk-forward MAE:{" "}
                      {num(inspector?.advanced?.walk_forward?.average_test_mae)}p
                    </Text>
                    <Text size="sm">
                      Temporal stability:{" "}
                      {pct(inspector?.advanced?.walk_forward?.stability)}
                    </Text>
                    <Text size="sm">
                      Nonlinear expert:{" "}
                      {inspector?.advanced?.nonlinear_model?.trained ? "trained" : "collecting data"}
                    </Text>
                    <Text size="sm">
                      Nonlinear validation R²:{" "}
                      {num(inspector?.advanced?.nonlinear_model?.validation_r2, 3)}
                    </Text>
                  </Stack>
                </Card>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card withBorder>
                  <Title order={4}>Mixture-of-experts gating</Title>
                  <Table mt="sm">
                    <Table.Tbody>
                      {Object.entries(inspector?.advanced?.expert_weights ?? {}).map(
                        ([key, value]) => (
                          <Table.Tr key={key}>
                            <Table.Td>{key.replaceAll("_", " ")}</Table.Td>
                            <Table.Td>{pct(Number(value))}</Table.Td>
                          </Table.Tr>
                        ),
                      )}
                    </Table.Tbody>
                  </Table>
                </Card>
              </Grid.Col>
            </Grid>

            <Card withBorder>
              <Title order={4}>Causal action effects</Title>
              <Table mt="sm" striped>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Action</Table.Th>
                    <Table.Th>Baseline</Table.Th>
                    <Table.Th>Samples</Table.Th>
                    <Table.Th>Estimated effect</Table.Th>
                    <Table.Th>95% interval</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(inspector?.advanced?.causal_effects ?? []).slice(0, 30).map((row: any) => (
                    <Table.Tr key={`${row.action}:${row.baseline_action}`}>
                      <Table.Td>{row.action}</Table.Td>
                      <Table.Td>{row.baseline_action}</Table.Td>
                      <Table.Td>{row.sample_count}</Table.Td>
                      <Table.Td>{num(row.estimated_treatment_effect, 4)}</Table.Td>
                      <Table.Td>
                        {num(row.confidence_low, 4)} → {num(row.confidence_high, 4)}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Card>

            <Card withBorder>
              <Title order={4}>Confidence heatmap</Title>
              <Table mt="sm" striped>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Category</Table.Th>
                    <Table.Th>Samples</Table.Th>
                    <Table.Th>Confidence</Table.Th>
                    <Table.Th>MAE</Table.Th>
                    <Table.Th>Reward</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(inspector?.advanced?.confidence_heatmap ?? []).map((row: any) => (
                    <Table.Tr key={row.category}>
                      <Table.Td>{row.category}</Table.Td>
                      <Table.Td>{row.samples}</Table.Td>
                      <Table.Td>{pct(row.confidence)}</Table.Td>
                      <Table.Td>{num(row.prediction_mae)}p</Table.Td>
                      <Table.Td>{num(row.reward, 4)}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Card>

            <Grid>
              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card withBorder>
                  <Title order={4}>Baseline comparison</Title>
                  <Table mt="sm">
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Strategy</Table.Th>
                        <Table.Th>Reward</Table.Th>
                        <Table.Th>Champion advantage</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {(inspector?.advanced?.baselines ?? []).map((row: any) => (
                        <Table.Tr key={row.strategy}>
                          <Table.Td>{row.strategy}</Table.Td>
                          <Table.Td>{num(row.estimated_reward, 4)}</Table.Td>
                          <Table.Td>{num(row.difference_vs_champion, 4)}</Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                </Card>
              </Grid.Col>

              <Grid.Col span={{ base: 12, md: 6 }}>
                <Card withBorder>
                  <Title order={4}>Stress tests</Title>
                  <Table mt="sm">
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Scenario</Table.Th>
                        <Table.Th>Reward ×</Table.Th>
                        <Table.Th>Drawdown</Table.Th>
                        <Table.Th>Blocked</Table.Th>
                        <Table.Th>Status</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {(inspector?.advanced?.stress_tests ?? []).map((row: any) => (
                        <Table.Tr key={row.scenario}>
                          <Table.Td>{row.scenario}</Table.Td>
                          <Table.Td>{num(row.expected_reward_multiplier, 2)}</Table.Td>
                          <Table.Td>{pct(row.expected_drawdown)}</Table.Td>
                          <Table.Td>{pct(row.blocked_fraction)}</Table.Td>
                          <Table.Td>{row.safe ? "Safe" : "Needs review"}</Table.Td>
                        </Table.Tr>
                      ))}
                    </Table.Tbody>
                  </Table>
                </Card>
              </Grid.Col>
            </Grid>

            <Card withBorder>
              <Title order={4}>Inventory ageing policy</Title>
              <Table mt="sm">
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Category</Table.Th>
                    <Table.Th>Samples</Table.Th>
                    <Table.Th>Soft limit</Table.Th>
                    <Table.Th>Liquidation pressure</Table.Th>
                    <Table.Th>Reward before</Table.Th>
                    <Table.Th>Reward after</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(inspector?.advanced?.inventory_age_policies ?? []).map((row: any) => (
                    <Table.Tr key={row.category}>
                      <Table.Td>{row.category}</Table.Td>
                      <Table.Td>{row.samples}</Table.Td>
                      <Table.Td>{num(row.optimal_soft_limit_hours)}h</Table.Td>
                      <Table.Td>{num(row.liquidation_pressure_after_hours)}h</Table.Td>
                      <Table.Td>{num(row.expected_reward_before_limit, 4)}</Table.Td>
                      <Table.Td>{num(row.expected_reward_after_limit, 4)}</Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Card>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="risk" pt="md">
          {config && (
            <Stack gap="md">
              <Card withBorder>
                <Title order={4}>Operating mode</Title>
                <Select
                  mt="sm"
                  value={config.mode}
                  data={[
                    { value: "paper", label: "Paper — no live WFM orders" },
                    { value: "conservative", label: "Conservative" },
                    { value: "balanced", label: "Balanced" },
                    { value: "growth", label: "Growth" },
                    { value: "liquid", label: "Liquid — prioritize turnover" },
                  ]}
                  onChange={(value) => value && setMode(value as OperatingMode)}
                />
              </Card>

              <Card withBorder>
                <Title order={4}>Capital guardrails</Title>
                <Grid mt="sm">
                  {[
                    ["Cash reserve", "cash_reserve_pct"],
                    ["Max item exposure", "max_item_exposure_pct"],
                    ["Max category exposure", "max_category_exposure_pct"],
                    ["Max correlated family exposure", "max_family_exposure_pct"],
                    ["Experimental budget", "experimental_budget_pct"],
                    ["Max single decision", "max_single_decision_pct"],
                    ["Max drawdown", "max_drawdown_pct"],
                  ].map(([label, key]) => (
                    <Grid.Col span={{ base: 12, md: 4 }} key={key}>
                      <NumberInput
                        label={label}
                        value={(config as any)[key] * 100}
                        suffix="%"
                        min={0}
                        max={100}
                        onChange={(value) =>
                          setConfig({
                            ...config,
                            [key]: Number(value) / 100,
                          })
                        }
                      />
                    </Grid.Col>
                  ))}
                </Grid>

                <Divider my="md" />
                <Title order={5}>Adaptive repricing</Title>
                <Grid mt="sm">
                  <Grid.Col span={{ base: 12, md: 4 }}>
                    <NumberInput
                      label="Minimum reprice interval"
                      description="Fast-moving market cooldown"
                      value={config.min_reprice_minutes}
                      suffix=" min"
                      min={0}
                      max={120}
                      onChange={(value) =>
                        setConfig({
                          ...config,
                          min_reprice_minutes: Number(value),
                        })
                      }
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 4 }}>
                    <NumberInput
                      label="Stable / price-war interval"
                      description="Longer cooldown in slow or competitive books"
                      value={config.stable_reprice_minutes}
                      suffix=" min"
                      min={0}
                      max={240}
                      onChange={(value) =>
                        setConfig({
                          ...config,
                          stable_reprice_minutes: Number(value),
                        })
                      }
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 4 }}>
                    <NumberInput
                      label="Minimum reward gain"
                      description="Required improvement before changing price"
                      value={config.reprice_min_reward_gain_pct * 100}
                      suffix="%"
                      min={0}
                      max={100}
                      onChange={(value) =>
                        setConfig({
                          ...config,
                          reprice_min_reward_gain_pct: Number(value) / 100,
                        })
                      }
                    />
                  </Grid.Col>
                </Grid>

                <Button mt="md" onClick={() => saveConfig(config)} loading={busy}>
                  Save risk settings
                </Button>
              </Card>

              <Card withBorder>
                <Title order={4}>Champion / challenger</Title>
                <Grid mt="sm">
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <Select
                      label="Champion policy"
                      value={config.champion_policy}
                      data={[
                        { value: "balanced", label: "Balanced" },
                        { value: "fast_turnover", label: "Fast turnover" },
                        { value: "max_reward", label: "Maximum reward" },
                        { value: "conservative", label: "Conservative" },
                        { value: "aggressive", label: "Aggressive" },
                      ]}
                      onChange={(value) =>
                        value &&
                        setConfig({
                          ...config,
                          champion_policy: value,
                        })
                      }
                    />
                  </Grid.Col>
                  <Grid.Col span={{ base: 12, md: 6 }}>
                    <Select
                      label="Challenger policy"
                      value={config.challenger_policy}
                      data={[
                        { value: "balanced", label: "Balanced" },
                        { value: "fast_turnover", label: "Fast turnover" },
                        { value: "max_reward", label: "Maximum reward" },
                        { value: "conservative", label: "Conservative" },
                        { value: "aggressive", label: "Aggressive" },
                      ]}
                      onChange={(value) =>
                        value &&
                        setConfig({
                          ...config,
                          challenger_policy: value,
                        })
                      }
                    />
                  </Grid.Col>
                </Grid>
                <Button mt="md" onClick={() => saveConfig(config)} loading={busy}>
                  Save policy settings
                </Button>
              </Card>

              <Card withBorder>
                <Title order={4}>Learning systems</Title>
                <Grid mt="sm">
                  {[
                    ["Automatic tuning", "automatic_tuning"],
                    ["Automatic promotion", "automatic_promotion"],
                    ["Automatic rollback", "automatic_rollback"],
                    ["Shadow policies", "shadow_mode"],
                    ["Market recorder", "market_recording"],
                    ["Event intelligence", "event_intelligence"],
                    ["Arbitrage engine", "arbitrage_engine"],
                    ["Continuous prices", "continuous_price_optimization"],
                    ["Quantity optimization", "quantity_optimization"],
                    ["Survival models", "survival_models"],
                    ["Hierarchical learning", "hierarchical_learning"],
                    ["Anomaly detection", "anomaly_detection"],
                    ["Regime detection", "regime_detection"],
                    ["Opportunity forecasting", "opportunity_forecasting"],
                    ["Distributional predictions", "distributional_predictions"],
                    ["CVaR downside-risk optimization", "cvar_risk_optimization"],
                    ["Conformal intervals", "conformal_intervals"],
                    ["Uncertainty decomposition", "uncertainty_decomposition"],
                    ["Learned market states", "learned_market_states"],
                    ["Order-book persistence", "book_persistence_model"],
                    ["Causal evaluation", "causal_evaluation"],
                    ["Active learning", "active_learning"],
                    ["Pareto portfolio diagnostics", "pareto_portfolio"],
                    ["Walk-forward validation", "walk_forward_validation"],
                    ["Stress testing", "stress_testing"],
                    ["Monte Carlo portfolio", "monte_carlo_portfolio"],
                    ["Automatic ablation", "automatic_ablation"],
                    ["Leakage detection", "leakage_detection"],
                    ["Nonlinear expert", "nonlinear_expert"],
                    ["Mixture of experts", "mixture_of_experts"],
                  ].map(([label, key]) => (
                    <Grid.Col span={{ base: 12, md: 4 }} key={key}>
                      <Switch
                        label={label}
                        checked={Boolean((config as any)[key])}
                        onChange={(event) =>
                          setConfig({
                            ...config,
                            [key]: event.currentTarget.checked,
                          })
                        }
                      />
                    </Grid.Col>
                  ))}
                </Grid>
                <Button mt="md" onClick={() => saveConfig(config)} loading={busy}>
                  Save learning settings
                </Button>
              </Card>

              <Card withBorder>
                <Title order={4}>Data controls</Title>

                <Grid mt="sm">
                  <Grid.Col span={{ base: 12, md: 4 }}>
                    <TextInput
                      label="Forget exact item key"
                      placeholder="wfm-id|subtype-json"
                      value={forgetItemKey}
                      onChange={(event) => setForgetItemKey(event.currentTarget.value)}
                    />
                    <Button
                      mt="xs"
                      variant="light"
                      color="orange"
                      disabled={!forgetItemKey.trim()}
                      onClick={() => forgetItem(forgetItemKey.trim())}
                      loading={busy}
                    >
                      Forget item
                    </Button>
                  </Grid.Col>

                  <Grid.Col span={{ base: 12, md: 4 }}>
                    <TextInput
                      label="Forget category"
                      placeholder="prime, arcane, relic, mod..."
                      value={forgetCategoryName}
                      onChange={(event) =>
                        setForgetCategoryName(event.currentTarget.value)
                      }
                    />
                    <Button
                      mt="xs"
                      variant="light"
                      color="orange"
                      disabled={!forgetCategoryName.trim()}
                      onClick={() => forgetCategory(forgetCategoryName)}
                      loading={busy}
                    >
                      Forget category
                    </Button>
                  </Grid.Col>

                  <Grid.Col span={{ base: 12, md: 4 }}>
                    <TextInput
                      type="date"
                      label="Forget data before"
                      value={forgetBeforeDate}
                      onChange={(event) => setForgetBeforeDate(event.currentTarget.value)}
                    />
                    <Button
                      mt="xs"
                      variant="light"
                      color="orange"
                      disabled={!forgetBeforeDate}
                      onClick={() => forgetBefore(forgetBeforeDate)}
                      loading={busy}
                    >
                      Forget older data
                    </Button>
                  </Grid.Col>
                </Grid>

                <Divider my="md" />

                <Group>
                  <Button variant="light" onClick={exportDataset} loading={busy}>
                    Export learning dataset
                  </Button>
                  <Button
                    color="orange"
                    variant="light"
                    onClick={() => resetLearning(true)}
                    loading={busy}
                  >
                    Reset models, keep market snapshots
                  </Button>
                  <Button
                    color="red"
                    variant="light"
                    onClick={() => resetLearning(false)}
                    loading={busy}
                  >
                    Reset all learning data
                  </Button>
                </Group>
              </Card>
            </Stack>
          )}
        </Tabs.Panel>

        <Tabs.Panel value="events" pt="md">
          <Group mb="sm">
            <Button
              variant="light"
              loading={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await invoke("learning_refresh_events");
                  await refresh();
                } catch (e) {
                  setError(String(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              Refresh Warframe events
            </Button>
          </Group>
          <Card withBorder>
            <Title order={4}>Active market events</Title>
            <Table mt="sm">
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Event</Table.Th>
                  <Table.Th>Type</Table.Th>
                  <Table.Th>Impact</Table.Th>
                  <Table.Th>Confidence</Table.Th>
                  <Table.Th>Source</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {(inspector?.active_events ?? []).map((event) => (
                  <Table.Tr key={event.id}>
                    <Table.Td>{event.title}</Table.Td>
                    <Table.Td>{event.kind}</Table.Td>
                    <Table.Td>{event.impact > 0 ? "+" : ""}{num(event.impact, 2)}</Table.Td>
                    <Table.Td>{pct(event.confidence)}</Table.Td>
                    <Table.Td>{event.source}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="decisions" pt="md">
          <Group mb="sm">
            <Button variant="light" onClick={loadDecisions} loading={busy}>
              Reload decisions
            </Button>
          </Group>
          <Card withBorder>
            <ScrollArea h={600}>
              <Table striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Time</Table.Th>
                    <Table.Th>Item</Table.Th>
                    <Table.Th>Side</Table.Th>
                    <Table.Th>Action</Table.Th>
                    <Table.Th>Price × Qty</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th>Predicted</Table.Th>
                    <Table.Th>Actual</Table.Th>
                    <Table.Th>Propensity</Table.Th>
                    <Table.Th>Regime</Table.Th>
                    <Table.Th>Actions</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {decisions.slice().reverse().map((decision) => (
                    <Table.Tr key={decision.id}>
                      <Table.Td>{new Date(decision.created_at).toLocaleString()}</Table.Td>
                      <Table.Td>{decision.item_name}</Table.Td>
                      <Table.Td>{decision.side}</Table.Td>
                      <Table.Td>{decision.chosen_action}</Table.Td>
                      <Table.Td>
                        {decision.price}p × {decision.quantity}
                      </Table.Td>
                      <Table.Td>{decision.status}</Table.Td>
                      <Table.Td>{num(decision.predicted_profit)}p</Table.Td>
                      <Table.Td>
                        {decision.actual_profit == null ? "—" : `${num(decision.actual_profit)}p`}
                      </Table.Td>
                      <Table.Td>{pct(decision.chosen_propensity)}</Table.Td>
                      <Table.Td>{decision.regime}</Table.Td>
                      <Table.Td>
                        <Group gap="xs" wrap="nowrap">
                          <Button
                            size="xs"
                            variant="light"
                            onClick={() => replayDecision(decision.id)}
                          >
                            Replay
                          </Button>
                          <Button
                            size="xs"
                            color="orange"
                            variant="subtle"
                            onClick={() => forgetItem(decision.item_key)}
                          >
                            Forget item
                          </Button>
                        </Group>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Card>

          {replay && (
            <Card withBorder mt="md">
              <Group justify="space-between">
                <div>
                  <Title order={4}>Digital-twin counterfactual replay</Title>
                  <Text size="sm" c="dimmed">
                    Recorded external market snapshots only; queue priority is not assumed.
                  </Text>
                </div>
                <Badge variant="light">
                  {replay.recorded_snapshot_count ?? 0} snapshots
                </Badge>
              </Group>

              <Grid mt="sm">
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <MetricCard
                    label="Executed action"
                    value={`${replay.decision?.price ?? 0}p × ${replay.decision?.quantity ?? 0}`}
                    detail={replay.decision?.chosen_action}
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <MetricCard
                    label="Best replay price"
                    value={
                      replay.best_alternative
                        ? `${replay.best_alternative.price}p × ${replay.best_alternative.quantity}`
                        : "—"
                    }
                  />
                </Grid.Col>
                <Grid.Col span={{ base: 12, md: 4 }}>
                  <MetricCard
                    label="Best replay reward"
                    value={
                      replay.best_alternative
                        ? num(Number(replay.best_alternative.reward ?? 0), 4)
                        : "—"
                    }
                  />
                </Grid.Col>
              </Grid>

              <ScrollArea h={300} mt="md">
                <Table striped>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Price</Table.Th>
                      <Table.Th>Qty</Table.Th>
                      <Table.Th>Filled</Table.Th>
                      <Table.Th>Sold</Table.Th>
                      <Table.Th>Fill time</Table.Th>
                      <Table.Th>Cycle</Table.Th>
                      <Table.Th>Profit</Table.Th>
                      <Table.Th>Reward</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {(replay.alternatives ?? []).slice(0, 50).map((alt: any, index: number) => (
                      <Table.Tr key={`${alt.price}-${alt.quantity}-${index}`}>
                        <Table.Td>{alt.price}p</Table.Td>
                        <Table.Td>{alt.quantity}</Table.Td>
                        <Table.Td>{alt.filled ? "yes" : "no"}</Table.Td>
                        <Table.Td>
                          {alt.sold == null ? "—" : alt.sold ? "yes" : "no"}
                        </Table.Td>
                        <Table.Td>
                          {alt.fill_hours == null ? "—" : `${num(Number(alt.fill_hours))}h`}
                        </Table.Td>
                        <Table.Td>{num(Number(alt.cycle_hours ?? 0))}h</Table.Td>
                        <Table.Td>{num(Number(alt.profit ?? 0))}p</Table.Td>
                        <Table.Td>{num(Number(alt.reward ?? 0), 4)}</Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </ScrollArea>
            </Card>
          )}
        </Tabs.Panel>
      </Tabs>

      <Divider />
      <Text size="xs" c="dimmed">
        Feature schema {inspector?.feature_schema_version ?? 4} · Reward schema{" "}
        {inspector?.reward_version ?? 4} · Policy schema {inspector?.policy_version ?? 4}
      </Text>
    </Stack>
  );
}
