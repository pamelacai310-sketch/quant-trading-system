"""Event-driven game-theoretic causal analysis.

This module turns unstructured news / policy events into a transparent,
auditable game-causal layer:
1. ingest news and policy calendar rows;
2. quantify geopolitical, monetary, AI, trade and credit risk;
3. build event-driven causal chains;
4. decide which competing market logic is dominant under current conditions.

The implementation is deliberately deterministic and explainable. It is not a
black-box news model; every score is backed by matched terms and explicit
conditions so downstream trading reports can say why a logic won.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").replace("%", "").strip()
    if text in {"", "-", "--", "None", "nan"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


@dataclass
class NewsEvent:
    """Normalized event row accepted by the game-causal engine."""

    event_id: str
    timestamp: str
    title: str
    summary: str = ""
    source: str = "unknown"
    relevance_score: float = 1.0
    sentiment_score: float = 0.0
    tags: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join([self.title, self.summary, " ".join(self.tags)]).strip()


@dataclass
class EventCausalChain:
    chain_id: str
    trigger_event_ids: List[str]
    trigger: str
    causal_steps: List[str]
    affected_assets: List[str]
    expected_direction: str
    confidence: float


@dataclass
class GameForce:
    name: str
    score: float
    expected_direction: str
    conditions: List[str]
    evidence_terms: List[str]


@dataclass
class GameDominance:
    asset: str
    dominant_logic: str
    winner: str
    winner_score: float
    runner_up: str
    runner_up_score: float
    confidence: float
    expected_direction: str
    market_implication: str
    forces: List[GameForce]


@dataclass
class PricingAssetRule:
    asset: str
    expected_direction: str
    weight: float = 1.0
    rationale: str = ""


@dataclass
class ContextRule:
    path: str
    operator: str
    threshold: float
    weight: float
    rationale: str


@dataclass
class GameSideSpec:
    side_id: str
    name: str
    core_thesis: str
    transmission_chain: List[str]
    evolution_path: List[str]
    dominance_conditions: List[str]
    market_pricing_if_wins: Dict[str, str]
    reversal_signals: List[str]
    domain_weights: Dict[str, float]
    context_rules: List[ContextRule] = field(default_factory=list)
    price_confirmations: List[PricingAssetRule] = field(default_factory=list)


@dataclass
class GameRelationSpec:
    relation_id: str
    relation_name: str
    category: str
    side_a: GameSideSpec
    side_b: GameSideSpec
    sensitive_assets: List[str]
    observation_checklist: List[str]
    layer_assets: Dict[str, List[str]]


@dataclass
class GameRelationReport:
    relation_id: str
    relation_name: str
    category: str
    core_logic: Dict[str, str]
    sensitive_assets: List[str]
    transmission_mechanisms: Dict[str, List[str]]
    evolution_paths: Dict[str, List[str]]
    dominance_conditions: Dict[str, List[str]]
    price_confirmation: Dict[str, Dict[str, Any]]
    side_scores: Dict[str, float]
    current_judgement: Dict[str, Any]
    layer_winners: Dict[str, Dict[str, Any]]
    market_pricing_forecast: Dict[str, Any]
    key_reversal_signals: List[str]
    observation_checklist: List[str]
    identification_status: Dict[str, Any] = field(default_factory=dict)
    actionability: str = "observe_only"


@dataclass
class EventWindowSnapshot:
    event_id: str
    asset: str
    event_timestamp: str
    pre_days: int
    post_days: int
    pre_return: float
    post_return: float
    event_to_latest_return: float
    volatility_ratio: float
    volume_ratio: float
    observed_direction: str
    usable_for_learning: bool


@dataclass
class PriceConfirmationMemoryRecord:
    relation_id: str
    side_id: str
    asset: str
    expected_direction: str
    learned_weight: float
    reliability: float
    sample_count: int
    avg_lead_lag_days: float
    last_updated: str


class GameCausalAnalysisEngine:
    """Quantify event-driven causal games and logic dominance."""

    def __init__(
        self,
        price_confirmation_memory: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.price_confirmation_memory: Dict[str, PriceConfirmationMemoryRecord] = {}
        for key, value in (price_confirmation_memory or {}).items():
            if isinstance(value, PriceConfirmationMemoryRecord):
                self.price_confirmation_memory[key] = value
            elif isinstance(value, Mapping):
                self.price_confirmation_memory[key] = PriceConfirmationMemoryRecord(
                    relation_id=str(value.get("relation_id", "")),
                    side_id=str(value.get("side_id", "")),
                    asset=str(value.get("asset", "")),
                    expected_direction=str(value.get("expected_direction", "")),
                    learned_weight=_to_float(value.get("learned_weight"), 1.0),
                    reliability=_to_float(value.get("reliability"), 0.5),
                    sample_count=int(_to_float(value.get("sample_count"), 0.0)),
                    avg_lead_lag_days=_to_float(value.get("avg_lead_lag_days"), 0.0),
                    last_updated=str(value.get("last_updated", datetime.utcnow().isoformat())),
                )

    DOMAIN_ONTOLOGY: Dict[str, Dict[str, Any]] = {
        "geopolitical_energy": {
            "description": "Geopolitical supply shock and energy security risk",
            "assets": ["crude_oil", "natural_gas", "gold", "shipping", "copper", "energy_equities"],
            "terms": {
                "iran": 1.00,
                "伊朗": 1.00,
                "hormuz": 1.00,
                "霍尔木兹": 1.00,
                "strait": 0.55,
                "opec": 0.80,
                "opec+": 0.85,
                "欧佩克": 0.80,
                "uae": 0.65,
                "阿联酋": 0.65,
                "war": 0.90,
                "战争": 0.90,
                "missile": 0.75,
                "导弹": 0.75,
                "ceasefire": 0.70,
                "停火": 0.70,
                "sanction": 0.72,
                "制裁": 0.72,
                "blockade": 0.95,
                "封锁": 0.95,
                "red sea": 0.75,
                "红海": 0.75,
                "supply disruption": 0.95,
                "供应中断": 0.95,
            },
        },
        "monetary_inflation": {
            "description": "Monetary policy credibility, inflation and real-rate pressure",
            "assets": ["gold", "duration_equities", "usd", "bonds", "industrial_metals"],
            "terms": {
                "fed": 0.85,
                "fomc": 0.80,
                "美联储": 0.85,
                "powell": 0.65,
                "warsh": 0.90,
                "沃什": 0.90,
                "rate hike": 0.88,
                "加息": 0.88,
                "rate cut": 0.75,
                "降息": 0.75,
                "inflation": 0.82,
                "通胀": 0.82,
                "cpi": 0.65,
                "ppi": 0.55,
                "real rate": 0.86,
                "实际利率": 0.86,
                "credibility": 0.78,
                "可信度": 0.78,
                "policy space": 0.82,
                "政策空间": 0.82,
                "jobs": 0.50,
                "employment": 0.50,
                "就业": 0.50,
            },
        },
        "ai_technology": {
            "description": "AI capex, chips, energy bottlenecks and technology restrictions",
            "assets": ["ai_equities", "semiconductors", "copper", "power_infrastructure", "data_centers"],
            "terms": {
                "ai": 0.75,
                "人工智能": 0.75,
                "capex": 0.82,
                "资本开支": 0.82,
                "data center": 0.82,
                "数据中心": 0.82,
                "chip": 0.72,
                "semiconductor": 0.78,
                "芯片": 0.72,
                "半导体": 0.78,
                "taiwan": 0.70,
                "台湾": 0.70,
                "export control": 0.88,
                "出口管制": 0.88,
                "cybersecurity": 0.68,
                "网络安全": 0.68,
                "power grid": 0.78,
                "电网": 0.78,
                "energy bottleneck": 0.92,
                "能源瓶颈": 0.92,
                "roi": 0.72,
                "回报": 0.72,
            },
        },
        "trade_credit": {
            "description": "Tariffs, trade diplomacy, credit stress and corporate cash-flow pressure",
            "assets": ["equities", "copper", "steel", "usd", "credit", "agriculture"],
            "terms": {
                "tariff": 0.90,
                "关税": 0.90,
                "trade war": 0.88,
                "贸易战": 0.88,
                "trade talks": 0.72,
                "贸易谈判": 0.72,
                "sanction": 0.68,
                "制裁": 0.68,
                "private credit": 0.82,
                "私人信贷": 0.82,
                "credit stress": 0.86,
                "信贷压力": 0.86,
                "bond panic": 0.90,
                "债券恐慌": 0.90,
                "yield spike": 0.86,
                "收益率飙升": 0.86,
                "refinancing": 0.70,
                "再融资": 0.70,
                "governance": 0.55,
                "治理": 0.55,
            },
        },
        "market_sentiment": {
            "description": "Risk appetite, crowding, valuation pressure and bubble concern",
            "assets": ["equities", "gold", "volatility", "usd"],
            "terms": {
                "risk-off": 0.84,
                "避险": 0.84,
                "risk-on": 0.65,
                "乐观": 0.65,
                "bubble": 0.80,
                "泡沫": 0.80,
                "valuation": 0.68,
                "估值": 0.68,
                "overheated": 0.76,
                "过热": 0.76,
                "volatility": 0.60,
                "波动": 0.60,
                "crowded": 0.62,
                "拥挤": 0.62,
            },
        },
    }

    def describe_capabilities(self) -> Dict[str, Any]:
        return {
            "news_ingestion": {
                "enabled": True,
                "accepted_inputs": ["list[dict]", "pandas.DataFrame", "policy calendar records"],
                "normalized_fields": ["timestamp", "title", "summary", "source", "relevance_score", "sentiment_score", "tags"],
            },
            "risk_models": {
                domain: {
                    "description": spec["description"],
                    "affected_assets": spec["assets"],
                    "term_count": len(spec["terms"]),
                }
                for domain, spec in self.DOMAIN_ONTOLOGY.items()
            },
            "event_driven_causal_chains": True,
            "event_windows": {
                "enabled": True,
                "default_window": "T-5_to_T+10",
                "fields": ["return", "volume", "volatility", "direction"],
            },
            "price_confirmation_learning": {
                "enabled": True,
                "memory_records": len(self.price_confirmation_memory),
                "contract": "historical confirmation reliability reweights sensitive asset votes",
            },
            "dynamic_game_dominance": [
                "gold_safe_haven_vs_real_rate_vs_inflation",
                "crude_geo_supply_vs_opec_vs_demand",
                "copper_ai_demand_vs_trade_drag_vs_growth",
                "ai_equity_growth_roi_vs_rate_valuation_vs_export_controls",
            ],
            "six_step_game_framework": {
                "enabled": True,
                "steps": [
                    "define_core_thesis_for_both_sides",
                    "identify_sensitive_pricing_assets",
                    "map_transmission_mechanisms",
                    "score_dominance_conditions",
                    "judge_short_term_and_medium_term_winners",
                    "translate_to_rates_equities_commodities_fx_risk_appetite",
                ],
                "relation_template_count": len(self._build_relation_specs()),
                "categories": [
                    "A_monetary_inflation_jobs_debt",
                    "B_energy_opec_geopolitics",
                    "C_ai_technology_data_center",
                    "D_earnings_valuation_cashflow",
                    "E_fx_gold_intervention",
                    "F_bonds_credit_financial_risk",
                    "G_trade_supply_chain_policy",
                    "H_consumption_jobs_growth",
                ],
            },
        }

    def analyze(
        self,
        news_items: Optional[Any] = None,
        policy_records: Optional[Any] = None,
        market_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete game-causal analysis layer."""
        events = self.ingest_news(news_items)
        events.extend(self.ingest_policy_records(policy_records))
        market_context = market_context or {}
        event_windows = self.build_event_windows(
            events,
            market_context.get("pricing_asset_panels", {}),
            pre_days=int(_to_float(market_context.get("event_pre_days"), 5)),
            post_days=int(_to_float(market_context.get("event_post_days"), 10)),
        )
        learned_records = self.learn_price_confirmation_weights(
            market_context.get("price_confirmation_history", [])
        )
        risk_scores = self.quantify_risks(events)
        event_chains = self.build_event_causal_chains(events, risk_scores)
        dominance = self.evaluate_game_dominance(risk_scores, market_context, event_chains)
        relation_reports = self.analyze_game_relations(
            risk_scores=risk_scores,
            market_context=market_context,
            event_chains=event_chains,
        )
        aggregate_score = self._aggregate_risk_score(risk_scores)
        return {
            "status": "active" if events else "no_events",
            "event_count": len(events),
            "events": [asdict(event) for event in events[:25]],
            "event_windows": [asdict(item) for item in event_windows],
            "price_confirmation_learning": {
                "updated_records": [asdict(item) for item in learned_records],
                "memory_size": len(self.price_confirmation_memory),
            },
            "risk_scores": risk_scores,
            "aggregate_risk_score": round(aggregate_score, 6),
            "event_causal_chains": [asdict(chain) for chain in event_chains],
            "dominant_game_logics": [asdict(item) for item in dominance],
            "game_relation_reports": [asdict(item) for item in relation_reports],
        }

    def analyze_game_relations(
        self,
        risk_scores: Mapping[str, Mapping[str, Any]],
        market_context: Mapping[str, Any],
        event_chains: Sequence[EventCausalChain],
    ) -> List[GameRelationReport]:
        """Apply the reusable X-vs-Y six-step framework to all relation specs."""
        reports: List[GameRelationReport] = []
        chain_ids = {chain.chain_id for chain in event_chains}
        for spec in self._build_relation_specs():
            side_a_price = self._score_price_confirmations(
                spec.side_a.price_confirmations,
                market_context,
                relation_id=spec.relation_id,
                side_id=spec.side_a.side_id,
            )
            side_b_price = self._score_price_confirmations(
                spec.side_b.price_confirmations,
                market_context,
                relation_id=spec.relation_id,
                side_id=spec.side_b.side_id,
            )
            side_a_score = self._score_relation_side(spec.side_a, risk_scores, market_context, side_a_price, chain_ids)
            side_b_score = self._score_relation_side(spec.side_b, risk_scores, market_context, side_b_price, chain_ids)
            winner, confidence, judgement = self._relation_judgement(spec, side_a_score, side_b_score)
            layer_winners = self._layer_winners(spec, risk_scores, market_context, side_a_score, side_b_score)
            identification = self._relation_identification_status(
                spec=spec,
                side_a_price=side_a_price,
                side_b_price=side_b_price,
                risk_scores=risk_scores,
                chain_ids=chain_ids,
            )
            reports.append(
                GameRelationReport(
                    relation_id=spec.relation_id,
                    relation_name=spec.relation_name,
                    category=spec.category,
                    core_logic={
                        "A": spec.side_a.core_thesis,
                        "B": spec.side_b.core_thesis,
                    },
                    sensitive_assets=spec.sensitive_assets,
                    transmission_mechanisms={
                        "A": spec.side_a.transmission_chain,
                        "B": spec.side_b.transmission_chain,
                    },
                    evolution_paths={
                        "A": spec.side_a.evolution_path,
                        "B": spec.side_b.evolution_path,
                    },
                    dominance_conditions={
                        "A": spec.side_a.dominance_conditions,
                        "B": spec.side_b.dominance_conditions,
                    },
                    price_confirmation={
                        "A": side_a_price,
                        "B": side_b_price,
                    },
                    side_scores={
                        "A": round(side_a_score, 6),
                        "B": round(side_b_score, 6),
                    },
                    current_judgement={
                        "winner": winner,
                        "confidence": confidence,
                        "summary": judgement,
                        "short_term": layer_winners.get("short_end", {}).get("winner", winner),
                        "medium_term": layer_winners.get("one_to_three_months", {}).get("winner", winner),
                    },
                    layer_winners=layer_winners,
                    market_pricing_forecast={
                        "if_A_wins": spec.side_a.market_pricing_if_wins,
                        "if_B_wins": spec.side_b.market_pricing_if_wins,
                        "current_prediction": self._current_relation_prediction(spec, winner),
                    },
                    key_reversal_signals=sorted(set(spec.side_a.reversal_signals + spec.side_b.reversal_signals)),
                    observation_checklist=spec.observation_checklist,
                    identification_status=identification,
                    actionability="trade_allowed" if identification.get("can_trade") else "observe_only",
                )
            )
        return reports

    def ingest_news(self, news_items: Optional[Any]) -> List[NewsEvent]:
        if news_items is None:
            return []
        records = self._records_from_any(news_items)
        return [self._normalize_event(record, fallback_source="news") for record in records]

    def ingest_policy_records(self, policy_records: Optional[Any]) -> List[NewsEvent]:
        records = self._records_from_any(policy_records)
        return [self._normalize_event(record, fallback_source="policy_calendar") for record in records]

    def build_event_windows(
        self,
        events: Sequence[NewsEvent],
        pricing_asset_panels: Optional[Mapping[str, Any]],
        pre_days: int = 5,
        post_days: int = 10,
    ) -> List[EventWindowSnapshot]:
        """Create T-pre to T+post asset reaction windows for event studies."""
        if not events or not isinstance(pricing_asset_panels, Mapping):
            return []
        snapshots: List[EventWindowSnapshot] = []
        for asset, raw_panel in pricing_asset_panels.items():
            frame = self._normalize_price_panel(raw_panel)
            if frame.empty:
                continue
            for event in events:
                event_time = pd.to_datetime(event.timestamp, errors="coerce")
                if pd.isna(event_time):
                    continue
                frame_before = frame[frame["date"] <= event_time]
                if frame_before.empty:
                    continue
                event_idx = int(frame_before.index[-1])
                pre_start = max(0, event_idx - pre_days)
                post_end = min(len(frame) - 1, event_idx + post_days)
                pre_slice = frame.iloc[pre_start : event_idx + 1]
                post_slice = frame.iloc[event_idx : post_end + 1]
                if pre_slice.empty or post_slice.empty:
                    continue
                pre_return = self._safe_return(pre_slice["close"].iloc[0], pre_slice["close"].iloc[-1])
                post_return = self._safe_return(post_slice["close"].iloc[0], post_slice["close"].iloc[-1])
                latest_return = self._safe_return(frame["close"].iloc[event_idx], frame["close"].iloc[-1])
                pre_vol = float(pre_slice["close"].pct_change().std() or 0.0)
                post_vol = float(post_slice["close"].pct_change().std() or 0.0)
                pre_volume = float(pre_slice.get("volume", pd.Series([1.0])).mean() or 1.0)
                post_volume = float(post_slice.get("volume", pd.Series([1.0])).mean() or 1.0)
                direction_basis = post_return if len(post_slice) > 1 else latest_return
                snapshots.append(
                    EventWindowSnapshot(
                        event_id=event.event_id,
                        asset=str(asset),
                        event_timestamp=event.timestamp,
                        pre_days=int(len(pre_slice) - 1),
                        post_days=int(len(post_slice) - 1),
                        pre_return=round(float(pre_return), 6),
                        post_return=round(float(post_return), 6),
                        event_to_latest_return=round(float(latest_return), 6),
                        volatility_ratio=round(float(post_vol / max(pre_vol, 1e-8)), 6),
                        volume_ratio=round(float(post_volume / max(pre_volume, 1e-8)), 6),
                        observed_direction="up" if direction_basis > 0 else ("down" if direction_basis < 0 else "flat"),
                        usable_for_learning=bool(len(post_slice) - 1 >= max(2, post_days // 3)),
                    )
                )
        return snapshots

    def learn_price_confirmation_weights(
        self,
        history: Optional[Any],
    ) -> List[PriceConfirmationMemoryRecord]:
        """Learn which sensitive assets reliably confirm each X-vs-Y side."""
        records = self._records_from_any(history)
        grouped: Dict[Tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
        for record in records:
            relation_id = str(record.get("relation_id", "")).strip()
            side_id = str(record.get("side_id", record.get("side", ""))).strip()
            asset = str(record.get("asset", "")).strip()
            expected_direction = str(record.get("expected_direction", "")).strip()
            if not relation_id or not side_id or not asset or not expected_direction:
                continue
            grouped.setdefault((relation_id, side_id, asset, expected_direction), []).append(record)

        updated: List[PriceConfirmationMemoryRecord] = []
        for (relation_id, side_id, asset, expected_direction), rows in grouped.items():
            reliabilities: List[float] = []
            lead_lags: List[float] = []
            for row in rows:
                observed = {
                    "direction": row.get("observed_direction", row.get("direction", "")),
                    "return": row.get("realized_return", row.get("event_return", row.get("post_return", 0.0))),
                }
                confirmation_score = self._direction_confirmation_score(observed, expected_direction)
                outcome = str(row.get("outcome", "")).lower()
                if outcome in {"success", "win", "true", "1"}:
                    outcome_score = 1.0
                elif outcome in {"failure", "loss", "false", "0"}:
                    outcome_score = 0.0
                else:
                    outcome_score = confirmation_score
                reliabilities.append(0.65 * confirmation_score + 0.35 * outcome_score)
                lead_lags.append(_to_float(row.get("lead_lag_days", row.get("lag_days", 0.0)), 0.0))
            reliability = float(sum(reliabilities) / max(len(reliabilities), 1))
            learned_weight = float(_clip(0.45 + reliability * 1.35, 0.20, 1.80))
            memory = PriceConfirmationMemoryRecord(
                relation_id=relation_id,
                side_id=side_id,
                asset=asset,
                expected_direction=expected_direction,
                learned_weight=round(learned_weight, 6),
                reliability=round(reliability, 6),
                sample_count=len(rows),
                avg_lead_lag_days=round(float(sum(lead_lags) / max(len(lead_lags), 1)), 6),
                last_updated=datetime.utcnow().isoformat(),
            )
            self.price_confirmation_memory[self._memory_key(relation_id, side_id, asset, expected_direction)] = memory
            updated.append(memory)
        return updated

    def quantify_risks(self, events: Sequence[NewsEvent]) -> Dict[str, Dict[str, Any]]:
        scores: Dict[str, Dict[str, Any]] = {}
        for domain, spec in self.DOMAIN_ONTOLOGY.items():
            event_hits: List[Tuple[NewsEvent, float, List[str]]] = []
            evidence_terms: List[str] = []
            for event in events:
                hit_score, terms = self._score_event_against_domain(event, spec["terms"])
                if hit_score <= 0:
                    continue
                event_hits.append((event, hit_score, terms))
                evidence_terms.extend(terms)

            if event_hits:
                weighted_hit = sum(
                    hit * _clip(event.relevance_score, 0.1, 1.5) * (1.0 + min(abs(event.sentiment_score), 1.0) * 0.10)
                    for event, hit, _ in event_hits
                )
                event_count_component = min(len(event_hits), 5) / 5.0
                max_hit = max(hit for _, hit, _ in event_hits)
                score = _clip(0.62 * min(weighted_hit / 2.2, 1.0) + 0.25 * max_hit + 0.13 * event_count_component)
            else:
                score = 0.0

            scores[domain] = {
                "score": round(score, 6),
                "event_count": len(event_hits),
                "matched_terms": sorted(set(evidence_terms))[:20],
                "affected_assets": list(spec["assets"]),
                "evidence_event_ids": [event.event_id for event, _, _ in event_hits[:10]],
                "description": spec["description"],
            }
        return scores

    def build_event_causal_chains(
        self,
        events: Sequence[NewsEvent],
        risk_scores: Mapping[str, Mapping[str, Any]],
    ) -> List[EventCausalChain]:
        chains: List[EventCausalChain] = []
        if risk_scores.get("geopolitical_energy", {}).get("score", 0.0) >= 0.20:
            event_ids = list(risk_scores["geopolitical_energy"].get("evidence_event_ids", []))
            chains.append(
                EventCausalChain(
                    chain_id="geo_energy_supply_shock",
                    trigger_event_ids=event_ids,
                    trigger="geopolitical_energy_event",
                    causal_steps=[
                        "conflict_or_cartel_instability",
                        "transport_route_or_output_risk",
                        "energy_supply_risk_premium",
                        "inflation_and_safe_haven_repricing",
                    ],
                    affected_assets=["crude_oil", "natural_gas", "gold", "shipping", "energy_equities"],
                    expected_direction="bullish_energy_bullish_gold_risk_off_equities",
                    confidence=self._chain_confidence("geopolitical_energy", risk_scores),
                )
            )
        if risk_scores.get("ai_technology", {}).get("score", 0.0) >= 0.20:
            event_ids = list(risk_scores["ai_technology"].get("evidence_event_ids", []))
            chains.append(
                EventCausalChain(
                    chain_id="ai_capex_power_materials_chain",
                    trigger_event_ids=event_ids,
                    trigger="ai_capex_or_technology_restriction_event",
                    causal_steps=[
                        "ai_compute_demand_changes",
                        "semiconductor_and_data_center_capex_repricing",
                        "power_grid_and_copper_demand_expectation",
                        "valuation_or_export_control_constraint",
                    ],
                    affected_assets=["ai_equities", "semiconductors", "copper", "power_infrastructure"],
                    expected_direction="conditional_bullish_ai_infra_and_copper_unless_controls_dominate",
                    confidence=self._chain_confidence("ai_technology", risk_scores),
                )
            )
        if risk_scores.get("monetary_inflation", {}).get("score", 0.0) >= 0.20:
            event_ids = list(risk_scores["monetary_inflation"].get("evidence_event_ids", []))
            chains.append(
                EventCausalChain(
                    chain_id="policy_space_real_rate_chain",
                    trigger_event_ids=event_ids,
                    trigger="monetary_policy_or_inflation_event",
                    causal_steps=[
                        "inflation_or_jobs_signal",
                        "central_bank_reaction_function",
                        "real_rate_and_liquidity_path",
                        "duration_asset_and_gold_repricing",
                    ],
                    affected_assets=["gold", "duration_equities", "usd", "bonds"],
                    expected_direction="hawkish_policy_bearish_duration_mixed_gold",
                    confidence=self._chain_confidence("monetary_inflation", risk_scores),
                )
            )
        if risk_scores.get("trade_credit", {}).get("score", 0.0) >= 0.20:
            event_ids = list(risk_scores["trade_credit"].get("evidence_event_ids", []))
            chains.append(
                EventCausalChain(
                    chain_id="tariff_credit_cashflow_chain",
                    trigger_event_ids=event_ids,
                    trigger="trade_policy_or_credit_stress_event",
                    causal_steps=[
                        "tariff_or_credit_stress_shock",
                        "corporate_margin_and_refinancing_pressure",
                        "cyclical_demand_and_risk_premium_repricing",
                    ],
                    affected_assets=["equities", "copper", "steel", "credit", "agriculture"],
                    expected_direction="bearish_cyclicals_bullish_defensive_hedges",
                    confidence=self._chain_confidence("trade_credit", risk_scores),
                )
            )
        return chains

    def evaluate_game_dominance(
        self,
        risk_scores: Mapping[str, Mapping[str, Any]],
        market_context: Mapping[str, Any],
        event_chains: Sequence[EventCausalChain],
    ) -> List[GameDominance]:
        geo = self._risk(risk_scores, "geopolitical_energy")
        monetary = self._risk(risk_scores, "monetary_inflation")
        ai = self._risk(risk_scores, "ai_technology")
        trade = self._risk(risk_scores, "trade_credit")
        sentiment = self._risk(risk_scores, "market_sentiment")
        growth = self._context_value(market_context, ["growth", "cross_asset_regime.growth"], 0.02)
        inflation = self._context_value(market_context, ["inflation", "cross_asset_regime.inflation"], 0.025)
        liquidity = self._context_value(market_context, ["liquidity", "cross_asset_regime.liquidity"], 0.01)
        ai_capex_growth = self._context_value(market_context, ["ai_capex_growth", "AI_DataCenter_Capex.growth"], 0.12)
        copper_inventory_days = self._context_value(market_context, ["copper_inventory_days", "LME_Inventory_Days.value"], 5.0)
        valuation_pressure = _clip(sentiment * 0.45 + monetary * 0.35 + max(-liquidity, 0.0) * 3.0)

        gold_forces = [
            GameForce(
                name="safe_haven_logic",
                score=_clip(0.68 * geo + 0.18 * trade + 0.14 * sentiment),
                expected_direction="bullish_gold",
                conditions=["geopolitical risk rises", "risk-off sentiment dominates"],
                evidence_terms=self._terms(risk_scores, ["geopolitical_energy", "market_sentiment"]),
            ),
            GameForce(
                name="inflation_hedge_logic",
                score=_clip(0.42 * monetary + 5.0 * max(inflation - 0.025, 0.0) + 0.12 * trade),
                expected_direction="bullish_gold",
                conditions=["inflation shock persists", "policy credibility is questioned"],
                evidence_terms=self._terms(risk_scores, ["monetary_inflation", "trade_credit"]),
            ),
            GameForce(
                name="real_rate_pressure_logic",
                score=_clip(0.52 * monetary + 2.8 * max(-liquidity, 0.0) + 0.20 * valuation_pressure),
                expected_direction="bearish_gold",
                conditions=["hawkish policy path dominates", "real rates or dollar pressure rise"],
                evidence_terms=self._terms(risk_scores, ["monetary_inflation"]),
            ),
        ]

        crude_forces = [
            GameForce(
                name="geopolitical_supply_shock",
                score=_clip(0.82 * geo + 0.10 * sentiment),
                expected_direction="bullish_crude",
                conditions=["route security risk", "war or sanctions threaten supply"],
                evidence_terms=self._terms(risk_scores, ["geopolitical_energy"]),
            ),
            GameForce(
                name="opec_supply_control",
                score=_clip(0.55 * geo + 0.18 * monetary),
                expected_direction="bullish_crude_if_cuts_bearish_if_breakdown",
                conditions=["cartel discipline or production-policy surprise"],
                evidence_terms=self._terms(risk_scores, ["geopolitical_energy"]),
            ),
            GameForce(
                name="demand_slowdown",
                score=_clip(0.38 * trade + 0.28 * monetary + 1.8 * max(0.02 - growth, 0.0)),
                expected_direction="bearish_crude",
                conditions=["growth slows", "tariffs or credit stress reduce demand"],
                evidence_terms=self._terms(risk_scores, ["trade_credit", "monetary_inflation"]),
            ),
        ]

        copper_forces = [
            GameForce(
                name="ai_power_grid_demand",
                score=_clip(0.46 * ai + 1.7 * max(ai_capex_growth - 0.10, 0.0) + 0.12 * max(5.0 - copper_inventory_days, 0.0)),
                expected_direction="bullish_copper",
                conditions=["AI capex remains strong", "power-grid and data-center bottlenecks bind"],
                evidence_terms=self._terms(risk_scores, ["ai_technology"]),
            ),
            GameForce(
                name="trade_tariff_drag",
                score=_clip(0.64 * trade + 0.20 * geo),
                expected_direction="bearish_copper",
                conditions=["tariffs or export restrictions hit cyclicals", "global trade uncertainty rises"],
                evidence_terms=self._terms(risk_scores, ["trade_credit", "geopolitical_energy"]),
            ),
            GameForce(
                name="china_growth_liquidity",
                score=_clip(1.8 * max(growth - 0.03, 0.0) + 1.2 * max(liquidity, 0.0)),
                expected_direction="bullish_copper",
                conditions=["growth impulse improves", "liquidity supports restocking"],
                evidence_terms=[],
            ),
        ]

        ai_equity_forces = [
            GameForce(
                name="ai_growth_roi_logic",
                score=_clip(0.52 * ai + 1.4 * max(ai_capex_growth - 0.10, 0.0) + 0.20 * max(growth, 0.0)),
                expected_direction="bullish_ai_infra",
                conditions=["AI capex converts into revenue", "data-center infrastructure remains scarce"],
                evidence_terms=self._terms(risk_scores, ["ai_technology"]),
            ),
            GameForce(
                name="rate_valuation_pressure",
                score=_clip(0.45 * monetary + 0.40 * valuation_pressure + 1.5 * max(-liquidity, 0.0)),
                expected_direction="bearish_high_duration_tech",
                conditions=["real-rate path tightens", "valuation or crowding concern dominates"],
                evidence_terms=self._terms(risk_scores, ["monetary_inflation", "market_sentiment"]),
            ),
            GameForce(
                name="export_control_geopolitical_drag",
                score=_clip(0.44 * ai + 0.35 * trade + 0.25 * geo),
                expected_direction="bearish_semiconductor_supply_chain",
                conditions=["export control or Taiwan risk affects supply chain"],
                evidence_terms=self._terms(risk_scores, ["ai_technology", "trade_credit", "geopolitical_energy"]),
            ),
        ]

        return [
            self._decide_game("gold", gold_forces),
            self._decide_game("crude_oil", crude_forces),
            self._decide_game("copper", copper_forces),
            self._decide_game("ai_equities", ai_equity_forces),
        ]

    def _build_relation_specs(self) -> List[GameRelationSpec]:
        def p(asset: str, direction: str, weight: float = 1.0, rationale: str = "") -> PricingAssetRule:
            return PricingAssetRule(asset=asset, expected_direction=direction, weight=weight, rationale=rationale)

        def r(path: str, operator: str, threshold: float, weight: float, rationale: str) -> ContextRule:
            return ContextRule(path=path, operator=operator, threshold=threshold, weight=weight, rationale=rationale)

        def side(
            side_id: str,
            name: str,
            thesis: str,
            chain: List[str],
            path: List[str],
            conditions: List[str],
            pricing: Dict[str, str],
            reversals: List[str],
            domains: Dict[str, float],
            context: Optional[List[ContextRule]] = None,
            prices: Optional[List[PricingAssetRule]] = None,
        ) -> GameSideSpec:
            return GameSideSpec(
                side_id=side_id,
                name=name,
                core_thesis=thesis,
                transmission_chain=chain,
                evolution_path=path,
                dominance_conditions=conditions,
                market_pricing_if_wins=pricing,
                reversal_signals=reversals,
                domain_weights=domains,
                context_rules=context or [],
                price_confirmations=prices or [],
            )

        return [
            GameRelationSpec(
                relation_id="fed_inflation_vs_policy_space",
                relation_name="Fed hawkish inflation pressure vs fiscal policy-space constraint",
                category="A_monetary_inflation_jobs_debt",
                side_a=side(
                    "A",
                    "hawkish_inflation_pressure",
                    "Inflation pressure and policy credibility force the Fed to price a higher policy-rate path.",
                    ["inflation_or_wage_pressure", "cut_expectations_fall_or_hike_risk_rises", "2Y_yield_and_usd_reprice", "duration_equities_de-rate"],
                    ["hot_data_prints", "front-end_rates_sell_off", "Fed_speakers_validate_higher_for_longer", "higher_rates_reinforce_valuation_pressure"],
                    ["CPI/PCE or wages are sticky", "Fed communication validates hawkish repricing", "2Y yield and USD rise", "gold weakens unless credibility risk offsets"],
                    {"rates": "2Y yields higher", "equities": "long-duration growth underperforms", "commodities": "gold pressured by real rates", "fx": "USD stronger", "risk_appetite": "lower"},
                    ["2Y yield stops rising after hot data", "Fed validates cuts despite sticky inflation", "USD fails to rally"],
                    {"monetary_inflation": 1.0, "market_sentiment": 0.2},
                    [r("inflation", ">", 0.03, 1.0, "inflation is above comfort zone"), r("liquidity", "<", 0.0, 0.6, "tight liquidity amplifies hawkish pricing")],
                    [p("US2Y", "up", 1.2), p("fed_funds_futures", "hawkish", 1.0), p("DXY", "up", 0.9), p("gold", "down", 0.6)],
                ),
                side_b=side(
                    "B",
                    "fiscal_policy_space_constraint",
                    "High debt, fiscal fragility and market stress constrain how far policy can tighten.",
                    ["debt_or_growth_fragility", "Fed_reaction_function_gets_constrained", "real_rate_upside_is_capped", "term_premium_or_gold_prices_fiscal_risk"],
                    ["debt_constraint_becomes_visible", "long-end_or_gold_decouples_from_front-end", "risk_assets_resist_front-end_hawkishness", "credibility_discount_becomes_dominant"],
                    ["10Y/30Y and gold rise together", "USD fails to benefit from higher yields", "credit or bank stress appears", "markets price fiscal dominance"],
                    {"rates": "long-end term premium higher but front-end capped", "equities": "banks and leveraged assets vulnerable", "commodities": "gold supported", "fx": "USD credibility mixed to weaker", "risk_appetite": "fragile"},
                    ["long-end yields fall with lower inflation expectations", "gold fails to confirm fiscal concern", "credit spreads remain tight"],
                    {"monetary_inflation": 0.6, "trade_credit": 0.3, "market_sentiment": 0.4},
                    [r("growth", "<", 0.02, 0.8, "slow growth constrains tightening"), r("liquidity", "<", 0.0, 0.8, "market fragility constrains policy")],
                    [p("US10Y", "up", 0.9), p("US30Y", "up", 1.1), p("term_premium", "up", 1.0), p("gold", "up", 0.9), p("bank_stocks", "down", 0.7)],
                ),
                sensitive_assets=["US2Y", "fed_funds_futures", "DXY", "gold", "US10Y", "US30Y", "term_premium", "bank_stocks", "credit_spreads"],
                observation_checklist=["CPI/PCE", "nonfarm payrolls and wages", "2Y vs 30Y curve", "Fed communication", "gold vs real rates", "credit spreads"],
                layer_assets={
                    "short_end": ["US2Y", "fed_funds_futures", "DXY"],
                    "long_end": ["US10Y", "US30Y", "term_premium", "gold"],
                    "risk_assets": ["duration_equities", "bank_stocks", "credit_spreads"],
                    "commodities": ["gold"],
                    "fx": ["DXY"],
                    "one_to_three_months": ["US10Y", "US30Y", "credit_spreads", "gold"],
                },
            ),
            GameRelationSpec(
                relation_id="geopolitical_risk_vs_risk_appetite",
                relation_name="Geopolitical risk premium vs market risk appetite",
                category="B_energy_opec_geopolitics",
                side_a=side(
                    "A",
                    "geopolitical_risk_premium",
                    "War, sanctions or transport-route risk raise the required risk premium across energy, gold and volatility.",
                    ["geopolitical_event", "supply_or_tail_risk_expectations_rise", "hedging_demand_and_risk_premium_expand", "oil_gold_vix_reprice"],
                    ["conflict_headline_hits", "sensitive_assets_confirm", "hedging_flows_expand", "higher_volatility_forces_de-risking"],
                    ["oil rises", "gold rises", "VIX or credit spreads widen", "shipping or insurance costs rise"],
                    {"rates": "inflation breakevens can rise", "equities": "cyclicals and airlines lag", "commodities": "oil and gold higher", "fx": "USD and havens stronger", "risk_appetite": "risk-off"},
                    ["oil, gold and VIX fail to confirm", "ceasefire becomes credible", "credit spreads stay tight"],
                    {"geopolitical_energy": 1.0, "market_sentiment": 0.4},
                    [r("liquidity", "<", 0.0, 0.4, "tight liquidity magnifies shock")],
                    [p("Brent", "up", 1.2), p("WTI", "up", 1.2), p("gold", "up", 1.0), p("VIX", "up", 1.0), p("credit_spreads", "widen", 0.8)],
                ),
                side_b=side(
                    "B",
                    "risk_appetite_looks_through_tail_risk",
                    "If liquidity, earnings and carry remain strong, markets ignore tail risk until sensitive assets confirm stress.",
                    ["headline_risk_appears", "oil_gold_vix_do_not_confirm", "equity_and_credit_carry_persist", "risk_assets_absorb_the_shock"],
                    ["bad_news_is_faded", "equities_reclaim_highs", "volatility_compresses", "tail_risk_gets_repriced_lower"],
                    ["equities make new highs", "VIX falls", "credit spreads are stable or tighter", "oil does not rise despite headlines"],
                    {"rates": "little safe-haven bid", "equities": "growth and cyclicals resilient", "commodities": "oil stable to lower", "fx": "high-beta FX resilient", "risk_appetite": "risk-on"},
                    ["oil breaks higher", "gold and VIX rise together", "shipping disruption becomes observable"],
                    {"market_sentiment": 0.7, "geopolitical_energy": -0.2},
                    [r("liquidity", ">", 0.01, 0.8, "liquidity supports risk appetite"), r("growth", ">", 0.025, 0.6, "growth offsets geopolitical risk")],
                    [p("Nasdaq", "up", 1.0), p("VIX", "down", 1.1), p("credit_spreads", "narrow", 1.0), p("Brent", "down", 0.6), p("gold", "down", 0.5)],
                ),
                sensitive_assets=["Brent", "WTI", "gold", "VIX", "credit_spreads", "Nasdaq", "shipping_rates", "energy_equities"],
                observation_checklist=["Brent/WTI spot and curve", "gold", "VIX", "credit spreads", "shipping and insurance costs", "energy vs airline equities"],
                layer_assets={
                    "short_end": ["VIX", "gold"],
                    "long_end": ["credit_spreads"],
                    "risk_assets": ["Nasdaq", "credit_spreads", "VIX"],
                    "commodities": ["Brent", "WTI", "gold"],
                    "fx": ["DXY", "em_fx"],
                    "one_to_three_months": ["Brent", "credit_spreads", "shipping_rates"],
                },
            ),
            GameRelationSpec(
                relation_id="opec_supply_vs_price_control",
                relation_name="OPEC+ supply increase vs cartel price control",
                category="B_energy_opec_geopolitics",
                side_a=side(
                    "A",
                    "supply_increase_logic",
                    "OPEC+ output increases shift expectations toward higher inventories and lower oil prices.",
                    ["opec_output_signal", "supply_expectations_rise", "inventory_build_or_curve_softens", "oil_price_falls"],
                    ["production_headline", "front_month_oil_sells_off", "inventory_data_confirms", "energy_equities_lag"],
                    ["announced increase is large", "inventories rise", "oil curve moves toward contango", "energy equities underperform"],
                    {"rates": "lower inflation pressure", "equities": "airlines/chemicals benefit", "commodities": "oil lower", "fx": "oil exporters weaker", "risk_appetite": "mixed"},
                    ["backwardation strengthens", "inventories draw", "geopolitical risk overwhelms supply increase"],
                    {"geopolitical_energy": 0.3, "trade_credit": 0.2},
                    [r("growth", "<", 0.02, 0.5, "weak demand makes extra supply bearish")],
                    [p("Brent", "down", 1.1), p("WTI", "down", 1.1), p("oil_inventories", "up", 1.0), p("oil_curve_backwardation", "down", 0.9)],
                ),
                side_b=side(
                    "B",
                    "price_control_logic",
                    "Limited increases, resilient demand or geopolitical risk keep supply effectively controlled.",
                    ["limited_output_or_geo_risk", "market_sees_supply_still_tight", "curve_or_energy_equities_confirm", "oil_stabilizes_or_rises"],
                    ["headline_is_absorbed", "curve_stays_tight", "energy_equities_hold", "shorts_cover"],
                    ["output increase is small", "demand remains firm", "backwardation holds", "energy equities confirm oil strength"],
                    {"rates": "inflation risk persists", "equities": "energy outperforms", "commodities": "oil stable to higher", "fx": "oil exporters supported", "risk_appetite": "supply-shock sensitive"},
                    ["inventories build persistently", "curve flips to contango", "energy equities fail to confirm"],
                    {"geopolitical_energy": 0.8},
                    [r("growth", ">", 0.025, 0.5, "demand supports price control")],
                    [p("Brent", "up", 1.0), p("WTI", "up", 1.0), p("oil_curve_backwardation", "up", 1.1), p("energy_equities", "up", 0.9)],
                ),
                sensitive_assets=["Brent", "WTI", "oil_inventories", "oil_curve_backwardation", "energy_equities", "airlines"],
                observation_checklist=["OPEC quota details", "inventory reports", "Brent/WTI curve", "energy equities", "airlines and chemicals"],
                layer_assets={
                    "short_end": ["Brent", "WTI"],
                    "long_end": ["oil_curve_backwardation", "oil_inventories"],
                    "risk_assets": ["energy_equities", "airlines"],
                    "commodities": ["Brent", "WTI", "oil_inventories"],
                    "fx": ["oil_exporter_fx"],
                    "one_to_three_months": ["oil_curve_backwardation", "energy_equities"],
                },
            ),
            GameRelationSpec(
                relation_id="ai_roi_vs_valuation_bubble",
                relation_name="AI investment ROI vs technology valuation bubble",
                category="C_ai_technology_data_center",
                side_a=side(
                    "A",
                    "ai_roi_supercycle",
                    "AI usage lifts revenue, margins and durable moats enough to justify higher technology valuations.",
                    ["ai_usage_growth", "cloud_chip_and_software_revenue_upgrades", "earnings_expectations_rise", "valuation_is_supported_by_profit_revision"],
                    ["capex_signal", "order_and_revenue_confirmation", "earnings_revisions_follow", "momentum_reinforces_ai_leadership"],
                    ["cloud revenue accelerates", "chip orders remain strong", "earnings revisions rise", "AI leaders outperform broad tech"],
                    {"rates": "less rate-sensitive if earnings revisions dominate", "equities": "AI infrastructure and quality growth lead", "commodities": "copper/power demand supported", "fx": "USD may benefit from US tech leadership", "risk_appetite": "risk-on but narrow"},
                    ["AI revenue fails to follow capex", "FCF estimates fall", "SOX leadership breaks"],
                    {"ai_technology": 1.0, "market_sentiment": 0.2},
                    [r("ai_capex_growth", ">", 0.18, 0.8, "capex confirms AI buildout"), r("growth", ">", 0.025, 0.4, "macro growth supports ROI")],
                    [p("Nasdaq", "up", 0.9), p("SOX", "up", 1.2), p("cloud_revenue_revision", "up", 1.0), p("ai_fcf_revision", "up", 0.9), p("copper", "up", 0.4)],
                ),
                side_b=side(
                    "B",
                    "valuation_bubble_pressure",
                    "AI capex and expectations are priced too far ahead of realized cash flow, making high-multiple tech vulnerable.",
                    ["capex_accelerates", "free_cash_flow_or_roi_lags", "market_questions_payback_period", "high_multiple_tech_de_rates"],
                    ["capex_up_but_roi_unclear", "FCF_revisions_fall", "multiple_compression_begins", "style_rotation_weakens_AI_leadership"],
                    ["capex rises faster than revenue", "FCF revisions are negative", "software lags chips", "real rates rise"],
                    {"rates": "higher real rates hurt duration", "equities": "high-multiple tech underperforms", "commodities": "AI-linked demand less certain", "fx": "USD mixed", "risk_appetite": "de-risks crowded growth"},
                    ["FCF revisions turn up", "software monetization accelerates", "SOX/Nasdaq regain leadership on earnings"],
                    {"market_sentiment": 0.8, "monetary_inflation": 0.5, "ai_technology": 0.3},
                    [r("liquidity", "<", 0.0, 0.7, "tight liquidity punishes valuation"), r("ai_capex_growth", ">", 0.25, 0.4, "very high capex can raise ROI concern")],
                    [p("Nasdaq", "down", 1.0), p("SOX", "down", 0.8), p("unprofitable_software", "down", 1.0), p("ai_fcf_revision", "down", 1.1), p("VIX", "up", 0.6)],
                ),
                sensitive_assets=["Nasdaq", "SOX", "cloud_revenue_revision", "ai_fcf_revision", "unprofitable_software", "copper", "VIX"],
                observation_checklist=["AI capex", "GPU orders", "cloud revenue", "software monetization", "FCF revisions", "SOX vs software breadth"],
                layer_assets={
                    "short_end": ["SOX", "Nasdaq"],
                    "long_end": ["ai_fcf_revision", "cloud_revenue_revision"],
                    "risk_assets": ["Nasdaq", "SOX", "unprofitable_software", "VIX"],
                    "commodities": ["copper", "power_prices"],
                    "fx": ["DXY"],
                    "one_to_three_months": ["ai_fcf_revision", "cloud_revenue_revision", "SOX"],
                },
            ),
            GameRelationSpec(
                relation_id="ai_capex_vs_power_bottleneck",
                relation_name="AI infrastructure demand vs power and data-center bottleneck",
                category="C_ai_technology_data_center",
                side_a=side(
                    "A",
                    "ai_infrastructure_demand",
                    "AI capex is real demand for chips, copper, data centers and grid equipment.",
                    ["ai_model_demand", "data_center_buildout", "power_and_copper_orders", "infrastructure_assets_reprice"],
                    ["AI demand surprise", "grid and data-center orders rise", "industrial suppliers outperform", "price action validates scarcity"],
                    ["capex and backlog rise", "power equipment leads", "copper inventories tighten", "data-center utilization high"],
                    {"rates": "capex cycle supports growth", "equities": "power/grid/data-center chain outperforms", "commodities": "copper supported", "fx": "commodity FX supported", "risk_appetite": "selective risk-on"},
                    ["orders slow", "copper fails to confirm", "utilities/grid bottlenecks halt deployment"],
                    {"ai_technology": 0.9},
                    [r("ai_capex_growth", ">", 0.2, 1.0, "AI capex confirms infrastructure demand"), r("copper_inventory_days", "<", 4.0, 0.7, "low inventory tightens metal link")],
                    [p("copper", "up", 1.1), p("power_equipment", "up", 1.0), p("data_center_reits", "up", 0.8), p("utilities", "up", 0.5)],
                ),
                side_b=side(
                    "B",
                    "bottleneck_cost_constraint",
                    "Power, permitting and data-center constraints slow AI deployment and raise costs before revenue catches up.",
                    ["grid_or_permitting_constraint", "deployment_delays_and_cost_inflation", "ROI_payback_extends", "AI_multiple_or_margin_expectations_fall"],
                    ["bottleneck_news_accumulates", "costs_rise", "deployment_guidance_slows", "AI_winners_narrow_or_correct"],
                    ["power prices rise faster than revenue", "data-center delays increase", "margin/FCF revisions fall", "AI stocks diverge from infrastructure"],
                    {"rates": "inflationary capex friction", "equities": "AI application names lag infrastructure", "commodities": "copper may rise but tech margins pressured", "fx": "neutral", "risk_appetite": "more selective"},
                    ["grid orders scale quickly", "power costs stabilize", "AI revenue revisions offset cost pressure"],
                    {"ai_technology": 0.5, "market_sentiment": 0.5},
                    [r("ai_capex_growth", ">", 0.25, 0.4, "high capex can become cost pressure")],
                    [p("power_prices", "up", 1.0), p("ai_margin_revision", "down", 1.1), p("data_center_delay_index", "up", 1.0), p("Nasdaq", "down", 0.5)],
                ),
                sensitive_assets=["copper", "power_equipment", "data_center_reits", "utilities", "power_prices", "ai_margin_revision", "data_center_delay_index"],
                observation_checklist=["AI capex", "data-center vacancy and utilization", "grid backlog", "power prices", "copper inventory", "AI margin revisions"],
                layer_assets={
                    "short_end": ["power_equipment", "copper"],
                    "long_end": ["ai_margin_revision", "data_center_delay_index"],
                    "risk_assets": ["Nasdaq", "power_equipment", "data_center_reits"],
                    "commodities": ["copper", "power_prices"],
                    "fx": ["commodity_fx"],
                    "one_to_three_months": ["ai_margin_revision", "data_center_delay_index", "power_equipment"],
                },
            ),
            GameRelationSpec(
                relation_id="earnings_vs_valuation_pressure",
                relation_name="Corporate earnings strength vs valuation pressure",
                category="D_earnings_valuation_cashflow",
                side_a=side(
                    "A",
                    "earnings_revision_logic",
                    "Earnings, margins and cash flow improve faster than valuation multiples compress.",
                    ["earnings_surprise", "EPS_and_margin_revisions_up", "buybacks_or_cashflow_support", "equity_prices_hold_or_rise"],
                    ["earnings_beats", "analysts_revise_up", "cash_returns_follow", "price_momentum_reinforces_quality"],
                    ["EPS revisions positive", "margins improve", "FCF and buybacks rise", "sector breadth improves"],
                    {"rates": "less important if EPS dominates", "equities": "quality and profitable growth outperform", "commodities": "demand-sensitive commodities supported", "fx": "risk FX supported", "risk_appetite": "higher"},
                    ["earnings beats stop producing price gains", "margin guidance weakens", "multiple compression accelerates"],
                    {"market_sentiment": 0.3},
                    [r("growth", ">", 0.025, 0.5, "growth helps earnings")],
                    [p("eps_revision", "up", 1.2), p("profit_margin_revision", "up", 1.0), p("buyback_index", "up", 0.8), p("equities", "up", 0.8)],
                ),
                side_b=side(
                    "B",
                    "valuation_pressure_logic",
                    "Even good earnings are not enough when multiples, policy uncertainty or discount rates pressure valuations.",
                    ["valuation_is_high", "discount_rate_or_policy_risk_rises", "price_fails_to_reward_earnings", "multiple_compression_dominates"],
                    ["good_news_gets_sold", "multiples_contract", "breadth_weakens", "investors_rotate_to_value_or_cash"],
                    ["P/E falls despite EPS beats", "real rates rise", "policy uncertainty rises", "breadth narrows"],
                    {"rates": "higher discount rates matter", "equities": "expensive growth lags", "commodities": "mixed", "fx": "USD can strengthen", "risk_appetite": "lower"},
                    ["multiples stabilize", "EPS revisions accelerate", "policy uncertainty falls"],
                    {"market_sentiment": 0.7, "monetary_inflation": 0.4, "trade_credit": 0.3},
                    [r("liquidity", "<", 0.0, 0.6, "liquidity tightens valuation tolerance")],
                    [p("pe_multiple", "down", 1.2), p("equities", "down", 0.8), p("real_rates", "up", 0.9), p("VIX", "up", 0.5)],
                ),
                sensitive_assets=["eps_revision", "profit_margin_revision", "buyback_index", "pe_multiple", "real_rates", "equities", "VIX"],
                observation_checklist=["EPS revisions", "margin guidance", "FCF", "buybacks", "P/E multiples", "real rates", "sector breadth"],
                layer_assets={
                    "short_end": ["equities", "VIX"],
                    "long_end": ["eps_revision", "pe_multiple", "real_rates"],
                    "risk_assets": ["equities", "VIX", "pe_multiple"],
                    "commodities": ["cyclical_commodities"],
                    "fx": ["DXY"],
                    "one_to_three_months": ["eps_revision", "profit_margin_revision", "pe_multiple"],
                },
            ),
            GameRelationSpec(
                relation_id="dollar_rates_vs_intervention_safehaven",
                relation_name="Dollar rate-differential logic vs intervention/safe-haven logic",
                category="E_fx_gold_intervention",
                side_a=side(
                    "A",
                    "rate_differential_dollar_strength",
                    "Higher US real yields and rate differentials support USD against non-US currencies.",
                    ["front_end_or_real_yields_rise", "carry_moves_in_favor_of_USD", "capital_flows_support_dollar", "non_US_FX_weakens"],
                    ["yield_gap_widens", "USD_breaks_higher", "imported_pressure_for_non_US_central_banks", "carry_trend_reinforces"],
                    ["US real rates rise", "DXY rises", "USDJPY rises", "gold weakens"],
                    {"rates": "US yields higher", "equities": "EM and duration assets pressured", "commodities": "gold pressured unless risk-off", "fx": "USD stronger", "risk_appetite": "mixed to lower"},
                    ["USD fails to rise with yields", "intervention headlines trigger reversal", "gold rises with USD"],
                    {"monetary_inflation": 0.8},
                    [r("liquidity", "<", 0.0, 0.5, "tight USD liquidity helps dollar")],
                    [p("DXY", "up", 1.1), p("USDJPY", "up", 1.0), p("real_rates", "up", 0.9), p("gold", "down", 0.6)],
                ),
                side_b=side(
                    "B",
                    "intervention_or_safehaven_override",
                    "FX intervention, fiscal credibility concerns or systemic risk can override simple rate differentials.",
                    ["fx_level_or_systemic_risk_extreme", "policy_intervention_or_safe_haven_flows", "carry_trade_unwinds", "USD_or_cross_FX_reprices_abruptly"],
                    ["verbal_or_actual_intervention", "spot_fx_reverses_fast", "volatility_spikes", "positioning_de-levers"],
                    ["USDJPY drops despite rate gap", "gold rises with USD", "FX volatility spikes", "official intervention appears"],
                    {"rates": "yields no longer explain FX", "equities": "exporters and carry trades vulnerable", "commodities": "gold supported", "fx": "intervened currency rebounds or USD credibility weakens", "risk_appetite": "risk-off"},
                    ["FX resumes tracking yield gaps", "officials stop pushing back", "volatility compresses"],
                    {"geopolitical_energy": 0.3, "market_sentiment": 0.6, "monetary_inflation": 0.3},
                    [r("liquidity", "<", 0.0, 0.4, "stress can trigger safe-haven override")],
                    [p("USDJPY", "down", 1.2), p("fx_volatility", "up", 1.0), p("gold", "up", 0.9), p("DXY", "down", 0.5)],
                ),
                sensitive_assets=["DXY", "USDJPY", "real_rates", "gold", "fx_volatility", "intervention_probability"],
                observation_checklist=["DXY", "USDJPY", "rate differentials", "real rates", "Japan MOF/BOJ comments", "gold", "FX volatility"],
                layer_assets={
                    "short_end": ["USDJPY", "DXY"],
                    "long_end": ["real_rates", "gold"],
                    "risk_assets": ["fx_volatility"],
                    "commodities": ["gold"],
                    "fx": ["DXY", "USDJPY", "fx_volatility"],
                    "one_to_three_months": ["real_rates", "intervention_probability", "gold"],
                },
            ),
            GameRelationSpec(
                relation_id="credit_repricing_vs_liquidity_stress",
                relation_name="Normal risk repricing vs liquidity/credit stress",
                category="F_bonds_credit_financial_risk",
                side_a=side(
                    "A",
                    "normal_risk_repricing",
                    "Higher rates are primarily normal discount-rate repricing while credit channels remain functional.",
                    ["yields_rise", "equity_multiples_adjust", "credit_spreads_stay_orderly", "risk_assets_reprice_without_deleveraging"],
                    ["bond_volatility_rises", "stocks_digest_rates", "credit_remains_stable", "carry_returns"],
                    ["rates rise but spreads stable", "banks stable", "VIX contained", "funding markets orderly"],
                    {"rates": "yields higher", "equities": "valuation compression but no credit crash", "commodities": "demand impact limited", "fx": "USD supported", "risk_appetite": "selective"},
                    ["credit spreads widen with yields", "bank stocks break down", "funding stress appears"],
                    {"monetary_inflation": 0.5},
                    [r("liquidity", ">", 0.0, 0.6, "liquidity absorbs rate shock")],
                    [p("US10Y", "up", 0.8), p("credit_spreads", "narrow", 1.0), p("bank_stocks", "up", 0.8), p("MOVE", "down", 0.5)],
                ),
                side_b=side(
                    "B",
                    "liquidity_credit_stress",
                    "Rate or private-credit pressure is becoming a true liquidity event rather than normal repricing.",
                    ["rates_or_private_credit_losses", "spreads_and_funding_stress_rise", "deleveraging_flows_expand", "risk_assets_gap_lower"],
                    ["credit_event_surfaces", "spreads_widen", "banks_and_BDCs_sell_off", "policy_put_gets_tested"],
                    ["credit spreads widen", "MOVE leads VIX", "bank/BDC stocks fall", "funding stress rises"],
                    {"rates": "safe-haven duration can rally after stress", "equities": "banks, small caps and leveraged assets lag", "commodities": "cyclicals pressured", "fx": "USD funding bid", "risk_appetite": "risk-off"},
                    ["spreads tighten", "funding stress fades", "banks regain leadership"],
                    {"trade_credit": 1.0, "market_sentiment": 0.5},
                    [r("liquidity", "<", 0.0, 0.8, "tight liquidity converts repricing into stress")],
                    [p("credit_spreads", "widen", 1.2), p("MOVE", "up", 1.0), p("bank_stocks", "down", 0.8), p("private_credit_defaults", "up", 1.0), p("VIX", "up", 0.6)],
                ),
                sensitive_assets=["US10Y", "credit_spreads", "MOVE", "VIX", "bank_stocks", "BDC", "private_credit_defaults", "funding_stress"],
                observation_checklist=["10Y/30Y", "credit spreads", "MOVE vs VIX", "banks/BDC", "leveraged loans", "funding stress"],
                layer_assets={
                    "short_end": ["funding_stress", "MOVE"],
                    "long_end": ["US10Y", "credit_spreads"],
                    "risk_assets": ["bank_stocks", "BDC", "VIX"],
                    "commodities": ["cyclical_commodities"],
                    "fx": ["DXY"],
                    "one_to_three_months": ["credit_spreads", "private_credit_defaults", "funding_stress"],
                },
            ),
            GameRelationSpec(
                relation_id="trade_diplomacy_vs_supply_chain_rebuild",
                relation_name="Trade diplomacy optimism vs supply-chain rebuild cost",
                category="G_trade_supply_chain_policy",
                side_a=side(
                    "A",
                    "trade_diplomacy_optimism",
                    "Negotiation, tariff relief or diplomatic de-escalation lowers risk premium and supports trade-sensitive assets.",
                    ["policy_talks_or_tariff_relief", "risk_premium_falls", "cross_border_margins_expected_to_improve", "equities_and_asia_fx_reprice"],
                    ["talks_improve", "tariff_tail_risk_falls", "credit_and_equities_confirm", "capex_sentiment_recovers"],
                    ["tariff headlines improve", "semis and industrials rise", "Asia FX strengthens", "credit spreads narrow"],
                    {"rates": "growth expectations firmer", "equities": "semis, industrials and exporters outperform", "commodities": "industrial metals supported", "fx": "Asia FX stronger", "risk_appetite": "risk-on"},
                    ["export controls tighten", "supply-chain capex keeps leaving", "tariff relief fails to lift margins"],
                    {"trade_credit": 0.2, "market_sentiment": 0.5},
                    [r("growth", ">", 0.025, 0.4, "growth supports diplomacy optimism")],
                    [p("semiconductors", "up", 1.0), p("industrials", "up", 0.8), p("asia_fx", "up", 0.8), p("credit_spreads", "narrow", 0.8)],
                ),
                side_b=side(
                    "B",
                    "supply_chain_rebuild_cost",
                    "Even if diplomacy improves, firms still face structural migration, export controls and margin pressure.",
                    ["export_control_or_tariff_uncertainty", "supply_chain_redundancy_costs_rise", "margins_and_capex_efficiency_fall", "structural_risk_discount_persists"],
                    ["policy_risk_stays", "firms_keep_diversifying_supply_chains", "costs_rise", "valuation_discount_sticks"],
                    ["export controls tighten", "capex migration continues", "margins fall", "Taiwan/chip risk premium rises"],
                    {"rates": "inflationary supply cost risk", "equities": "global exporters and semis pressured", "commodities": "metals mixed", "fx": "Asia FX vulnerable", "risk_appetite": "lower"},
                    ["durable trade deal", "export controls ease", "margins recover"],
                    {"trade_credit": 1.0, "geopolitical_energy": 0.2, "ai_technology": 0.4},
                    [r("liquidity", "<", 0.0, 0.4, "tight conditions punish policy uncertainty")],
                    [p("semiconductors", "down", 1.0), p("asia_fx", "down", 0.8), p("supply_chain_cost_index", "up", 1.1), p("profit_margin_revision", "down", 0.9)],
                ),
                sensitive_assets=["semiconductors", "industrials", "asia_fx", "credit_spreads", "supply_chain_cost_index", "profit_margin_revision"],
                observation_checklist=["tariffs", "export controls", "Taiwan risk premium", "semis", "Asia FX", "supply-chain capex", "margins"],
                layer_assets={
                    "short_end": ["semiconductors", "asia_fx"],
                    "long_end": ["supply_chain_cost_index", "profit_margin_revision"],
                    "risk_assets": ["semiconductors", "industrials", "credit_spreads"],
                    "commodities": ["industrial_metals"],
                    "fx": ["asia_fx"],
                    "one_to_three_months": ["supply_chain_cost_index", "profit_margin_revision", "semiconductors"],
                },
            ),
            GameRelationSpec(
                relation_id="jobs_resilience_vs_consumer_fatigue",
                relation_name="Labor-market resilience vs consumer fatigue",
                category="H_consumption_jobs_growth",
                side_a=side(
                    "A",
                    "labor_income_resilience",
                    "Strong employment and wage income keep consumption and soft-landing expectations intact.",
                    ["job_growth_and_wages", "household_income_supports_spending", "retail_and_small_caps_confirm", "soft_landing_priced"],
                    ["payrolls_hold", "retail_sales_confirm", "consumer_cyclicals_rally", "credit_fears_fade"],
                    ["payrolls strong", "retail sales strong", "delinquencies stable", "small caps and retail stocks rise"],
                    {"rates": "growth keeps yields supported", "equities": "retail, small caps and cyclicals outperform", "commodities": "demand-sensitive commodities supported", "fx": "high-beta FX supported", "risk_appetite": "risk-on"},
                    ["retail sales weaken despite jobs", "delinquencies rise", "retail stocks fail to confirm"],
                    {"market_sentiment": 0.3, "monetary_inflation": 0.3},
                    [r("growth", ">", 0.025, 0.7, "growth supports labor-income narrative")],
                    [p("retail_sales", "up", 1.1), p("consumer_discretionary", "up", 1.0), p("small_caps", "up", 0.8), p("credit_card_delinquency", "down", 0.9)],
                ),
                side_b=side(
                    "B",
                    "consumer_fatigue",
                    "Consumers are being squeezed by rates, prices and credit stress despite still-resilient headline jobs.",
                    ["prices_and_rates_squeeze_households", "sentiment_and_credit_quality_deteriorate", "spending_slows", "growth_assets_reprice"],
                    ["sentiment_breaks", "delinquencies_rise", "retail_sales_miss", "small_caps_and_retail_sell_off"],
                    ["retail sales weak", "consumer confidence falls", "delinquencies rise", "retail/small caps lag"],
                    {"rates": "growth scare can lower long yields", "equities": "retail, small caps and cyclicals lag", "commodities": "demand-sensitive commodities pressured", "fx": "defensive USD bid", "risk_appetite": "lower"},
                    ["wage growth and spending re-accelerate", "delinquencies stabilize", "retail stocks recover"],
                    {"trade_credit": 0.5, "market_sentiment": 0.6, "monetary_inflation": 0.2},
                    [r("growth", "<", 0.02, 0.8, "weak growth supports consumer fatigue"), r("liquidity", "<", 0.0, 0.4, "tight liquidity hurts consumers")],
                    [p("retail_sales", "down", 1.1), p("consumer_confidence", "down", 0.9), p("credit_card_delinquency", "up", 1.0), p("small_caps", "down", 0.8)],
                ),
                sensitive_assets=["retail_sales", "consumer_confidence", "credit_card_delinquency", "consumer_discretionary", "small_caps", "credit_spreads"],
                observation_checklist=["nonfarm payrolls", "wages", "retail sales", "consumer confidence", "credit-card delinquencies", "small caps", "retail equities"],
                layer_assets={
                    "short_end": ["retail_sales", "consumer_confidence"],
                    "long_end": ["credit_card_delinquency", "credit_spreads"],
                    "risk_assets": ["consumer_discretionary", "small_caps"],
                    "commodities": ["cyclical_commodities"],
                    "fx": ["DXY"],
                    "one_to_three_months": ["credit_card_delinquency", "retail_sales", "small_caps"],
                },
            ),
        ]

    def _score_relation_side(
        self,
        side: GameSideSpec,
        risk_scores: Mapping[str, Mapping[str, Any]],
        market_context: Mapping[str, Any],
        price_confirmation: Mapping[str, Any],
        chain_ids: set[str],
    ) -> float:
        domain_total = sum(abs(weight) for weight in side.domain_weights.values())
        if domain_total > 0:
            domain_score = sum(self._risk(risk_scores, domain) * weight for domain, weight in side.domain_weights.items()) / domain_total
        else:
            domain_score = 0.0
        context_score = self._score_context_rules(side.context_rules, market_context)
        price_score = _to_float(price_confirmation.get("score"), 0.0)
        price_available = bool(price_confirmation.get("data_available"))
        chain_score = self._side_chain_score(side, chain_ids)

        components = [
            (domain_score, 0.44),
            (context_score, 0.28),
            (chain_score, 0.08),
        ]
        if price_available:
            components.append((price_score, 0.38))
        else:
            components.append((0.0, 0.06))
        weight_sum = sum(weight for _, weight in components)
        return _clip(sum(score * weight for score, weight in components) / max(weight_sum, 1e-9))

    def _score_context_rules(self, rules: Sequence[ContextRule], market_context: Mapping[str, Any]) -> float:
        if not rules:
            return 0.0
        score = 0.0
        total = 0.0
        for rule in rules:
            total += rule.weight
            value = self._context_value(market_context, [rule.path], 0.0)
            passed = self._context_rule_passes(value, rule.operator, rule.threshold)
            if passed:
                score += rule.weight
            elif self._context_rule_nearly_passes(value, rule.operator, rule.threshold):
                score += rule.weight * 0.45
        return _clip(score / max(total, 1e-9))

    @staticmethod
    def _context_rule_passes(value: float, operator: str, threshold: float) -> bool:
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        if operator == "abs>":
            return abs(value) > threshold
        if operator == "positive":
            return value > 0
        if operator == "negative":
            return value < 0
        return False

    @staticmethod
    def _context_rule_nearly_passes(value: float, operator: str, threshold: float) -> bool:
        band = max(abs(threshold) * 0.25, 0.005)
        if operator in {">", ">="}:
            return threshold - band <= value <= threshold
        if operator in {"<", "<="}:
            return threshold <= value <= threshold + band
        if operator == "abs>":
            return abs(value) >= max(threshold - band, 0.0)
        return False

    def _score_price_confirmations(
        self,
        rules: Sequence[PricingAssetRule],
        market_context: Mapping[str, Any],
        relation_id: str = "",
        side_id: str = "",
    ) -> Dict[str, Any]:
        confirmations: List[Dict[str, Any]] = []
        total = 0.0
        score = 0.0
        for rule in rules:
            signal = self._asset_signal(market_context, rule.asset)
            memory = self._price_memory(relation_id, side_id, rule.asset, rule.expected_direction)
            effective_weight = rule.weight * (memory.learned_weight if memory else 1.0)
            item = {
                "asset": rule.asset,
                "expected_direction": rule.expected_direction,
                "weight": rule.weight,
                "effective_weight": round(float(effective_weight), 6),
                "learned_reliability": memory.reliability if memory else None,
                "learned_sample_count": memory.sample_count if memory else 0,
                "avg_lead_lag_days": memory.avg_lead_lag_days if memory else None,
                "data_available": bool(signal),
                "score": 0.0,
                "observed": signal,
                "rationale": rule.rationale,
            }
            if signal:
                item["score"] = self._direction_confirmation_score(signal, rule.expected_direction)
                total += effective_weight
                score += item["score"] * effective_weight
            confirmations.append(item)
        return {
            "score": round(_clip(score / max(total, 1e-9)) if total else 0.0, 6),
            "data_available": bool(total),
            "confirmations": confirmations,
        }

    def _relation_identification_status(
        self,
        spec: GameRelationSpec,
        side_a_price: Mapping[str, Any],
        side_b_price: Mapping[str, Any],
        risk_scores: Mapping[str, Mapping[str, Any]],
        chain_ids: set[str],
    ) -> Dict[str, Any]:
        price_available = bool(side_a_price.get("data_available") or side_b_price.get("data_available"))
        price_score = max(_to_float(side_a_price.get("score"), 0.0), _to_float(side_b_price.get("score"), 0.0))
        category_risk = max(
            self._risk(risk_scores, domain)
            for domain in ["geopolitical_energy", "monetary_inflation", "ai_technology", "trade_credit", "market_sentiment"]
        )
        chain_present = bool(chain_ids)
        learned_samples = sum(
            int(item.get("learned_sample_count") or 0)
            for side in [side_a_price, side_b_price]
            for item in side.get("confirmations", [])
        )
        stability_proxy = _clip(0.45 * price_score + 0.25 * min(learned_samples / 12.0, 1.0) + 0.30 * (1.0 if chain_present else 0.0))

        if price_available and price_score >= 0.55 and (chain_present or learned_samples >= 4):
            status = "identifiable"
        elif price_available and price_score >= 0.30:
            status = "weak_identifiable"
        elif category_risk >= 0.20 or chain_present:
            status = "correlation_only"
        else:
            status = "unavailable"
        return {
            "relation_id": spec.relation_id,
            "identification_status": status,
            "can_trade": status in {"identifiable", "weak_identifiable"},
            "price_confirmation_available": price_available,
            "best_price_confirmation_score": round(float(price_score), 6),
            "event_chain_present": chain_present,
            "learned_confirmation_samples": learned_samples,
            "stability_proxy": round(float(stability_proxy), 6),
            "rule": "trade_allowed only when sensitive assets provide price confirmation; pure narratives remain observe_only",
        }

    @staticmethod
    def _asset_signal(market_context: Mapping[str, Any], asset: str) -> Dict[str, Any]:
        candidates = [
            market_context.get("asset_signals", {}),
            market_context.get("pricing_assets", {}),
            market_context.get("sensitive_assets", {}),
            market_context.get("market_prices", {}),
        ]
        aliases = {asset, asset.lower(), asset.upper()}
        for payload in candidates:
            if not isinstance(payload, Mapping):
                continue
            for key in aliases:
                if key in payload:
                    value = payload[key]
                    if isinstance(value, Mapping):
                        return dict(value)
                    return {"value": value}
        return {}

    def _direction_confirmation_score(self, signal: Mapping[str, Any], expected_direction: str) -> float:
        expected = self._direction_sign(expected_direction)
        observed_direction = str(signal.get("direction", signal.get("trend", ""))).lower()
        if observed_direction:
            observed = self._direction_sign(observed_direction)
            if observed != 0 and expected != 0:
                return 1.0 if observed == expected else 0.0
            if expected_direction.lower() in observed_direction:
                return 1.0
        numeric = self._extract_signal_numeric(signal)
        if numeric is None:
            return 0.0
        if expected == 0:
            return _clip(abs(numeric) * 5.0)
        return _clip(numeric * expected * 6.0)

    @staticmethod
    def _direction_sign(direction: str) -> int:
        text = direction.lower()
        positive = {"up", "higher", "rise", "rising", "stronger", "widen", "wider", "bullish", "hawkish", "risk_on", "narrow"}
        negative = {"down", "lower", "fall", "falling", "weaker", "narrower", "bearish", "dovish", "risk_off"}
        if text in positive:
            return 1
        if text in negative:
            return -1
        return 0

    @staticmethod
    def _extract_signal_numeric(signal: Mapping[str, Any]) -> Optional[float]:
        for key in ["zscore", "momentum", "return", "change", "pct_change", "spread_change", "value"]:
            if key in signal:
                return _to_float(signal.get(key), 0.0)
        return None

    @staticmethod
    def _side_chain_score(side: GameSideSpec, chain_ids: set[str]) -> float:
        text = " ".join(side.transmission_chain + side.evolution_path + side.dominance_conditions).lower()
        score = 0.0
        if "geo" in text or "oil" in text or "supply" in text:
            score = max(score, 0.35 if "geo_energy_supply_shock" in chain_ids else 0.0)
        if "ai" in text or "data_center" in text or "chip" in text:
            score = max(score, 0.35 if "ai_capex_power_materials_chain" in chain_ids else 0.0)
        if "fed" in text or "rate" in text or "inflation" in text:
            score = max(score, 0.35 if "policy_space_real_rate_chain" in chain_ids else 0.0)
        if "tariff" in text or "credit" in text or "margin" in text:
            score = max(score, 0.35 if "tariff_credit_cashflow_chain" in chain_ids else 0.0)
        return score

    def _relation_judgement(
        self,
        spec: GameRelationSpec,
        side_a_score: float,
        side_b_score: float,
    ) -> Tuple[str, float, str]:
        spread = side_a_score - side_b_score
        if abs(spread) < 0.06:
            winner = "mixed"
            summary = "A and B are both being priced; treat this as a contested narrative until sensitive assets confirm a break."
        elif spread > 0:
            winner = spec.side_a.name
            summary = f"A is dominant now: {spec.side_a.name} has stronger event/context/price confirmation."
        else:
            winner = spec.side_b.name
            summary = f"B is dominant now: {spec.side_b.name} has stronger event/context/price confirmation."
        confidence = round(_clip(0.48 + abs(spread) * 0.85 + max(side_a_score, side_b_score) * 0.18, 0.35, 0.95), 6)
        return winner, confidence, summary

    def _layer_winners(
        self,
        spec: GameRelationSpec,
        risk_scores: Mapping[str, Mapping[str, Any]],
        market_context: Mapping[str, Any],
        side_a_score: float,
        side_b_score: float,
    ) -> Dict[str, Dict[str, Any]]:
        winners: Dict[str, Dict[str, Any]] = {}
        for layer, assets in spec.layer_assets.items():
            side_a_rules = [rule for rule in spec.side_a.price_confirmations if rule.asset in assets]
            side_b_rules = [rule for rule in spec.side_b.price_confirmations if rule.asset in assets]
            price_a = self._score_price_confirmations(side_a_rules, market_context)
            price_b = self._score_price_confirmations(side_b_rules, market_context)
            layer_a = side_a_score
            layer_b = side_b_score
            reason = "No layer-specific price signal; falling back to aggregate narrative score."
            if price_a.get("data_available") or price_b.get("data_available"):
                layer_a = _clip(0.58 * side_a_score + 0.42 * _to_float(price_a.get("score"), 0.0))
                layer_b = _clip(0.58 * side_b_score + 0.42 * _to_float(price_b.get("score"), 0.0))
                reason = "Layer-specific sensitive assets are available and included in the winner score."
            winner, confidence, _ = self._relation_judgement(spec, layer_a, layer_b)
            winners[layer] = {
                "winner": winner,
                "confidence": confidence,
                "A_score": round(layer_a, 6),
                "B_score": round(layer_b, 6),
                "reason": reason,
            }
        return winners

    @staticmethod
    def _current_relation_prediction(spec: GameRelationSpec, winner: str) -> Dict[str, str]:
        if winner == spec.side_a.name:
            return spec.side_a.market_pricing_if_wins
        if winner == spec.side_b.name:
            return spec.side_b.market_pricing_if_wins
        return {
            "rates": "mixed; wait for sensitive-rate assets to confirm",
            "equities": "mixed; avoid assuming one narrative owns the tape",
            "commodities": "mixed; use commodity confirmation",
            "fx": "mixed",
            "risk_appetite": "contested",
        }

    def _records_from_any(self, payload: Any) -> List[Mapping[str, Any]]:
        if payload is None:
            return []
        if hasattr(payload, "to_dict"):
            try:
                return payload.to_dict(orient="records")
            except TypeError:
                return list(payload.to_dict().values())
        if isinstance(payload, Mapping):
            if "records" in payload and isinstance(payload["records"], list):
                return [record for record in payload["records"] if isinstance(record, Mapping)]
            return [payload]
        if isinstance(payload, list):
            return [item if isinstance(item, Mapping) else {"title": str(item)} for item in payload]
        return [{"title": str(payload)}]

    @staticmethod
    def _normalize_price_panel(raw_panel: Any) -> pd.DataFrame:
        if raw_panel is None:
            return pd.DataFrame()
        if isinstance(raw_panel, pd.DataFrame):
            frame = raw_panel.copy()
        else:
            records = raw_panel
            if isinstance(raw_panel, Mapping) and "records" in raw_panel:
                records = raw_panel["records"]
            try:
                frame = pd.DataFrame(records)
            except Exception:
                return pd.DataFrame()
        if frame.empty:
            return pd.DataFrame()
        rename = {}
        for column in frame.columns:
            lower = str(column).lower()
            if lower in {"date", "datetime", "timestamp", "time"}:
                rename[column] = "date"
            elif lower in {"close", "last", "settle", "price"}:
                rename[column] = "close"
            elif lower in {"volume", "vol"}:
                rename[column] = "volume"
        frame = frame.rename(columns=rename)
        if "date" not in frame.columns or "close" not in frame.columns:
            return pd.DataFrame()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        if "volume" not in frame.columns:
            frame["volume"] = 1.0
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(1.0)
        frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return frame

    @staticmethod
    def _safe_return(start: Any, end: Any) -> float:
        start_value = _to_float(start, 0.0)
        end_value = _to_float(end, 0.0)
        if start_value == 0.0:
            return 0.0
        return end_value / start_value - 1.0

    @staticmethod
    def _memory_key(relation_id: str, side_id: str, asset: str, expected_direction: str) -> str:
        return "|".join([relation_id, side_id, asset.lower(), expected_direction.lower()])

    def _price_memory(
        self,
        relation_id: str,
        side_id: str,
        asset: str,
        expected_direction: str,
    ) -> Optional[PriceConfirmationMemoryRecord]:
        exact = self._memory_key(relation_id, side_id, asset, expected_direction)
        if exact in self.price_confirmation_memory:
            return self.price_confirmation_memory[exact]
        relation_asset = [
            record
            for record in self.price_confirmation_memory.values()
            if record.relation_id == relation_id
            and record.asset.lower() == asset.lower()
            and record.expected_direction.lower() == expected_direction.lower()
        ]
        if relation_asset:
            return max(relation_asset, key=lambda item: (item.sample_count, item.reliability))
        fallback = self._memory_key("", "", asset, expected_direction)
        return self.price_confirmation_memory.get(fallback)

    def _normalize_event(self, record: Mapping[str, Any], fallback_source: str) -> NewsEvent:
        title = self._first_text(record, ["title", "headline", "event", "事件", "内容", "指标", "name"]) or self._record_text(record)
        summary = self._first_text(record, ["summary", "content", "description", "摘要", "详情"]) or ""
        timestamp = self._first_text(record, ["timestamp", "datetime", "date", "time", "日期"]) or datetime.utcnow().isoformat()
        source = self._first_text(record, ["source", "provider", "来源"]) or fallback_source
        tags = _as_list(record.get("tags") or record.get("tag") or record.get("分类"))
        relevance = _clip(_to_float(record.get("relevance_score", record.get("relevance", 1.0)), 1.0), 0.0, 1.5)
        sentiment = _clip(_to_float(record.get("sentiment_score", record.get("sentiment", 0.0)), 0.0), -1.0, 1.0)
        event_id = str(record.get("event_id") or self._event_hash(timestamp, title, summary, source))
        return NewsEvent(
            event_id=event_id,
            timestamp=str(timestamp),
            title=title,
            summary=summary,
            source=source,
            relevance_score=relevance,
            sentiment_score=sentiment,
            tags=tags,
        )

    @staticmethod
    def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
        lower_map = {str(key).lower(): key for key in record.keys()}
        for key in keys:
            actual = lower_map.get(key.lower())
            if actual is None:
                continue
            value = record.get(actual)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _record_text(record: Mapping[str, Any]) -> str:
        parts = []
        for key, value in record.items():
            if value is None:
                continue
            parts.append(f"{key}:{value}")
        return " ".join(parts)[:500]

    @staticmethod
    def _event_hash(timestamp: str, title: str, summary: str, source: str) -> str:
        digest = hashlib.sha1("|".join([str(timestamp), title, summary, source]).encode("utf-8")).hexdigest()
        return f"evt_{digest[:12]}"

    @staticmethod
    def _score_event_against_domain(event: NewsEvent, terms: Mapping[str, float]) -> Tuple[float, List[str]]:
        text = event.text.lower()
        hits = [(term, weight) for term, weight in terms.items() if term.lower() in text]
        if not hits:
            return 0.0, []
        weight_sum = sum(weight for _, weight in hits)
        max_weight = max(weight for _, weight in hits)
        score = _clip(0.68 * max_weight + 0.32 * min(weight_sum / 2.4, 1.0))
        return score, [term for term, _ in hits]

    @staticmethod
    def _risk(risk_scores: Mapping[str, Mapping[str, Any]], domain: str) -> float:
        return _clip(_to_float(risk_scores.get(domain, {}).get("score"), 0.0))

    @staticmethod
    def _terms(risk_scores: Mapping[str, Mapping[str, Any]], domains: Sequence[str]) -> List[str]:
        terms: List[str] = []
        for domain in domains:
            terms.extend(str(term) for term in risk_scores.get(domain, {}).get("matched_terms", []))
        return sorted(set(terms))[:20]

    def _chain_confidence(self, domain: str, risk_scores: Mapping[str, Mapping[str, Any]]) -> float:
        score = self._risk(risk_scores, domain)
        event_count = _to_float(risk_scores.get(domain, {}).get("event_count"), 0.0)
        return round(_clip(0.45 + score * 0.35 + min(event_count, 4.0) * 0.05, 0.0, 0.95), 6)

    @staticmethod
    def _aggregate_risk_score(risk_scores: Mapping[str, Mapping[str, Any]]) -> float:
        weights = {
            "geopolitical_energy": 0.32,
            "monetary_inflation": 0.22,
            "ai_technology": 0.18,
            "trade_credit": 0.18,
            "market_sentiment": 0.10,
        }
        return _clip(sum(weights[domain] * _to_float(risk_scores.get(domain, {}).get("score"), 0.0) for domain in weights))

    @staticmethod
    def _context_value(context: Mapping[str, Any], paths: Sequence[str], default: float) -> float:
        for path in paths:
            cursor: Any = context
            found = True
            for part in path.split("."):
                if isinstance(cursor, Mapping) and part in cursor:
                    cursor = cursor[part]
                else:
                    found = False
                    break
            if found:
                return _to_float(cursor, default)
        return default

    def _decide_game(self, asset: str, forces: List[GameForce]) -> GameDominance:
        ordered = sorted(forces, key=lambda item: item.score, reverse=True)
        winner = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else GameForce("none", 0.0, "neutral", [], [])
        separation = winner.score - runner_up.score
        confidence = _clip(0.50 + separation * 0.70 + winner.score * 0.18, 0.35, 0.95)
        return GameDominance(
            asset=asset,
            dominant_logic=winner.name,
            winner=winner.name,
            winner_score=round(winner.score, 6),
            runner_up=runner_up.name,
            runner_up_score=round(runner_up.score, 6),
            confidence=round(confidence, 6),
            expected_direction=winner.expected_direction,
            market_implication=self._market_implication(asset, winner),
            forces=ordered,
        )

    @staticmethod
    def _market_implication(asset: str, winner: GameForce) -> str:
        return f"{asset}: {winner.name} dominates, expected_direction={winner.expected_direction}"


def create_game_causal_analysis_engine() -> GameCausalAnalysisEngine:
    return GameCausalAnalysisEngine()
