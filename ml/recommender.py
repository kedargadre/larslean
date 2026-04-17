"""TOPSIS Multi-Criteria Recommender — 'Where Should I Live?' engine."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import IRISH_COUNTIES


def topsis_rank(
    scores_df: pd.DataFrame,
    budget: float,
    commute_tolerance: str,
    family_size: int,
    work_mode: str,
    priorities: dict = None,
) -> pd.DataFrame:
    """
    Rank all 26 counties using TOPSIS (Technique for Order of Preference
    by Similarity to Ideal Solution).

    Args:
        scores_df: DataFrame with all county features and scores.
        budget: Monthly take-home salary (€).
        commute_tolerance: "low" | "medium" | "high"
        family_size: 1–5
        work_mode: "remote" | "hybrid" | "office"
        priorities: Optional dict of user-weighted criteria (0–1 each).

    Returns:
        DataFrame ranked by TOPSIS closeness score (1 = best match).
    """

    df = scores_df.copy()

    # ── 1. Personalised affordability score (budget-relative) ────
    # Replace static affordability_score with one computed from the
    # user's actual budget vs. local rent — this is the key driver
    # that makes results differ by budget.
    if "avg_monthly_rent" in df.columns:
        rent_share = df["avg_monthly_rent"] / max(budget, 1)
        # Score 100 when rent is 0% of budget, approaches 0 as rent nears budget.
        # Allow slightly negative so over-budget areas still rank lower, not equal.
        df["_personal_afford"] = np.clip((1 - rent_share / 0.8) * 100, -20, 100)
    else:
        df["_personal_afford"] = df.get("affordability_score", 50)

    # ── 2. Build decision matrix columns ────────────────────────
    # Benefit criteria (higher = better):
    #   _personal_afford, livability_score, employment_rate, transport_score
    # Cost criteria (lower = better):
    #   congestion_delay_minutes, est_annual_energy_cost, risk_score

    criteria = {
        # (column, is_benefit, base_weight)
        "_personal_afford":         (True,  0.30),
        "livability_score":         (True,  0.20),
        "employment_rate":          (True,  0.10),
        "transport_score":          (True,  0.10),
        "congestion_delay_minutes": (False, 0.05),
        "est_annual_energy_cost":   (False, 0.05),
        "risk_score":               (False, 0.20),
    }

    # ── 3. Adjust weights based on user profile ─────────────────
    weights = {}
    for col, (is_benefit, base_w) in criteria.items():
        weights[col] = base_w

    # Budget sensitivity: tighter budget → weight personal affordability higher
    if budget < 2500:
        weights["_personal_afford"] += 0.15
        weights["livability_score"] -= 0.05
    elif budget > 5000:
        weights["livability_score"] += 0.10
        weights["_personal_afford"] -= 0.05

    # Commute tolerance
    if commute_tolerance == "low":
        weights["transport_score"] += 0.10
        weights["congestion_delay_minutes"] += 0.05
    elif commute_tolerance == "high":
        weights["transport_score"] -= 0.05

    # Work mode
    if work_mode == "remote":
        weights["congestion_delay_minutes"] -= 0.03
        weights["transport_score"] -= 0.05
        weights["livability_score"] += 0.05
        weights["est_annual_energy_cost"] += 0.03  # WFH energy matters more
    elif work_mode == "office":
        weights["transport_score"] += 0.08
        weights["congestion_delay_minutes"] += 0.05

    # Family size: larger families care more about affordability + energy
    if family_size >= 3:
        weights["_personal_afford"] += 0.05
        weights["est_annual_energy_cost"] += 0.03

    # Apply user priority overrides
    # Map "affordability_score" from UI → internal "_personal_afford"
    if priorities:
        _key_map = {"affordability_score": "_personal_afford"}
        for k, v in priorities.items():
            mapped = _key_map.get(k, k)
            if mapped in weights:
                weights[mapped] = v

    # Normalize weights to sum to 1
    total_w = sum(weights.values())
    weights = {k: max(v, 0.01) / total_w for k, v in weights.items()}

    # ── 3. Build the decision matrix ────────────────────────────
    available_cols = [c for c in criteria.keys() if c in df.columns]
    D = df[available_cols].fillna(0).values.astype(float)

    # ── 4. Normalize the decision matrix (Min-Max normalization) ─
    # Vector normalization compresses low-variance clusters. Min-Max spreads 
    # them across [0,1], ensuring user weightings actually affect the ranking.
    D_min = D.min(axis=0)
    D_max = D.max(axis=0)
    diffs = D_max - D_min
    diffs[diffs == 0] = 1  # prevent division by zero
    R = (D - D_min) / diffs

    # ── 5. Apply weights ────────────────────────────────────────
    W = np.array([weights.get(c, 0.05) for c in available_cols])
    V = R * W

    # ── 6. Determine ideal and anti-ideal solutions ─────────────
    ideal = np.zeros(len(available_cols))
    anti_ideal = np.zeros(len(available_cols))

    for i, col in enumerate(available_cols):
        is_benefit = criteria[col][0]
        if is_benefit:
            ideal[i] = V[:, i].max()
            anti_ideal[i] = V[:, i].min()
        else:
            ideal[i] = V[:, i].min()
            anti_ideal[i] = V[:, i].max()

    # ── 7. Calculate distances ──────────────────────────────────
    dist_ideal = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    dist_anti = np.sqrt(((V - anti_ideal) ** 2).sum(axis=1))

    # ── 8. TOPSIS closeness coefficient ─────────────────────────
    denom = dist_ideal + dist_anti
    denom[denom == 0] = 1
    closeness = dist_anti / denom

    # ── 9. Build results ────────────────────────────────────────
    df["match_score"] = np.round(closeness * 100, 1)

    # Budget fit: how much remains after rent + energy + commute
    bedroom_est = max(1, family_size - 1) if family_size > 1 else 1
    rent_mul = {1: 0.75, 2: 1.0, 3: 1.25, 4: 1.5}.get(bedroom_est, 1.0)

    df["est_monthly_cost"] = (
        df["avg_monthly_rent"] * rent_mul
        + df["est_annual_energy_cost"] / 12 * (1 + (family_size - 1) * 0.15)
        + (df["congestion_delay_minutes"] * 2 * 22 if work_mode != "remote" else 80)
    )
    df["monthly_remaining"] = budget - df["est_monthly_cost"]
    df["budget_fit"] = np.where(
        df["monthly_remaining"] > 1000, "🟢 Comfortable",
        np.where(df["monthly_remaining"] > 300, "🟡 Tight",
                 np.where(df["monthly_remaining"] > 0, "🔴 Stretched",
                          np.where(df["monthly_remaining"] > -500, "⛔ You'd be in the red",
                                   "⛔ You'd be seriously in the red")))
    )

    # Sort by match score
    result = df.sort_values("match_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result


def get_recommendation_narrative(
    top_counties: pd.DataFrame,
    budget: float,
    family_size: int,
    work_mode: str,
    commute_tolerance: str,
) -> str:
    """Generate a human-readable recommendation summary (for LLM prompt or standalone)."""
    if len(top_counties) == 0:
        return "Unable to generate recommendations with the provided criteria."

    top3 = top_counties.head(3)
    lines = []
    lines.append(f"**Your Profile:** €{budget:,.0f}/month • {family_size}-person household • {work_mode.title()} worker • {commute_tolerance.title()} commute tolerance\n")
    lines.append("---\n")

    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top3.iterrows()):
        county = row["county"]
        score = row["match_score"]
        remaining = row["monthly_remaining"]
        fit = row["budget_fit"]
        risk = row.get("risk_score", 50)
        livability = row.get("livability_score", 50)
        afford = row.get("affordability_score", 50)
        rent = row.get("avg_monthly_rent", 0)

        lines.append(f"### {medals[i]} #{i+1} — {county} (Match: {score:.0f}/100)")
        lines.append(f"- **Budget Fit:** {fit} — €{remaining:,.0f} remaining/month")
        lines.append(f"- **Rent:** €{rent:,.0f}/mo | **Risk:** {risk:.0f} | **Livability:** {livability:.0f} | **Affordability:** {afford:.0f}")

        # Contextual insight
        if afford > 65 and risk < 40:
            lines.append(f"- 💡 *{county} is a hidden gem — affordable AND low risk*")
        elif afford > 65:
            lines.append(f"- 💡 *Great value for money, but keep an eye on the risk factors*")
        elif livability > 65:
            lines.append(f"- 💡 *Excellent quality of life — worth the premium*")
        lines.append("")

    # Bottom line
    best = top3.iloc[0]
    lines.append(f"**Bottom line:** Based on your profile, **{best['county']}** is your strongest match with a {best['match_score']:.0f}/100 score.")

    return "\n".join(lines)
