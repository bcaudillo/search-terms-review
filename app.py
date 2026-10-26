import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Search Term Review", layout="wide")

# ---------------------------------------------------------------
# Queries: aggregate in the database, not in pandas.
# Only one row per term/keyword/match type/campaign/ad group comes back,
# instead of one row per day.
# ---------------------------------------------------------------
BASE_FROM = """
    FROM google_ads.search_term_keyword_stats s
    LEFT JOIN google_ads.campaign_history c
           ON c.id = s.campaign_id AND c._fivetran_active
    LEFT JOIN google_ads.ad_group_history g
           ON g.id = s.ad_group_id AND g._fivetran_active
"""

AGG_QUERY = text(f"""
    SELECT s.search_term,
           COALESCE(s.info_text, '(no keyword)')                 AS keyword,
           UPPER(COALESCE(s.search_term_match_type, 'UNKNOWN'))  AS match_type,
           c.name                                                AS campaign,
           g.name                                                AS ad_group,
           SUM(s.impressions)                                    AS impressions,
           SUM(s.clicks)                                         AS clicks,
           SUM(s.cost_micros) / 1000000.0                        AS spend,
           SUM(s.conversions)                                    AS conversions
    {BASE_FROM}
    WHERE s.date BETWEEN :start AND :end
    GROUP BY 1, 2, 3, 4, 5
""")

DATE_BOUNDS_QUERY = text("""
    SELECT MIN(date) AS min_d, MAX(date) AS max_d
    FROM google_ads.search_term_keyword_stats
""")

KEYWORD_QUERY = text("""
    SELECT DISTINCT info_text
    FROM google_ads.search_term_keyword_stats
    WHERE info_text IS NOT NULL
""")

LOOSE_MATCH = ("BROAD", "PHRASE", "NEAR_PHRASE")
EXACT_MATCH = ("EXACT", "NEAR_EXACT")

METRICS = {
    "Spend ($)": "spend",
    "Clicks": "clicks",
    "Impressions": "impressions",
    "Conversions": "conversions",
    "CPC ($)": "cpc",
    "CPA ($)": "cpa",
}
OPERATORS = [">=", "<=", ">", "<", "="]
OP_LABELS = {">=": "at least", "<=": "at most", ">": "more than",
             "<": "less than", "=": "exactly"}

COL_CONFIG = {
    "search_term": "Search term",
    "keyword": "Keyword",
    "match_type": "Match type",
    "campaign": "Campaign",
    "ad_group": "Ad group",
    "impressions": st.column_config.NumberColumn("Impr.", format="%d"),
    "clicks": st.column_config.NumberColumn("Clicks", format="%d"),
    "spend": st.column_config.NumberColumn("Spend", format="$%.2f"),
    "cpc": st.column_config.NumberColumn("CPC", format="$%.2f"),
    "conversions": st.column_config.NumberColumn("Conv.", format="%.1f"),
    "cpa": st.column_config.NumberColumn("CPA", format="$%.2f"),
    "suggestion": "Suggestion",
    "reason": "Why it's relevant",
}


# ---------------------------------------------------------------
# Data access
# ---------------------------------------------------------------
@st.cache_resource
def get_engine():
    """One shared engine for the whole app (the old code made a new one on every reload)."""
    return create_engine(
        st.secrets["URL"],
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,   # drop dead connections (e.g. after the DB sleeps)
        pool_recycle=300,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_date_bounds():
    with get_engine().connect() as conn:
        row = conn.execute(DATE_BOUNDS_QUERY).one()
    return pd.to_datetime(row.min_d).date(), pd.to_datetime(row.max_d).date()


@st.cache_data(ttl=3600, show_spinner=False)
def load_bid_keywords():
    with get_engine().connect() as conn:
        kws = pd.read_sql(KEYWORD_QUERY, conn)["info_text"]
    return frozenset(clean_keyword(k) for k in kws)


# max_entries caps memory: only the last few date ranges stay cached.
@st.cache_data(ttl=3600, max_entries=5, show_spinner="Loading search terms...")
def load_terms(start, end):
    with get_engine().connect() as conn:
        df = pd.read_sql(AGG_QUERY, conn, params={"start": start, "end": end})
    num = ["impressions", "clicks", "spend", "conversions"]
    df[num] = df[num].fillna(0)
    return df


def clean_keyword(text_):
    return re.sub(r'[\[\]"+]', "", str(text_)).strip().lower()


def apply_rules(df, rules, logic):
    masks = []
    for col, op, v in rules:
        s = df[col]
        if op == ">=":
            m = s >= v
        elif op == "<=":
            m = s <= v
        elif op == ">":
            m = s > v
        elif op == "<":
            m = s < v
        else:
            m = s == v
        masks.append(m.fillna(False))
    if not masks:
        return pd.Series(False, index=df.index)
    out = masks[0]
    for m in masks[1:]:
        out = (out & m) if logic == "AND" else (out | m)
    return out


STEPS = {"spend": 10.0, "clicks": 1.0, "impressions": 100.0,
         "conversions": 1.0, "cpc": 0.5, "cpa": 10.0}


def rule_builder(kind, defaults):
    """Dropdown to add a rule; each chosen rule shows an operator and a -/+ number box."""
    key = f"{kind}_rules"
    if key not in st.session_state:
        st.session_state[key] = [{"id": i, "metric": m, "op": o, "value": v}
                                 for i, (m, o, v) in enumerate(defaults)]
        st.session_state[f"{key}_next"] = len(defaults)
    rules = st.session_state[key]

    def add_rule():
        choice = st.session_state[f"{key}_add"]
        if choice:
            rid = st.session_state[f"{key}_next"]
            st.session_state[f"{key}_next"] += 1
            default_op = "<=" if METRICS[choice] in ("cpa", "cpc") else ">="
            rules.append({"id": rid, "metric": choice, "op": default_op, "value": 0.0})
            st.session_state[f"{key}_add"] = None

    def remove_rule(rid):
        st.session_state[key] = [r for r in st.session_state[key] if r["id"] != rid]

    st.sidebar.selectbox(
        "Add a rule", list(METRICS), index=None, placeholder="Select a metric...",
        key=f"{key}_add", on_change=add_rule,
    )

    active = []
    for r in st.session_state[key]:
        col = METRICS[r["metric"]]
        c_op, c_val, c_x = st.sidebar.columns([1.6, 1.6, 0.5], vertical_alignment="bottom")
        op = c_op.selectbox(r["metric"], OPERATORS, index=OPERATORS.index(r["op"]),
                            format_func=OP_LABELS.get, key=f"{key}_op_{r['id']}")
        val = c_val.number_input(" ", value=float(r["value"]), step=STEPS[col], min_value=0.0,
                                 key=f"{key}_val_{r['id']}", label_visibility="hidden")
        c_x.button("✕", key=f"{key}_x_{r['id']}", on_click=remove_rule, args=(r["id"],),
                   help="Remove rule")
        r["op"], r["value"] = op, val
        active.append((col, op, val))
    if not active:
        st.sidebar.caption("No rules yet. Pick a metric above.")
    return active


# ---------------------------------------------------------------
# Sidebar: relevance + configurable rules
# ---------------------------------------------------------------
st.sidebar.header("Relevance")
words_text = st.sidebar.text_input(
    "Relevant words (comma-separated)",
    value="fivetran",
    help="Any search term containing one of these is marked relevant. "
         "Terms that match a keyword you already bid on are relevant automatically.",
)
relevant_words = [w.strip().lower() for w in words_text.split(",") if w.strip()]

st.sidebar.divider()
st.sidebar.header("Negatives rules")
st.sidebar.caption("Feeds the Recommended negatives tab.")
neg_rules = rule_builder("neg", [("Spend ($)", ">=", 100.0), ("Conversions", "<=", 0.0)])
neg_logic = st.sidebar.radio("Match", ["All rules (AND)", "Any rule (OR)"],
                             horizontal=True, key="neg_logic")
neg_logic = "AND" if neg_logic.startswith("All") else "OR"

st.sidebar.divider()
st.sidebar.header("Adds rules")
st.sidebar.caption("Feeds the Recommended adds tab.")
add_rules = rule_builder("add", [("Conversions", ">=", 1.0), ("CPA ($)", "<=", 200.0)])
add_logic = st.sidebar.radio("Match", ["All rules (AND)", "Any rule (OR)"],
                             horizontal=True, key="add_logic")
add_logic = "AND" if add_logic.startswith("All") else "OR"

# ---------------------------------------------------------------
# Top filter bar
# ---------------------------------------------------------------
st.title("Search Term Review")

try:
    min_d, max_d = load_date_bounds()
except Exception as e:  # keep the page up with a clear message instead of a crash
    st.error(f"Couldn't reach the database. Try again in a minute.\n\n{e}")
    st.stop()

f1, f2, f3, f4 = st.columns([1.2, 2, 1.4, 2])
with f1:
    date_range = st.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

# While she's mid-selection the picker returns one date; use the full range until both are set.
start, end = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d)

try:
    raw = load_terms(start, end)
except Exception as e:
    st.error(f"Couldn't load search terms. Try again in a minute.\n\n{e}")
    st.stop()

with f2:
    campaigns = st.multiselect("Campaigns", sorted(raw["campaign"].dropna().unique()),
                               placeholder="All campaigns")
with f3:
    match_types = st.multiselect("Match types", sorted(raw["match_type"].unique()),
                                 placeholder="All match types")
with f4:
    search = st.text_input("Search terms or keywords", placeholder="e.g. etl, pricing")

# Data is already aggregated, so these are cheap filters (no groupby needed).
view = raw
if campaigns:
    view = view[view["campaign"].isin(campaigns)]
if match_types:
    view = view[view["match_type"].isin(match_types)]
view = view.copy()

view["cpc"] = (view["spend"] / view["clicks"]).where(view["clicks"] > 0)
view["cpa"] = (view["spend"] / view["conversions"]).where(view["conversions"] > 0)

if search:
    hit = (view["search_term"].str.contains(search, case=False, regex=False, na=False)
           | view["keyword"].str.contains(search, case=False, regex=False, na=False))
    view = view[hit]

# ---------------------------------------------------------------
# Relevance (automatic)
# ---------------------------------------------------------------
bid_keywords = load_bid_keywords()
term_lower = view["search_term"].str.lower()
view["is_keyword"] = term_lower.isin(bid_keywords)


def relevance_reason(term, is_kw):
    reasons = []
    if is_kw:
        reasons.append("Matches a keyword you bid on")
    hits = [w for w in relevant_words if w in term]
    if hits:
        reasons.append("Contains: " + ", ".join(hits))
    return "; ".join(reasons)


view["reason"] = [relevance_reason(t, k) for t, k in zip(term_lower, view["is_keyword"])]
auto_relevant_terms = set(view.loc[view["reason"] != "", "search_term"])

# Terms she has removed from the relevant list (overrides the automatic rules)
removed_terms = set(st.session_state.get("not_relevant", []))
view["relevant"] = (view["reason"] != "") & ~view["search_term"].isin(removed_terms)

# Vectorized (was a row-by-row .apply)
view["suggestion"] = np.select(
    [
        view["relevant"],
        view["conversions"] > 0,
        view["match_type"].isin(LOOSE_MATCH),
        view["match_type"].isin(EXACT_MATCH),
    ],
    [
        "Relevant - keep",
        "Converting - keep",
        "Tighten match type or add negative",
        "Review keyword itself",
    ],
    default="Review",
)

# ---------------------------------------------------------------
# Relevant terms (top)
# ---------------------------------------------------------------
relevant = view[view["relevant"]].sort_values("spend", ascending=False)
with st.expander(f"Relevant terms ({len(relevant)})", expanded=True):
    rel_cols = ["search_term", "reason", "keyword", "match_type", "clicks", "spend", "conversions", "cpa"]
    st.dataframe(relevant[rel_cols], hide_index=True, use_container_width=True,
                 column_config=COL_CONFIG, height=250)
    st.subheader("Relevant terms", help="A term is relevant if it matches a keyword you bid on, or contains a word from the Relevant words list in the sidebar.")
    st.multiselect(
        "Remove from relevant",
        sorted(auto_relevant_terms | removed_terms),
        key="not_relevant",
        placeholder="Type to search for a term that shouldn't be here...",
        help="Removed terms are treated like any other term: they can show up as "
             "wasted spend and in Recommended negatives. Click ✕ on a term to put it back.",
    )

# ---------------------------------------------------------------
# Wasted spend summary + heat map (click a cell to filter the tables below)
# ---------------------------------------------------------------
# Wasted spend = spend on search terms with ZERO conversions in the selected
# dates, EXCLUDING terms marked relevant (e.g. anything containing "fivetran").
wasted = view["spend"].where((view["conversions"] == 0) & (~view["relevant"]), 0)

total_spend = view["spend"].sum()
total_wasted = wasted.sum()
wasted_terms = int((wasted > 0).sum())

st.subheader("Where the money is going")
m1, m2, m3 = st.columns(3)
m1.metric("Wasted spend", f"${total_wasted:,.0f}")
m2.metric("Share of total spend", f"{(total_wasted / total_spend * 100) if total_spend else 0:.0f}%")
m3.metric("Non-converting terms", f"{wasted_terms:,}")
st.caption(
    "**Wasted spend** = spend on search terms with zero conversions in the selected dates. "
    "Terms marked relevant are never counted as wasted."
)

hm_view = st.radio(
    "Color the heat map by",
    ["Wasted spend", "CPA"],
    horizontal=True,
    help="CPA = spend divided by conversions, the cost to get one conversion. "
         "Darker = more expensive.",
)

hm = (view[["keyword", "match_type", "spend", "conversions"]]
      .assign(wasted=wasted)
      .groupby(["keyword", "match_type"])[["spend", "wasted", "conversions"]]
      .sum().reset_index())
hm["cpa"] = (hm["spend"] / hm["conversions"]).where(hm["conversions"] > 0)
hm["wasted_share"] = (hm["wasted"] / hm["spend"]).where(hm["spend"] > 0)
metric = "wasted" if hm_view == "Wasted spend" else "cpa"
hm = hm.dropna(subset=[metric])
hm = hm[hm[metric] > 0]
top_keywords = (hm.groupby("keyword")[metric].sum()
                .sort_values(ascending=False).head(20).index)
hm = hm[hm["keyword"].isin(top_keywords)]

picked = []
if hm.empty:
    st.info("Nothing to show for the current filters.")
else:
    cell = alt.selection_point(name="cell", fields=["keyword", "match_type"])
    base = alt.Chart(hm).encode(
        x=alt.X("match_type:N", title="Match type", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("keyword:N", title=None,
                sort=alt.EncodingSortField(field=metric, op="sum", order="descending")),
    )
    rects = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color(f"{metric}:Q", title=hm_view, scale=alt.Scale(scheme="reds"),
                        legend=alt.Legend(format="$,.0f")),
        opacity=alt.condition(cell, alt.value(1), alt.value(0.3)),
        tooltip=[
            alt.Tooltip("keyword:N", title="Keyword"),
            alt.Tooltip("match_type:N", title="Match type"),
            alt.Tooltip("spend:Q", title="Total spend", format="$,.2f"),
            alt.Tooltip("wasted:Q", title="Wasted spend", format="$,.2f"),
            alt.Tooltip("wasted_share:Q", title="Share wasted", format=".0%"),
            alt.Tooltip("conversions:Q", title="Conversions", format=",.1f"),
            alt.Tooltip("cpa:Q", title="CPA", format="$,.2f"),
        ],
    ).add_params(cell)
    # Streamlit can't do click-selection on layered charts, so no text layer here;
    # dollar values are in the tooltip and the legend.
    chart = rects.properties(height=max(250, 32 * hm["keyword"].nunique()))
    event = st.altair_chart(chart, use_container_width=True, on_select="rerun", key="heatmap")
    picked = event.get("selection", {}).get("cell", []) if event else []
    st.caption("Each cell is one keyword at one match type. Hover for details. "
               "Click a cell to filter the tables below; click empty space to clear.")

if picked:
    keys = {(p.get("keyword"), p.get("match_type")) for p in picked}
    view = view[[(k, m) in keys for k, m in zip(view["keyword"], view["match_type"])]]
    st.info("Filtered to: " + ", ".join(f"{k} ({m})" for k, m in keys))

# ---------------------------------------------------------------
# Tables
# ---------------------------------------------------------------
negatives = view[apply_rules(view, neg_rules, neg_logic) & ~view["relevant"]] \
    .sort_values("spend", ascending=False)
adds = view[apply_rules(view, add_rules, add_logic)
            & ~view["is_keyword"]
            & ~view["match_type"].isin(EXACT_MATCH)] \
    .sort_values("conversions", ascending=False)

tab_all, tab_neg, tab_add = st.tabs([
    f"All search terms ({len(view)})",
    f"Recommended negatives ({len(negatives)})",
    f"Recommended adds ({len(adds)})",
])

table_cols = ["search_term", "keyword", "match_type", "campaign", "ad_group",
              "impressions", "clicks", "spend", "cpc", "conversions", "cpa", "suggestion"]

with tab_all:
    st.dataframe(view.sort_values(["clicks", "spend"], ascending=False)[table_cols],
                 hide_index=True, use_container_width=True, column_config=COL_CONFIG)

with tab_neg:
    st.dataframe(negatives[table_cols], hide_index=True, use_container_width=True,
                 column_config=COL_CONFIG)
    st.download_button("Download negatives CSV", negatives[table_cols].to_csv(index=False),
                       "recommended_negatives.csv", "text/csv")

with tab_add:
    st.dataframe(adds[table_cols], hide_index=True, use_container_width=True,
                 column_config=COL_CONFIG)
    st.download_button("Download adds CSV", adds[table_cols].to_csv(index=False),
                       "recommended_adds.csv", "text/csv")