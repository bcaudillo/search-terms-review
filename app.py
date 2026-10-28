import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Search Term Review", layout="wide")

# ---------------------------------------------------------------
# Queries (aggregation happens in the database, not in pandas)
# ---------------------------------------------------------------
BASE_FROM = """
    FROM google_ads.search_term_keyword_stats s
    LEFT JOIN google_ads.campaign_history c
           ON c.id = s.campaign_id AND c._fivetran_active
    LEFT JOIN google_ads.ad_group_history g
           ON g.id = s.ad_group_id AND g._fivetran_active
"""

# One row per term / keyword / match type / campaign / ad group.
# status = the term's most recent status in Google Ads (ADDED, EXCLUDED, ...).
TERMS_QUERY = text(f"""
    SELECT s.search_term,
           COALESCE(s.info_text, '(no keyword)')                 AS keyword,
           UPPER(COALESCE(s.search_term_match_type, 'UNKNOWN'))  AS match_type,
           c.name                                                AS campaign,
           g.name                                                AS ad_group,
           (ARRAY_AGG(s.status ORDER BY s.date DESC))[1]         AS status,
           SUM(s.impressions)                                    AS impressions,
           SUM(s.clicks)                                         AS clicks,
           SUM(s.cost_micros) / 1000000.0                        AS spend,
           SUM(s.conversions)                                    AS conversions
    {BASE_FROM}
    WHERE s.date BETWEEN :start AND :end
    GROUP BY 1, 2, 3, 4, 5
""")

# One row per day per campaign, for trends and the campaign heat map.
DAILY_QUERY = text(f"""
    SELECT s.date,
           c.name                          AS campaign,
           SUM(s.clicks)                   AS clicks,
           SUM(s.cost_micros) / 1000000.0  AS spend,
           SUM(s.conversions)              AS conversions
    {BASE_FROM}
    WHERE s.date BETWEEN :start AND :end
    GROUP BY 1, 2
""")

# Every search term ever added or negated (all time, not limited by the date range).
ADDED_NEGATED_QUERY = text(f"""
    SELECT * FROM (
        SELECT s.search_term,
               UPPER(COALESCE(s.search_term_match_type, 'UNKNOWN'))  AS match_type,
               c.name                                                AS campaign,
               g.name                                                AS ad_group,
               (ARRAY_AGG(s.status ORDER BY s.date DESC))[1]         AS status,
               MAX(s.date)                                           AS last_seen
        {BASE_FROM}
        GROUP BY 1, 2, 3, 4
    ) t
    WHERE UPPER(status) IN ('ADDED', 'EXCLUDED', 'ADDED_EXCLUDED')
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

STATUS_LABELS = {
    "ADDED": "Added",
    "EXCLUDED": "Negated",
    "ADDED_EXCLUDED": "Added & negated",
    "NONE": "Not added",
}

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
STEPS = {"spend": 10.0, "clicks": 1.0, "impressions": 100.0,
         "conversions": 1.0, "cpc": 0.5, "cpa": 10.0}

NEG_DEFAULTS = [("Spend ($)", ">=", 100.0), ("Conversions", "<=", 0.0)]
ADD_DEFAULTS = [("Conversions", ">=", 1.0), ("CPA ($)", "<=", 200.0)]

WASTED_HELP = ("Wasted spend = spend on the Recommended negatives: terms that match your "
               "negatives rules and aren't keywords you bid on. Spend on keywords you bid on "
               "is never counted as wasted, since some are kept for visibility.")

COL_CONFIG = {
    "search_term": "Search term",
    "keyword": "Keyword",
    "match_type": "Match type",
    "campaign": "Campaign",
    "ad_group": "Ad group",
    "brand_type": "Type",
    "status_label": "In Google Ads",
    "last_seen": st.column_config.DateColumn("Last seen"),
    "impressions": st.column_config.NumberColumn("Impr.", format="%d"),
    "clicks": st.column_config.NumberColumn("Clicks", format="%d"),
    "spend": st.column_config.NumberColumn("Spend", format="$%.2f"),
    "cpc": st.column_config.NumberColumn("CPC", format="$%.2f"),
    "conversions": st.column_config.NumberColumn("Conv.", format="%.1f"),
    "cpa": st.column_config.NumberColumn("CPA", format="$%.2f",
                                         help="Cost per conversion: spend divided by conversions."),
    "suggestion": "Suggestion",
    "why": "Why",
}


# ---------------------------------------------------------------
# Data access
# ---------------------------------------------------------------
@st.cache_resource
def get_engine():
    return create_engine(st.secrets["URL"], pool_size=2, max_overflow=2,
                         pool_pre_ping=True, pool_recycle=300)


def clean_keyword(text_):
    return re.sub(r'[\[\]"+]', "", str(text_)).strip().lower()


def brand_type(name):
    """Campaign names ending in B = brand, G = non-brand."""
    last = str(name).strip()[-1:].upper() if isinstance(name, str) else ""
    return {"B": "Brand", "G": "Non-brand"}.get(last, "Other")


def label_status(df):
    df["status"] = df["status"].fillna("UNKNOWN").astype(str).str.upper()
    df["status_label"] = df["status"].map(STATUS_LABELS).fillna("Unknown")
    return df


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


@st.cache_data(ttl=3600, max_entries=5, show_spinner="Loading search terms...")
def load_terms(start, end):
    with get_engine().connect() as conn:
        df = pd.read_sql(TERMS_QUERY, conn, params={"start": start, "end": end})
    num = ["impressions", "clicks", "spend", "conversions"]
    df[num] = df[num].astype(float).fillna(0)
    df = label_status(df)
    df["brand_type"] = df["campaign"].map(brand_type)
    return df


@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def load_daily(start, end):
    with get_engine().connect() as conn:
        df = pd.read_sql(DAILY_QUERY, conn, params={"start": start, "end": end})
    num = ["clicks", "spend", "conversions"]
    df[num] = df[num].astype(float).fillna(0)
    df["date"] = pd.to_datetime(df["date"])
    df["brand_type"] = df["campaign"].map(brand_type)
    return df


@st.cache_data(ttl=3600, show_spinner="Loading added and negated terms...")
def load_added_negated():
    with get_engine().connect() as conn:
        df = pd.read_sql(ADDED_NEGATED_QUERY, conn)
    df = label_status(df)
    df["last_seen"] = pd.to_datetime(df["last_seen"]).dt.date
    df["brand_type"] = df["campaign"].map(brand_type)
    return df


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def add_rates(df):
    df["cpc"] = (df["spend"] / df["clicks"]).where(df["clicks"] > 0)
    df["cpa"] = (df["spend"] / df["conversions"]).where(df["conversions"] > 0)
    return df


def summarize(df, by):
    """Roll term-level rows up to campaigns, ad groups, etc."""
    out = (df.groupby(by, dropna=False)[["clicks", "spend", "conversions"]]
           .sum().reset_index())
    out = add_rates(out)
    return out.sort_values("spend", ascending=False)


def why_text(df):
    """A plain-language reason for each row."""
    out = []
    for spend, clicks, conv, cpa in zip(df["spend"], df["clicks"], df["conversions"], df["cpa"]):
        if conv > 0:
            out.append(f"{conv:,.1f} conversions at ${cpa:,.0f} each (${spend:,.0f} spent)")
        elif spend > 0:
            out.append(f"${spend:,.0f} spent over {clicks:,.0f} clicks with no conversions")
        else:
            out.append("No spend in these dates")
    return out


def suggestion(df):
    return np.select(
        [df["conversions"] > 0,
         df["match_type"].isin(LOOSE_MATCH),
         df["match_type"].isin(EXACT_MATCH)],
        ["Converting - keep",
         "Tighten match type or add negative",
         "Review keyword itself"],
        default="Review",
    )


def download(df, cols, filename, label="Download CSV"):
    st.download_button(label, df[cols].to_csv(index=False), filename, "text/csv",
                       key=f"dl_{filename}")


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


def saved_rules(kind, defaults):
    """The rules MK set on the Search terms page (or the defaults), usable on any page."""
    rules = st.session_state.get(f"{kind}_rules")
    if rules is None:
        rules = [{"metric": m, "op": o, "value": v} for m, o, v in defaults]
    active = [(METRICS[r["metric"]], r["op"], r["value"]) for r in rules]
    return active, st.session_state.get(f"{kind}_logic_saved", "AND")


def recommended_negatives(df):
    rules, logic = saved_rules("neg", NEG_DEFAULTS)
    # Never negate your own keyword or something already negated.
    return df[apply_rules(df, rules, logic)
              & ~df["is_keyword"]
              & ~df["status"].isin(["EXCLUDED", "ADDED_EXCLUDED"])]


def recommended_adds(df):
    rules, logic = saved_rules("add", ADD_DEFAULTS)
    # Never recommend adding something already added or already a keyword.
    return df[apply_rules(df, rules, logic)
              & ~df["is_keyword"]
              & ~df["match_type"].isin(EXACT_MATCH)
              & ~df["status"].isin(["ADDED", "ADDED_EXCLUDED"])]


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
    if not st.session_state[key]:
        st.sidebar.caption("No rules yet. Pick a metric above.")

    # Remember AND/OR even when she switches pages.
    saved = st.session_state.get(f"{kind}_logic_saved", "AND")
    choice = st.sidebar.radio("Match", ["All rules (AND)", "Any rule (OR)"],
                              index=0 if saved == "AND" else 1,
                              horizontal=True, key=f"{kind}_logic")
    st.session_state[f"{kind}_logic_saved"] = "AND" if choice.startswith("All") else "OR"


def heatmap(df, x, y, color, color_title, x_title, x_sort=None, key=None, selectable=False):
    """Simple heat map. Returns the clicked cells when selectable."""
    sel = alt.selection_point(name="cell", fields=[x, y]) if selectable else None
    chart = alt.Chart(df).mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        x=alt.X(f"{x}:N", title=x_title, sort=x_sort,
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y(f"{y}:N", title=None,
                sort=alt.EncodingSortField(field=color, op="sum", order="descending")),
        color=alt.Color(f"{color}:Q", title=color_title, scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(format="$,.0f")),
        tooltip=[
            alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:N"),
            alt.Tooltip("spend:Q", title="Spend", format="$,.2f"),
            alt.Tooltip("conversions:Q", title="Conversions", format=",.1f"),
            alt.Tooltip("cpa:Q", title="CPA", format="$,.2f"),
        ],
    ).properties(height=max(250, 30 * df[y].nunique()))
    if selectable:
        chart = chart.encode(opacity=alt.condition(sel, alt.value(1), alt.value(0.3))).add_params(sel)
        event = st.altair_chart(chart, use_container_width=True, on_select="rerun", key=key)
        return event.get("selection", {}).get("cell", []) if event else []
    st.altair_chart(chart, use_container_width=True)
    return []


# ---------------------------------------------------------------
# Pages
# ---------------------------------------------------------------
def overview_page():
    st.title("Overview")
    st.caption(f"{start:%b %d, %Y} to {end:%b %d, %Y}")

    negatives = recommended_negatives(terms)
    spend, conv = terms["spend"].sum(), terms["conversions"].sum()
    wasted = negatives["spend"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spend", f"${spend:,.0f}")
    m2.metric("Conversions", f"{conv:,.1f}")
    m3.metric("Cost per conversion", f"${spend / conv:,.0f}" if conv else "–")
    m4.metric("Wasted spend", f"${wasted:,.0f}",
              f"{len(negatives):,} terms to negate", delta_color="off", help=WASTED_HELP)

    st.subheader("Spend over time")
    trend = daily.groupby("date")[["spend", "conversions"]].sum().reset_index()
    trend["spend"] = trend["spend"].round(2)
    if trend.empty:
        st.info("No data for the current filters.")
    else:
        st.line_chart(trend, x="date", y="spend", y_label="Spend ($)", x_label="")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top campaigns by spend")
        camp = summarize(terms, ["campaign", "brand_type"]).head(10)
        st.dataframe(camp[["campaign", "brand_type", "spend", "conversions", "cpa"]],
                     hide_index=True, use_container_width=True, column_config=COL_CONFIG)
    with c2:
        st.subheader("Top terms to negate", help=WASTED_HELP)
        top_neg = negatives.sort_values("spend", ascending=False).head(10)
        st.dataframe(top_neg[["search_term", "campaign", "clicks", "spend"]],
                     hide_index=True, use_container_width=True, column_config=COL_CONFIG)


def search_terms_page():
    st.title("Search terms")

    st.sidebar.divider()
    st.sidebar.subheader("Negatives rules")
    st.sidebar.caption("Feeds the Recommended negatives tab and Wasted spend.")
    rule_builder("neg", NEG_DEFAULTS)

    st.sidebar.divider()
    st.sidebar.subheader("Adds rules")
    st.sidebar.caption("Feeds the Recommended adds tab.")
    rule_builder("add", ADD_DEFAULTS)

    view = terms.copy()
    view["why"] = why_text(view)
    view["suggestion"] = suggestion(view)

    # Heat map: keyword x match type, click a cell to filter the tables
    st.subheader("Where's the money going?")
    hm_view = st.radio("Color the heat map by", ["Spend", "CPA"], horizontal=True,
                       help="CPA = spend divided by conversions. Darker = higher.")
    hm = summarize(view, ["keyword", "match_type"])
    metric = "spend" if hm_view == "Spend" else "cpa"
    hm = hm[hm[metric] > 0]
    top_kw = hm.groupby("keyword")["spend"].sum().sort_values(ascending=False).head(20).index
    hm = hm[hm["keyword"].isin(top_kw)]

    picked = []
    if hm.empty:
        st.info("Nothing to show for the current filters.")
    else:
        picked = heatmap(hm, "match_type", "keyword", metric, hm_view, "Match type",
                         key="kw_heatmap", selectable=True)
        st.caption("Each cell is one keyword at one match type. Top 20 keywords by spend. "
                   "Click a cell to filter the tables below; click empty space to clear.")
    if picked:
        keys = {(p.get("keyword"), p.get("match_type")) for p in picked}
        view = view[[(k, m) in keys for k, m in zip(view["keyword"], view["match_type"])]]
        st.info("Filtered to: " + ", ".join(f"{k} ({m})" for k, m in keys))

    negatives = recommended_negatives(view).sort_values("spend", ascending=False)
    adds = recommended_adds(view).sort_values("conversions", ascending=False)

    cols = ["search_term", "status_label", "why", "keyword", "match_type", "campaign",
            "ad_group", "impressions", "clicks", "spend", "cpc", "conversions", "cpa",
            "suggestion"]

    tab_neg, tab_add, tab_all = st.tabs([
        f"Recommended negatives ({len(negatives)})",
        f"Recommended adds ({len(adds)})",
        f"All search terms ({len(view)})",
    ])
    with tab_neg:
        st.metric("Wasted spend", f"${negatives['spend'].sum():,.2f}", help=WASTED_HELP)
        st.dataframe(negatives[cols], hide_index=True, use_container_width=True,
                     column_config=COL_CONFIG)
        download(negatives, cols, "recommended_negatives.csv")
    with tab_add:
        st.dataframe(adds[cols], hide_index=True, use_container_width=True,
                     column_config=COL_CONFIG)
        download(adds, cols, "recommended_adds.csv")
    with tab_all:
        st.dataframe(view.sort_values(["clicks", "spend"], ascending=False)[cols],
                     hide_index=True, use_container_width=True, column_config=COL_CONFIG)
        download(view, cols, "all_search_terms.csv")


def campaigns_page():
    st.title("Campaigns")

    st.subheader("Where's the money going?")
    hm_view = st.radio("Color by", ["Spend", "CPA"], horizontal=True, key="camp_hm")
    wk = daily.assign(week=daily["date"].dt.to_period("W").dt.start_time)
    wk = wk.groupby(["campaign", "week"])[["clicks", "spend", "conversions"]].sum().reset_index()
    wk = add_rates(wk)
    metric = "spend" if hm_view == "Spend" else "cpa"
    wk = wk[wk[metric] > 0]
    top = wk.groupby("campaign")["spend"].sum().sort_values(ascending=False).head(25).index
    wk = wk[wk["campaign"].isin(top)].sort_values("week")
    wk["week_label"] = wk["week"].dt.strftime("%b %d")
    if wk.empty:
        st.info("No data for the current filters.")
    else:
        heatmap(wk, "week_label", "campaign", metric, hm_view, "Week starting",
                x_sort=list(wk["week_label"].unique()))
        st.caption("Each cell is one campaign in one week. Hover for details. "
                   "Showing the top 25 campaigns by spend.")

    st.subheader("All campaigns")
    camp = summarize(terms, ["campaign", "brand_type"])
    cols = ["campaign", "brand_type", "clicks", "spend", "conversions", "cpa"]
    st.dataframe(camp[cols], hide_index=True, use_container_width=True, column_config=COL_CONFIG)
    download(camp, cols, "campaigns.csv")

    st.subheader("Drill into a campaign")
    choice = st.selectbox("Campaign", camp["campaign"].dropna().tolist(), index=None,
                          placeholder="Pick a campaign to see its ad groups...")
    if choice:
        ag = summarize(terms[terms["campaign"] == choice], ["ad_group"])
        ag_cols = ["ad_group", "clicks", "spend", "conversions", "cpa"]
        st.dataframe(ag[ag_cols], hide_index=True, use_container_width=True,
                     column_config=COL_CONFIG)


def ad_groups_page():
    st.title("Ad groups")

    st.subheader("Where's the money going?")
    ag = summarize(terms, ["campaign", "ad_group"])
    cols = ["campaign", "ad_group", "clicks", "spend", "conversions", "cpa"]
    st.dataframe(ag[cols], hide_index=True, use_container_width=True, column_config=COL_CONFIG)
    download(ag, cols, "ad_groups.csv")

    st.subheader("Drill into an ad group")
    pairs = {f"{c} › {a}": (c, a)
             for c, a in ag[["campaign", "ad_group"]].dropna().itertuples(index=False, name=None)}
    label = st.selectbox("Ad group", list(pairs), index=None,
                         placeholder="Pick an ad group to see its search terms...")
    if label:
        camp_name, ag_name = pairs[label]
        sub = terms[(terms["campaign"] == camp_name) & (terms["ad_group"] == ag_name)].copy()
        sub["why"] = why_text(sub)
        sub_cols = ["search_term", "status_label", "why", "keyword", "match_type",
                    "clicks", "spend", "conversions", "cpa"]
        st.dataframe(sub.sort_values("spend", ascending=False)[sub_cols], hide_index=True,
                     use_container_width=True, column_config=COL_CONFIG)
        download(sub, sub_cols, "ad_group_search_terms.csv")


def added_negated_page():
    st.title("Added & negated")
    st.caption("Every search term marked added or negated in Google Ads. All time, not "
               "limited by the date range. The other filters still apply.")

    try:
        hist = load_added_negated()
    except Exception as e:
        st.error(f"Couldn't load added and negated terms.\n\n{e}")
        return

    if allowed_campaigns is not None:
        hist = hist[hist["campaign"].isin(allowed_campaigns)]
    if match_types:
        hist = hist[hist["match_type"].isin(match_types)]
    if search:
        hist = hist[hist["search_term"].str.contains(search, case=False, regex=False, na=False)]

    negated = hist[hist["status"].isin(["EXCLUDED", "ADDED_EXCLUDED"])]
    added = hist[hist["status"].isin(["ADDED", "ADDED_EXCLUDED"])]
    cols = ["search_term", "match_type", "campaign", "ad_group", "brand_type", "last_seen"]

    tab_neg, tab_add = st.tabs([f"Negated ({len(negated)})", f"Added ({len(added)})"])
    with tab_neg:
        st.dataframe(negated.sort_values("search_term")[cols], hide_index=True,
                     use_container_width=True, column_config=COL_CONFIG)
        download(negated, cols, "negated_terms.csv")
    with tab_add:
        st.dataframe(added.sort_values("search_term")[cols], hide_index=True,
                     use_container_width=True, column_config=COL_CONFIG)
        download(added, cols, "added_terms.csv")


# ---------------------------------------------------------------
# Navigation + shared filter sidebar (applies to every page)
# ---------------------------------------------------------------
pg = st.navigation([
    st.Page(overview_page, title="Overview", default=True),
    st.Page(search_terms_page, title="Search terms"),
    st.Page(campaigns_page, title="Campaigns"),
    st.Page(ad_groups_page, title="Ad groups"),
    st.Page(added_negated_page, title="Added & negated"),
])

st.sidebar.header("Filter")

try:
    min_d, max_d = load_date_bounds()
except Exception as e:
    st.error(f"Couldn't reach the database. Try again in a minute.\n\n{e}")
    st.stop()

date_range = st.sidebar.date_input("Date range", (min_d, max_d),
                                   min_value=min_d, max_value=max_d)
start, end = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d)

try:
    raw = load_terms(start, end)
    daily_raw = load_daily(start, end)
    bid_keywords = load_bid_keywords()
except Exception as e:
    st.error(f"Couldn't load data. Try again in a minute.\n\n{e}")
    st.stop()

brand = st.sidebar.radio("Campaign type", ["All", "Brand", "Non-brand"], horizontal=True,
                         help="Brand = campaign name ends in B. Non-brand = ends in G.")
camp_types = raw[["campaign", "brand_type"]].drop_duplicates()
if brand != "All":
    camp_types = camp_types[camp_types["brand_type"] == brand]
campaign_options = sorted(camp_types["campaign"].dropna().unique())
campaigns = st.sidebar.multiselect("Campaigns", campaign_options, placeholder="All campaigns")
match_types = st.sidebar.multiselect("Match types", sorted(raw["match_type"].unique()),
                                     placeholder="All match types")
search = st.sidebar.text_input("Search terms or keywords", placeholder="e.g. etl, pricing")

allowed_campaigns = None
if brand != "All" or campaigns:
    allowed_campaigns = set(campaigns) if campaigns else set(campaign_options)

terms, daily = raw, daily_raw
if allowed_campaigns is not None:
    terms = terms[terms["campaign"].isin(allowed_campaigns)]
    daily = daily[daily["campaign"].isin(allowed_campaigns)]
if match_types:
    terms = terms[terms["match_type"].isin(match_types)]
if search:
    terms = terms[terms["search_term"].str.contains(search, case=False, regex=False, na=False)
                  | terms["keyword"].str.contains(search, case=False, regex=False, na=False)]

terms = add_rates(terms.copy())
terms["is_keyword"] = terms["search_term"].str.lower().isin(bid_keywords)

pg.run()
