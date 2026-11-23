import datetime as dt
import logging
import re
from collections import Counter
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Search Term Review", layout="wide")
log = logging.getLogger("search_terms")

# ---------------------------------------------------------------
# Queries (aggregation happens in the database, not in pandas)
#
# Two sources:
#   * ad_group_stats: every click and dollar. Used for campaign and ad group numbers.
#     Matches the Google Ads Campaigns and Ad groups screens.
#   * search_term_keyword_stats: the search terms report. Google leaves out low-volume
#     terms and video campaigns have none, so it covers only part of spend. Used for
#     search-term lists, wasted spend, recommendations, themes and Added & negated.
# ---------------------------------------------------------------
CAMPAIGN_NAME = "COALESCE(c.name, 'Campaign ' || s.campaign_id)"
AD_GROUP_NAME = "COALESCE(g.name, '(unknown ad group)')"
KEYWORD = "COALESCE(s.info_text, '(no keyword)')"
MATCH_TYPE = "UPPER(COALESCE(s.search_term_match_type, 'UNKNOWN'))"

TERM_FROM = """
    FROM google_ads.search_term_keyword_stats s
    LEFT JOIN google_ads.campaign_history c
           ON c.id = s.campaign_id AND c._fivetran_active
    LEFT JOIN google_ads.ad_group_history g
           ON g.id = s.ad_group_id AND g._fivetran_active
"""

# Sidebar match-type and search filters. Only search-term rows have these fields.
TERM_FILTERS = f"""
      AND (CARDINALITY(CAST(:match_types AS text[])) = 0
           OR {MATCH_TYPE} = ANY(CAST(:match_types AS text[])))
      AND (CAST(:search AS text) = ''
           OR s.search_term ILIKE :search_like OR s.info_text ILIKE :search_like)
"""

# A term's status per campaign, all time (not limited by the date range).
# Negated anywhere in the campaign wins, then added, otherwise not added.
STATUS_CTE = """
    status AS (
        SELECT campaign_id, search_term,
               CASE WHEN BOOL_OR(UPPER(status) IN ('EXCLUDED', 'ADDED_EXCLUDED')) THEN 'EXCLUDED'
                    WHEN BOOL_OR(UPPER(status) = 'ADDED') THEN 'ADDED'
                    ELSE 'NONE' END  AS status,
               MAX(date)             AS last_seen
        FROM google_ads.search_term_keyword_stats
        GROUP BY 1, 2
    )
"""


def terms_query(by_period):
    """One row per campaign + search term (per period for trends).

    A term can match several keywords, ad groups and match types. Totals add all of them;
    keyword is the one with the most spend (ties alphabetical), and match type / ad group
    are the biggest within that keyword, so every load gives the same answer.
    """
    period = "CASE WHEN s.date > :mid THEN 'current' ELSE 'prior' END" if by_period else "'all'"
    return text(f"""
    WITH base AS (
        SELECT {period}         AS period,
               s.campaign_id,
               {CAMPAIGN_NAME}  AS campaign,
               s.search_term,
               {KEYWORD}        AS keyword,
               {MATCH_TYPE}     AS match_type,
               {AD_GROUP_NAME}  AS ad_group,
               s.impressions, s.clicks, s.cost_micros, s.conversions
        {TERM_FROM}
        WHERE s.date BETWEEN :start AND :end
        {TERM_FILTERS}
    ),
    combos AS (
        SELECT period, campaign_id, search_term, keyword, match_type, ad_group,
               SUM(cost_micros) AS cost,
               SUM(SUM(cost_micros)) OVER (
                   PARTITION BY period, campaign_id, search_term, keyword) AS keyword_cost
        FROM base
        GROUP BY 1, 2, 3, 4, 5, 6
    ),
    best AS (
        SELECT DISTINCT ON (period, campaign_id, search_term)
               period, campaign_id, search_term, keyword, match_type, ad_group
        FROM combos
        ORDER BY period, campaign_id, search_term,
                 keyword_cost DESC, keyword, cost DESC, match_type, ad_group
    ),
    totals AS (
        SELECT period, campaign_id, search_term,
               MAX(campaign)                   AS campaign,
               COUNT(DISTINCT keyword)         AS keywords,
               SUM(impressions)                AS impressions,
               SUM(clicks)                     AS clicks,
               SUM(cost_micros) / 1000000.0    AS spend,
               SUM(conversions)                AS conversions
        FROM base
        GROUP BY 1, 2, 3
    )
    SELECT t.period, t.campaign_id, t.campaign, t.search_term,
           b.keyword, b.match_type, b.ad_group, t.keywords,
           t.impressions, t.clicks, t.spend, t.conversions
    FROM totals t
    JOIN best b USING (period, campaign_id, search_term)
    """)


# All-time status per campaign + term. Loaded once and attached in pandas: joining it
# inside the filtered queries made Postgres pick a very slow plan.
STATUS_QUERY = text(f"WITH {STATUS_CTE} SELECT campaign_id, search_term, status FROM status")


TERMS_QUERY = terms_query(by_period=False)
TREND_TERMS_QUERY = terms_query(by_period=True)

# Campaign / ad group numbers per day, from the full ad group report.
CAMPAIGN_DAILY_QUERY = text(f"""
    SELECT s.date,
           {CAMPAIGN_NAME}                 AS campaign,
           {AD_GROUP_NAME}                 AS ad_group,
           SUM(s.impressions)              AS impressions,
           SUM(s.clicks)                   AS clicks,
           SUM(s.cost_micros) / 1000000.0  AS spend,
           SUM(s.conversions)              AS conversions
    FROM google_ads.ad_group_stats s
    LEFT JOIN google_ads.campaign_history c
           ON c.id = s.campaign_id AND c._fivetran_active
    LEFT JOIN google_ads.ad_group_history g
           ON g.id = s.id AND g._fivetran_active
    WHERE s.date BETWEEN :start AND :end
    GROUP BY 1, 2, 3
""")

# The same, summed from search terms. Used when a match-type or search filter is on,
# and for the split by match type.
TERM_DAILY_QUERY = text(f"""
    SELECT s.date,
           {CAMPAIGN_NAME}                 AS campaign,
           {AD_GROUP_NAME}                 AS ad_group,
           {MATCH_TYPE}                    AS match_type,
           SUM(s.impressions)              AS impressions,
           SUM(s.clicks)                   AS clicks,
           SUM(s.cost_micros) / 1000000.0  AS spend,
           SUM(s.conversions)              AS conversions
    {TERM_FROM}
    WHERE s.date BETWEEN :start AND :end
    {TERM_FILTERS}
    GROUP BY 1, 2, 3, 4
""")

# Keyword x match type, unaggregated by term, for the Search terms heat map.
KEYWORD_CELLS_QUERY = text(f"""
    SELECT {CAMPAIGN_NAME}                 AS campaign,
           {KEYWORD}                       AS keyword,
           {MATCH_TYPE}                    AS match_type,
           SUM(s.impressions)              AS impressions,
           SUM(s.clicks)                   AS clicks,
           SUM(s.cost_micros) / 1000000.0  AS spend,
           SUM(s.conversions)              AS conversions
    {TERM_FROM}
    WHERE s.date BETWEEN :start AND :end
    {TERM_FILTERS}
    GROUP BY 1, 2, 3
""")

# Terms with any spend in the heat map cells that were clicked.
CELL_TERMS_QUERY = text(f"""
    SELECT DISTINCT s.campaign_id, s.search_term
    FROM google_ads.search_term_keyword_stats s
    WHERE s.date BETWEEN :start AND :end
    {TERM_FILTERS}
      AND ({KEYWORD}, {MATCH_TYPE}) IN (
          SELECT * FROM UNNEST(CAST(:cell_keywords AS text[]),
                               CAST(:cell_match_types AS text[])))
""")

# Keyword volume per period from the Keywords report (matches the Google Ads Keywords screen).
KEYWORD_TREND_QUERY = text("""
    SELECT CASE WHEN k.date > :mid THEN 'current' ELSE 'prior' END  AS period,
           COALESCE(c.name, 'Campaign ' || k.campaign_id)           AS campaign,
           LOWER(TRIM(a.keyword_text))                              AS keyword,
           SUM(k.impressions)                                       AS impressions,
           SUM(k.clicks)                                            AS clicks,
           SUM(k.cost_micros) / 1000000.0                           AS spend,
           SUM(k.conversions)                                       AS conversions
    FROM google_ads.keyword_stats k
    JOIN google_ads.ad_group_criterion_history a
      ON a.id = k.ad_group_criterion_criterion_id
     AND a.ad_group_id = k.ad_group_id
     AND a._fivetran_active
    LEFT JOIN google_ads.campaign_history c
           ON c.id = k.campaign_id AND c._fivetran_active
    WHERE k.date BETWEEN :start AND :end
      AND a.keyword_text IS NOT NULL
    GROUP BY 1, 2, 3
""")

# Fallback: keyword volume summed from search terms (when match-type or search filter is on).
KEYWORD_TREND_TERMS_QUERY = text(f"""
    SELECT CASE WHEN s.date > :mid THEN 'current' ELSE 'prior' END  AS period,
           {CAMPAIGN_NAME}                                          AS campaign,
           LOWER(TRIM(s.info_text))                                 AS keyword,
           SUM(s.impressions)                                       AS impressions,
           SUM(s.clicks)                                            AS clicks,
           SUM(s.cost_micros) / 1000000.0                           AS spend,
           SUM(s.conversions)                                       AS conversions
    {TERM_FROM}
    WHERE s.date BETWEEN :start AND :end
      AND s.info_text IS NOT NULL
    {TERM_FILTERS}
    GROUP BY 1, 2, 3
""")

# Every term added or negated, all time. One row per campaign + term; match type and
# ad group are from the term's most recent row.
ADDED_NEGATED_QUERY = text(f"""
    WITH {STATUS_CTE},
    flagged AS MATERIALIZED (
        SELECT * FROM status WHERE status IN ('ADDED', 'EXCLUDED')
    )
    SELECT DISTINCT ON (s.campaign_id, s.search_term)
           s.campaign_id, s.search_term,
           {CAMPAIGN_NAME}  AS campaign,
           {AD_GROUP_NAME}  AS ad_group,
           {MATCH_TYPE}     AS match_type,
           f.status, f.last_seen
    FROM flagged f
    JOIN google_ads.search_term_keyword_stats s
      ON s.campaign_id = f.campaign_id AND s.search_term = f.search_term
    LEFT JOIN google_ads.campaign_history c
           ON c.id = s.campaign_id AND c._fivetran_active
    LEFT JOIN google_ads.ad_group_history g
           ON g.id = s.ad_group_id AND g._fivetran_active
    WHERE TRUE
    {TERM_FILTERS}
    ORDER BY s.campaign_id, s.search_term, s.date DESC,
             {AD_GROUP_NAME}, {MATCH_TYPE}
""")

DATE_BOUNDS_QUERY = text("""
    SELECT MIN(date) AS min_d, MAX(date) AS max_d
    FROM google_ads.search_term_keyword_stats
""")

MATCH_TYPES_QUERY = text(f"""
    SELECT DISTINCT {MATCH_TYPE} AS match_type
    FROM google_ads.search_term_keyword_stats s
""")

KEYWORD_QUERY = text("""
    SELECT DISTINCT info_text
    FROM google_ads.search_term_keyword_stats
    WHERE info_text IS NOT NULL
""")

# Conversion tracking changed on Aug 27, 2026; numbers before Aug 28 are inflated.
CONVERSION_CHANGE = dt.date(2026, 8, 28)

ACCOUNT_TZ = ZoneInfo("America/Los_Angeles")
TREND_DAYS = 30
SMALL_BASE = 10  # below this, show a multiple instead of a percent change

# Codes inside campaign names, e.g. "AMER-...-EN-B Brand UK Headlines".
BRAND_RE = re.compile(r"(?:^|[-\s])([BG])(?:\s|$)")
REGION_RE = re.compile(r"(?:^|[-\s])(AMER|EMEA|APAC|LATAM)(?=-|\s|$)")
REGION_ORDER = ["AMER", "EMEA", "APAC", "LATAM", "Other"]

NUM_COLS = ["impressions", "clicks", "spend", "conversions"]

# Heat map / chart options: label -> (column, legend format)
VIEW_OPTIONS = {
    "Spend": ("spend", "$,.0f"),
    "CPA": ("cpa", "$,.0f"),
    "Clicks": ("clicks", ",.0f"),
    "Impressions": ("impressions", ",.0f"),
}

# Words ignored when finding themes in search terms.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "my", "near", "me", "of", "on", "or", "the", "to",
    "vs", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}

LOOSE_MATCH = ("BROAD", "PHRASE", "NEAR_PHRASE")
EXACT_MATCH = ("EXACT", "NEAR_EXACT")

STATUS_LABELS = {
    "ADDED": "Added",
    "EXCLUDED": "Negated",
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
               "is never counted as wasted, since some are kept for visibility. Rules are "
               "judged on each term's total in a campaign, across all its keywords.")

SEARCH_TERMS_ONLY = ("Search terms only — lower than the total in Google Ads. Match type and "
                     "search filters only exist on search terms, so these numbers are summed "
                     "search terms, not the full campaign report.")

COL_CONFIG = {
    "search_term": "Search term",
    "keyword": st.column_config.TextColumn(
        "Keyword", help="The keyword with the most spend for this term in this campaign."),
    "keywords": st.column_config.NumberColumn(
        "Keywords", format="%d",
        help="How many keywords this term matched. Numbers are the total across all of them."),
    "match_type": "Match type",
    "campaign": "Campaign",
    "ad_group": "Ad group",
    "brand_type": "Type",
    "region": "Region",
    "status_label": st.column_config.TextColumn(
        "In Google Ads", help="All time, not just these dates. Negated anywhere in the "
                              "campaign wins over added."),
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


def read_sql(query, params=None):
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params=params or {})


def filter_params(match_types, search):
    """SQL parameters for the match-type and search filters."""
    escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return {"match_types": list(match_types), "search": search,
            "search_like": f"%{escaped}%"}


def clean_keyword(text_):
    return re.sub(r'[\[\]"+]', "", str(text_)).strip().lower()


def brand_type(name):
    """A standalone B (brand) or G (non-brand) code anywhere in the campaign name."""
    m = BRAND_RE.search(name) if isinstance(name, str) else None
    return {"B": "Brand", "G": "Non-brand"}[m.group(1)] if m else "Other"


def region(name):
    m = REGION_RE.search(name) if isinstance(name, str) else None
    return m.group(1) if m else "Other"


def numeric(df):
    df[NUM_COLS] = df[NUM_COLS].astype(float).fillna(0)
    return df


def tag_campaigns(df):
    df["brand_type"] = df["campaign"].map(brand_type)
    df["region"] = df["campaign"].map(region)
    return df


def label_status(df):
    df["status"] = df["status"].fillna("NONE").astype(str).str.upper()
    df["status_label"] = df["status"].map(STATUS_LABELS).fillna("Unknown")
    return df


def last_complete_day(end):
    """Today in the account's time zone is still syncing, so stop at yesterday."""
    yesterday = dt.datetime.now(ACCOUNT_TZ).date() - dt.timedelta(days=1)
    return min(end, yesterday)


@st.cache_data(ttl=3600, show_spinner=False)
def load_date_bounds():
    row = read_sql(DATE_BOUNDS_QUERY).iloc[0]
    return pd.to_datetime(row.min_d).date(), pd.to_datetime(row.max_d).date()


@st.cache_data(ttl=3600, show_spinner=False)
def load_match_types():
    return sorted(read_sql(MATCH_TYPES_QUERY)["match_type"])


@st.cache_data(ttl=3600, show_spinner=False)
def load_bid_keywords():
    kws = read_sql(KEYWORD_QUERY)["info_text"]
    return frozenset(clean_keyword(k) for k in kws)


@st.cache_data(ttl=3600, max_entries=5, show_spinner="Loading campaigns...")
def load_campaign_daily(start, end):
    df = read_sql(CAMPAIGN_DAILY_QUERY, {"start": start, "end": end})
    df = tag_campaigns(numeric(df))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_status():
    return read_sql(STATUS_QUERY)


def attach_status(df):
    """Add the all-time "In Google Ads" status; terms never flagged are "not added"."""
    df = df.merge(load_status(), on=["campaign_id", "search_term"], how="left")
    return label_status(df)


@st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading search terms...")
def load_terms(start, end, match_types, search):
    df = read_sql(TERMS_QUERY, {"start": start, "end": end,
                                **filter_params(match_types, search)})
    return tag_campaigns(attach_status(numeric(df)))


@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def load_term_daily(start, end, match_types, search):
    df = read_sql(TERM_DAILY_QUERY, {"start": start, "end": end,
                                     **filter_params(match_types, search)})
    df = tag_campaigns(numeric(df))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def load_keyword_cells(start, end, match_types, search):
    df = read_sql(KEYWORD_CELLS_QUERY, {"start": start, "end": end,
                                        **filter_params(match_types, search)})
    return tag_campaigns(numeric(df))


@st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
def load_cell_terms(start, end, match_types, search, cells):
    df = read_sql(CELL_TERMS_QUERY, {
        "start": start, "end": end, **filter_params(match_types, search),
        "cell_keywords": [k for k, _ in cells], "cell_match_types": [m for _, m in cells]})
    return set(zip(df["campaign_id"], df["search_term"]))


@st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading trends...")
def load_trends(t_end, match_types, search):
    """Two back-to-back 30-day periods ending on t_end: term rows and keyword rows."""
    mid = t_end - dt.timedelta(days=TREND_DAYS)
    t_start = t_end - dt.timedelta(days=2 * TREND_DAYS - 1)
    params = {"start": t_start, "mid": mid, "end": t_end, **filter_params(match_types, search)}
    terms_ = tag_campaigns(attach_status(numeric(read_sql(TREND_TERMS_QUERY, params))))
    kw_query = KEYWORD_TREND_TERMS_QUERY if (match_types or search) else KEYWORD_TREND_QUERY
    keywords_ = tag_campaigns(numeric(read_sql(kw_query, params)))
    return terms_, keywords_, t_start, mid


@st.cache_data(ttl=3600, max_entries=10, show_spinner="Loading added and negated terms...")
def load_added_negated(match_types, search):
    df = read_sql(ADDED_NEGATED_QUERY, filter_params(match_types, search))
    df = tag_campaigns(label_status(df))
    df["last_seen"] = pd.to_datetime(df["last_seen"]).dt.date
    return df


def load_failed(what):
    """Plain message for users; details go to the log, not the screen."""
    log.exception("Couldn't load %s", what)
    st.error(f"Couldn't load {what}. Try again in a minute. If it keeps happening, "
             f"check the app logs.")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def add_rates(df):
    df["cpc"] = (df["spend"] / df["clicks"]).where(df["clicks"] > 0)
    df["cpa"] = (df["spend"] / df["conversions"]).where(df["conversions"] > 0)
    return df


def summarize(df, by):
    """Roll rows up to campaigns, ad groups, etc."""
    out = df.groupby(by, dropna=False)[NUM_COLS].sum().reset_index()
    out = add_rates(out)
    return out.sort_values(["spend"] + by, ascending=[False] + [True] * len(by))


def by_campaign(df):
    if allowed_campaigns is not None:
        df = df[df["campaign"].isin(allowed_campaigns)]
    return df


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
              & (df["status"] != "EXCLUDED")]


def recommended_adds(df):
    rules, logic = saved_rules("add", ADD_DEFAULTS)
    # Never recommend adding something already added or already a keyword.
    return df[apply_rules(df, rules, logic)
              & ~df["is_keyword"]
              & ~df["match_type"].isin(EXACT_MATCH)
              & (df["status"] != "ADDED")]


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


def heatmap(df, x, y, view, x_title, x_sort=None, key=None, selectable=False):
    """Simple heat map colored by a VIEW_OPTIONS label. Returns clicked cells when selectable."""
    color, fmt = VIEW_OPTIONS[view]
    sel = alt.selection_point(name="cell", fields=[x, y]) if selectable else None
    chart = alt.Chart(df).mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
        x=alt.X(f"{x}:N", title=x_title, sort=x_sort,
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y(f"{y}:N", title=None,
                sort=alt.EncodingSortField(field=color, op="sum", order="descending")),
        color=alt.Color(f"{color}:Q", title=view, scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(format=fmt)),
        tooltip=[
            alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:N"),
            alt.Tooltip("impressions:Q", title="Impressions", format=",.0f"),
            alt.Tooltip("clicks:Q", title="Clicks", format=",.0f"),
            alt.Tooltip("spend:Q", title="Spend", format="$,.2f"),
            alt.Tooltip("conversions:Q", title="Conversions", format=",.1f"),
            alt.Tooltip("cpa:Q", title="CPA", format="$,.2f"),
        ],
    ).properties(height=max(250, 30 * df[y].nunique()))
    if selectable:
        chart = chart.encode(opacity=alt.condition(sel, alt.value(1), alt.value(0.3))).add_params(sel)
        event = st.altair_chart(chart, on_select="rerun", key=key)
        return event.get("selection", {}).get("cell", []) if event else []
    st.altair_chart(chart)
    return []


def change_text(cur, prior):
    """Percent change, or a multiple when the prior value is too small for a percent to mean much."""
    if prior <= 0:
        return "New" if cur > 0 else "–"
    if prior < SMALL_BASE:
        ratio = cur / prior
        return f"{ratio:,.0f}× (small base)" if ratio >= 10 else f"{ratio:,.1f}× (small base)"
    return f"{(cur - prior) / prior * 100:+,.0f}%"


def change_column(cur, prior):
    return [change_text(c, p) for c, p in zip(cur, prior)]


def themes(search_terms, weights):
    """Count one- and two-word phrases across search terms, weighted (e.g. by clicks)."""
    counts = Counter()
    for term, w in zip(search_terms, weights):
        words = [x for x in re.findall(r"[a-z0-9]+", str(term).lower()) if x not in STOPWORDS]
        phrases = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}
        for p in phrases:
            counts[p] += w
    return counts


def prep_terms(df):
    df = add_rates(by_campaign(df).copy())
    df["is_keyword"] = df["search_term"].str.lower().isin(bid_keywords)
    return df


def search_terms_note():
    if terms_only:
        st.info(SEARCH_TERMS_ONLY)


# ---------------------------------------------------------------
# Pages
# ---------------------------------------------------------------
def overview_page():
    st.title("Overview")
    st.caption(f"{start:%b %d, %Y} to {end:%b %d, %Y}")

    if start < CONVERSION_CHANGE:
        st.warning(f"Conversion tracking changed on Aug 27, 2026. Conversions before "
                   f"{CONVERSION_CHANGE:%b %d} are inflated, so set the start date to "
                   f"{CONVERSION_CHANGE:%b %d} or later for accurate CPA.")
    search_terms_note()

    negatives = recommended_negatives(terms)
    spend, conv = campaign_daily["spend"].sum(), campaign_daily["conversions"].sum()
    clicks = campaign_daily["clicks"].sum()
    wasted = negatives["spend"].sum()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spend", f"${spend:,.0f}")
    m2.metric("Conversions", f"{conv:,.1f}")
    m3.metric("Cost per conversion", f"${spend / conv:,.0f}" if conv else "–")
    m4.metric("Avg. CPC", f"${spend / clicks:,.2f}" if clicks else "–",
              help="Spend divided by clicks. Use the Campaign type filter to compare "
                   "brand and non-brand.")
    m5.metric("Wasted spend", f"${wasted:,.0f}",
              f"{len(negatives):,} terms to negate", delta_color="off", help=WASTED_HELP)
    if not terms_only and spend:
        term_spend = terms["spend"].sum()
        st.caption(f"Spend, conversions, CPA and CPC are from the full campaign report and "
                   f"match Google Ads. The search terms report covers ${term_spend:,.0f} "
                   f"({term_spend / spend:.0%}) of this spend; Google hides low-volume terms "
                   f"and video campaigns have none. Wasted spend comes from search terms.")

    st.subheader("Weekly volume")
    c_metric, c_split = st.columns(2)
    vol = c_metric.radio("Show", ["Clicks", "Impressions"], horizontal=True, key="ov_vol")
    split = c_split.radio("Split by", ["Total", "Match type"], horizontal=True, key="ov_split")
    col = vol.lower()
    vday = campaign_daily if split == "Total" else term_daily
    vday = vday.assign(week=vday["date"].dt.to_period("W").dt.start_time)
    by = ["week"] if split == "Total" else ["week", "match_type"]
    trend = vday.groupby(by)[[col]].sum().reset_index().sort_values(by)
    if trend.empty:
        st.info("No data for the current filters.")
    else:
        trend["week_label"] = trend["week"].dt.strftime("%b %d")
        weeks = list(trend["week_label"].unique())
        enc_color = (alt.value("#1f77b4") if split == "Total"
                     else alt.Color("match_type:N", title="Match type"))
        tips = [alt.Tooltip("week_label:N", title="Week starting"),
                alt.Tooltip(f"{col}:Q", title=vol, format=",.0f")]
        if split == "Match type":
            tips.append(alt.Tooltip("match_type:N", title="Match type"))
        chart = alt.Chart(trend).mark_bar().encode(
            x=alt.X("week_label:N", title="Week starting", sort=weeks,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{col}:Q", title=vol, stack=True),
            color=enc_color,
            tooltip=tips,
        ).properties(height=300)
        st.altair_chart(chart)
        note = ("Weeks start on Monday. The first and last weeks may be partial, "
                "depending on your date range.")
        if split == "Match type" and not terms_only:
            note += (" Match type only exists on search terms, so this split is summed search "
                     "terms and is lower than the total in Google Ads.")
        st.caption(note)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top campaigns by spend")
        camp = summarize(campaign_daily, ["campaign", "brand_type", "region"]).head(10)
        st.dataframe(camp[["campaign", "brand_type", "region", "impressions", "clicks", "spend",
                           "conversions", "cpa"]],
                     hide_index=True, column_config=COL_CONFIG)
    with c2:
        st.subheader("Top terms to negate", help=WASTED_HELP)
        top_neg = negatives.sort_values(["spend", "search_term"], ascending=[False, True]).head(10)
        st.dataframe(top_neg[["search_term", "campaign", "impressions", "clicks", "spend",
                              "conversions"]],
                     hide_index=True, column_config=COL_CONFIG)


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
    hm_view = st.radio("Color the heat map by", list(VIEW_OPTIONS), horizontal=True,
                       help="CPA = spend divided by conversions. Darker = higher. "
                            "Impressions shows terms that are very visible but may not "
                            "be getting clicks.")
    try:
        cells = by_campaign(load_keyword_cells(start, end, mt_key, search))
    except Exception:
        load_failed("the keyword heat map")
        cells = pd.DataFrame(columns=["keyword", "match_type"] + NUM_COLS)
    hm = summarize(cells, ["keyword", "match_type"])
    metric = VIEW_OPTIONS[hm_view][0]
    rank_by = metric if metric in ("clicks", "impressions") else "spend"
    hm = hm[hm[metric] > 0]
    top_kw = (hm.groupby("keyword")[rank_by].sum().reset_index()
              .sort_values([rank_by, "keyword"], ascending=[False, True])
              .head(20)["keyword"])
    hm = hm[hm["keyword"].isin(top_kw)]

    picked = []
    if hm.empty:
        st.info("Nothing to show for the current filters.")
    else:
        picked = heatmap(hm, "match_type", "keyword", hm_view, "Match type",
                         key="kw_heatmap", selectable=True)
        st.caption(f"Each cell is one keyword at one match type (search terms report). Top 20 "
                   f"keywords by {rank_by}. Click a cell to show the search terms with spend "
                   f"in it; click empty space to clear.")
    if picked:
        keys = tuple(sorted({(p.get("keyword"), p.get("match_type")) for p in picked}))
        try:
            in_cells = load_cell_terms(start, end, mt_key, search, keys)
            view = view[[(c, t) in in_cells
                         for c, t in zip(view["campaign_id"], view["search_term"])]]
            st.info("Filtered to terms with spend in: "
                    + ", ".join(f"{k} ({m})" for k, m in keys)
                    + ". Each term's numbers are still its total across all keywords.")
        except Exception:
            load_failed("the terms for that cell")

    negatives = recommended_negatives(view).sort_values(["spend", "search_term"],
                                                        ascending=[False, True])
    adds = recommended_adds(view).sort_values(["conversions", "search_term"],
                                              ascending=[False, True])

    cols = ["search_term", "status_label", "why", "keyword", "keywords", "match_type",
            "campaign", "ad_group", "impressions", "clicks", "spend", "cpc", "conversions",
            "cpa", "suggestion"]

    tab_neg, tab_add, tab_all, tab_trend = st.tabs([
        f"Recommended negatives ({len(negatives)})",
        f"Recommended adds ({len(adds)})",
        f"All search terms ({len(view)})",
        "Trends (30-day)",
    ])
    with tab_trend:
        trends_section()
    with tab_neg:
        st.metric("Wasted spend", f"${negatives['spend'].sum():,.2f}", help=WASTED_HELP)
        st.dataframe(negatives[cols], hide_index=True,
                     column_config=COL_CONFIG)
        download(negatives, cols, "recommended_negatives.csv")
    with tab_add:
        st.dataframe(adds[cols], hide_index=True,
                     column_config=COL_CONFIG)
        download(adds, cols, "recommended_adds.csv")
    with tab_all:
        st.caption("One row per search term per campaign. Numbers are the term's total across "
                   "all the keywords and ad groups it matched.")
        all_rows = view.sort_values(["clicks", "spend", "search_term"],
                                    ascending=[False, False, True])
        st.dataframe(all_rows[cols], hide_index=True,
                     column_config=COL_CONFIG)
        download(all_rows, cols, "all_search_terms.csv")


def trends_section():
    """Month-over-month view: keyword volume and themes in recommended negatives."""
    t_end = last_complete_day(end)
    try:
        tr, kw_rows, t_start, mid = load_trends(t_end, mt_key, search)
    except Exception:
        load_failed("trends")
        return
    tr = prep_terms(tr)
    kw_rows = by_campaign(kw_rows)
    cur_label = f"{mid + dt.timedelta(days=1):%b %d} – {t_end:%b %d}"
    prior_label = f"{t_start:%b %d} – {mid:%b %d}"
    st.caption(f"Last 30 complete days ({cur_label}) vs. the 30 days before ({prior_label}). "
               f"Ends on the last full day of your date range (today is left out because it's "
               f"still syncing); the other filters apply.")
    if t_start < min_d:
        st.warning(f"Search term data starts on {min_d:%b %d}, so the earlier period is "
                   f"missing days and its numbers will look low.")
    if t_start < CONVERSION_CHANGE:
        st.warning("The earlier period includes days before the Aug 27 conversion tracking "
                   "change, so conversion-based negatives rules may differ between periods.")

    vol = st.radio("Measure by", ["Clicks", "Impressions"], horizontal=True, key="tr_vol")
    col = vol.lower()

    # --- Keywords going up or down ---
    st.subheader("Keywords going up or down")
    if terms_only:
        st.info(SEARCH_TERMS_ONLY)
    else:
        st.caption("From the Keywords report. Compare with the Google Ads Keywords screen.")
    kw = kw_rows.pivot_table(index="keyword", columns="period", values=["clicks", "impressions"],
                             aggfunc="sum", fill_value=0)
    if kw.empty:
        st.info("No keyword data for these periods.")
    else:
        kw.columns = [f"{m}_{p}" for m, p in kw.columns]
        for m in ("clicks", "impressions"):
            for p in ("current", "prior"):
                kw[f"{m}_{p}"] = kw.get(f"{m}_{p}", 0)
            kw[f"{m}_change"] = kw[f"{m}_current"] - kw[f"{m}_prior"]
            kw[f"{m}_pct"] = change_column(kw[f"{m}_current"], kw[f"{m}_prior"])
        kw = kw.reset_index()
        kw["abs_change"] = kw[f"{col}_change"].abs()
        movers = (kw[kw[f"{col}_change"] != 0]
                  .sort_values(["abs_change", "keyword"], ascending=[False, True]).head(15))
        if not movers.empty:
            movers["direction"] = np.where(movers[f"{col}_change"] > 0, "Up", "Down")
            chart = alt.Chart(movers).mark_bar().encode(
                x=alt.X(f"{col}_change:Q", title=f"Change in {vol.lower()}"),
                y=alt.Y("keyword:N", title=None, sort="-x"),
                color=alt.Color("direction:N", title=None,
                                scale=alt.Scale(domain=["Up", "Down"],
                                                range=["#2e7d32", "#c62828"])),
                tooltip=[alt.Tooltip("keyword:N", title="Keyword"),
                         alt.Tooltip(f"{col}_prior:Q", title="Prior 30 days", format=",.0f"),
                         alt.Tooltip(f"{col}_current:Q", title="Last 30 days", format=",.0f"),
                         alt.Tooltip(f"{col}_pct:N", title="Change")],
            ).properties(height=max(250, 26 * len(movers)))
            st.altair_chart(chart)
            st.caption(f"The 15 keywords with the biggest change in {vol.lower()}. When the "
                       f"earlier period had fewer than {SMALL_BASE}, the change is shown as a "
                       f"multiple, since a percent on a tiny number is misleading.")
        kw_cols = ["keyword", "clicks_prior", "clicks_current", "clicks_pct",
                   "impressions_prior", "impressions_current", "impressions_pct"]
        st.dataframe(kw.sort_values([f"{col}_change", "keyword"], ascending=[False, True])[kw_cols],
                     hide_index=True, column_config={
                         "keyword": "Keyword",
                         "clicks_prior": st.column_config.NumberColumn("Clicks (prior)", format="%d"),
                         "clicks_current": st.column_config.NumberColumn("Clicks (last 30)", format="%d"),
                         "clicks_pct": "Clicks change",
                         "impressions_prior": st.column_config.NumberColumn("Impr. (prior)", format="%d"),
                         "impressions_current": st.column_config.NumberColumn("Impr. (last 30)", format="%d"),
                         "impressions_pct": "Impr. change",
                     })
        download(kw, kw_cols, "keyword_trends.csv")

    # --- Themes in recommended negatives ---
    st.subheader("Top themes in recommended negatives",
                 help="Words and two-word phrases that show up across the recommended "
                      "negatives in each period, weighted by the measure above. Uses the "
                      "negatives rules from the sidebar.")
    rows = []
    for period in ("prior", "current"):
        neg = recommended_negatives(tr[tr["period"] == period])
        for phrase, val in themes(neg["search_term"], neg[col]).items():
            rows.append({"theme": phrase, "period": period, "value": val})
    th = pd.DataFrame(rows)
    if th.empty:
        st.info("No recommended negatives in these periods.")
        return
    th = th.pivot_table(index="theme", columns="period", values="value",
                        aggfunc="sum", fill_value=0)
    for p in ("current", "prior"):
        th[p] = th.get(p, 0)
    th["change"] = th["current"] - th["prior"]
    th["pct"] = change_column(th["current"], th["prior"])
    th = th.reset_index().sort_values(["current", "theme"], ascending=[False, True])
    top = th.head(15)
    long = top.melt(id_vars="theme", value_vars=["prior", "current"],
                    var_name="period", value_name="value")
    long["period"] = long["period"].map({"prior": "Prior 30 days", "current": "Last 30 days"})
    chart = alt.Chart(long).mark_bar().encode(
        y=alt.Y("theme:N", title=None, sort=list(top["theme"])),
        x=alt.X("value:Q", title=vol),
        yOffset=alt.YOffset("period:N", sort=["Prior 30 days", "Last 30 days"]),
        color=alt.Color("period:N", title=None, sort=["Prior 30 days", "Last 30 days"],
                        scale=alt.Scale(range=["#b0bec5", "#1f77b4"])),
        tooltip=[alt.Tooltip("theme:N", title="Theme"), alt.Tooltip("period:N", title="Period"),
                 alt.Tooltip("value:Q", title=vol, format=",.0f")],
    ).properties(height=max(300, 40 * len(top)))
    st.altair_chart(chart)
    st.caption("The 15 biggest themes in the last 30 days, next to the 30 days before.")
    th_cols = ["theme", "prior", "current", "pct"]
    st.dataframe(th[th_cols], hide_index=True, column_config={
        "theme": "Theme",
        "prior": st.column_config.NumberColumn(f"{vol} (prior)", format="%d"),
        "current": st.column_config.NumberColumn(f"{vol} (last 30)", format="%d"),
        "pct": "Change",
    })
    download(th, th_cols, "negative_themes.csv")


def campaigns_page():
    st.title("Campaigns")
    search_terms_note()

    st.subheader("Where's the money going?")
    hm_view = st.radio("Color by", list(VIEW_OPTIONS), horizontal=True, key="camp_hm")
    wk = campaign_daily.assign(week=campaign_daily["date"].dt.to_period("W").dt.start_time)
    wk = wk.groupby(["campaign", "week"])[NUM_COLS].sum().reset_index()
    wk = add_rates(wk)
    metric = VIEW_OPTIONS[hm_view][0]
    rank_by = metric if metric in ("clicks", "impressions") else "spend"
    wk = wk[wk[metric] > 0]
    top = (wk.groupby("campaign")[rank_by].sum().reset_index()
           .sort_values([rank_by, "campaign"], ascending=[False, True])
           .head(25)["campaign"])
    wk = wk[wk["campaign"].isin(top)].sort_values("week")
    wk["week_label"] = wk["week"].dt.strftime("%b %d")
    if wk.empty:
        st.info("No data for the current filters.")
    else:
        heatmap(wk, "week_label", "campaign", hm_view, "Week starting",
                x_sort=list(wk["week_label"].unique()))
        st.caption(f"Each cell is one campaign in one week. Hover for details. "
                   f"Showing the top 25 campaigns by {rank_by}.")

    st.subheader("By region")
    split_type = st.toggle("Split by campaign type", key="region_split",
                           help="Brand vs. non-brand within each region.")
    reg_by = ["region", "brand_type"] if split_type else ["region"]
    reg = summarize(campaign_daily, reg_by)
    total_spend = reg["spend"].sum()
    reg["share"] = (reg["spend"] / total_spend * 100) if total_spend else np.nan
    reg_cols = reg_by + ["impressions", "clicks", "spend", "share", "cpc", "conversions", "cpa"]
    st.dataframe(reg[reg_cols], hide_index=True, column_config={
        **COL_CONFIG,
        "share": st.column_config.NumberColumn("% of spend", format="%.1f%%"),
    })
    download(reg, reg_cols, "regions.csv")
    st.caption("Region comes from AMER, EMEA, APAC or LATAM in the campaign name; "
               "campaigns without one show as Other. Respects the sidebar filters.")

    st.subheader("All campaigns")
    if not terms_only:
        st.caption("Compare with the Google Ads Campaigns screen for the same dates.")
    camp = summarize(campaign_daily, ["campaign", "brand_type", "region"])
    cols = ["campaign", "brand_type", "region", "impressions", "clicks", "spend", "cpc",
            "conversions", "cpa"]
    st.dataframe(camp[cols], hide_index=True, column_config=COL_CONFIG)
    download(camp, cols, "campaigns.csv")

    st.subheader("Drill into a campaign")
    choice = st.selectbox("Campaign", camp["campaign"].dropna().tolist(), index=None,
                          placeholder="Pick a campaign to see its ad groups...")
    if choice:
        ag = summarize(campaign_daily[campaign_daily["campaign"] == choice], ["ad_group"])
        ag_cols = ["ad_group", "impressions", "clicks", "spend", "cpc", "conversions", "cpa"]
        st.dataframe(ag[ag_cols], hide_index=True,
                     column_config=COL_CONFIG)


def added_negated_page():
    st.title("Added & negated")
    st.caption(f"Every search term marked added or negated in Google Ads, as far back as this "
               f"database goes ({min_d:%b %d, %Y}). Not limited by the date range; the other "
               f"filters still apply. One row per term per campaign. If a term was ever "
               f"negated in a campaign, it shows as negated there, even if it was also added.")

    try:
        hist = by_campaign(load_added_negated(mt_key, search))
    except Exception:
        load_failed("added and negated terms")
        return

    negated = hist[hist["status"] == "EXCLUDED"]
    added = hist[hist["status"] == "ADDED"]
    cols = ["search_term", "match_type", "campaign", "ad_group", "brand_type", "region",
            "last_seen"]

    tab_neg, tab_add = st.tabs([f"Negated ({len(negated)})", f"Added ({len(added)})"])
    with tab_neg:
        st.dataframe(negated.sort_values(["search_term", "campaign"])[cols], hide_index=True,
                     column_config=COL_CONFIG)
        download(negated, cols, "negated_terms.csv")
    with tab_add:
        st.dataframe(added.sort_values(["search_term", "campaign"])[cols], hide_index=True,
                     column_config=COL_CONFIG)
        download(added, cols, "added_terms.csv")


def how_numbers_work_page():
    st.title("How numbers work")
    st.markdown("""
The app reads a copy of Google Ads data synced by Fivetran. It only recommends changes;
it never writes to Google Ads.

| Number | Comes from (Google report) | Compare with in Google Ads |
|---|---|---|
| Overview spend, conversions, cost per conversion, CPC | Ad group report (`ad_group_stats`) | Campaigns screen, total row |
| Weekly volume (Total) | Ad group report | Campaigns screen, by week |
| Campaign table, campaign heat map | Ad group report | Campaigns screen |
| Ad group drill-down | Ad group report | Ad groups screen |
| Weekly volume split by match type | Search terms report | Search terms screen |
| Search term lists, recommendations, wasted spend | Search terms report (`search_term_keyword_stats`) | Search terms screen |
| Keyword trends | Keywords report (`keyword_stats`) | Keywords screen |
| Themes in recommended negatives | Search terms report | Search terms screen |
| Added & negated, "In Google Ads" | Search terms report status, all time | Search terms screen, "Added/Excluded" column |

**Why search-term totals are lower than campaign totals.** Google leaves low-volume search
terms out of the search terms report for privacy, and video (YouTube) campaigns have no
search terms at all. So anything built from search terms covers only part of spend.

**When a match type or search filter is on**, every number comes from search terms, because
only search terms have those fields. Pages show a note when this happens.

**Search terms are judged per campaign.** A term can match several keywords and ad groups.
The app adds those together into one row per term per campaign before applying your rules,
and shows the keyword with the most spend.

**"In Google Ads" status** looks at every date in the database, not just your date range.
Negated anywhere in the campaign wins; otherwise added; otherwise not added.

**Trends** compare the last 30 complete days with the 30 days before. Today is left out
because it's still syncing (account time zone: Pacific).

**Campaign type and region** come from codes in the campaign name: a standalone `B` (brand)
or `G` (non-brand), and `AMER`, `EMEA`, `APAC` or `LATAM`.

**Conversions before Aug 28, 2026** are inflated because conversion tracking changed on
Aug 27.
""")


# ---------------------------------------------------------------
# Navigation + shared filter sidebar (applies to every page)
# ---------------------------------------------------------------
pg = st.navigation([
    st.Page(overview_page, title="Overview", default=True),
    st.Page(campaigns_page, title="Campaigns"),
    st.Page(search_terms_page, title="Search terms"),
    st.Page(added_negated_page, title="Added & negated"),
    st.Page(how_numbers_work_page, title="How numbers work"),
])

st.sidebar.header("Filter")

try:
    min_d, max_d = load_date_bounds()
except Exception:
    load_failed("data from the database")
    st.stop()

default_start = min(max(min_d, CONVERSION_CHANGE), max_d)
date_range = st.sidebar.date_input("Date range", (default_start, max_d),
                                   min_value=min_d, max_value=max_d,
                                   help="Starts on Aug 28 by default because conversion "
                                        "tracking changed on Aug 27.")
start, end = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d)

try:
    campaign_raw = load_campaign_daily(start, end)
    match_type_options = load_match_types()
    bid_keywords = load_bid_keywords()
except Exception:
    load_failed("data")
    st.stop()

brand = st.sidebar.radio("Campaign type", ["All", "Brand", "Non-brand"], horizontal=True,
                         help="From a standalone code in the campaign name: B = brand, "
                              "G = non-brand.")
region_options = [r for r in REGION_ORDER if r in set(campaign_raw["region"])]
regions = st.sidebar.multiselect("Regions", region_options, placeholder="All regions",
                                 help="From AMER, EMEA, APAC or LATAM in the campaign name.")

camp_types = campaign_raw[["campaign", "brand_type", "region"]].drop_duplicates()
if brand != "All":
    camp_types = camp_types[camp_types["brand_type"] == brand]
if regions:
    camp_types = camp_types[camp_types["region"].isin(regions)]
campaign_options = sorted(camp_types["campaign"].dropna().unique())
# Drop picked campaigns that the type or region filter no longer allows.
if "campaigns" in st.session_state:
    st.session_state["campaigns"] = [c for c in st.session_state["campaigns"]
                                     if c in campaign_options]
campaigns = st.sidebar.multiselect("Campaigns", campaign_options, placeholder="All campaigns",
                                   key="campaigns")
match_types = st.sidebar.multiselect("Match types", match_type_options,
                                     placeholder="All match types")
search = st.sidebar.text_input("Search terms or keywords", placeholder="e.g. etl, pricing").strip()

allowed_campaigns = None
if brand != "All" or regions or campaigns:
    allowed_campaigns = set(campaigns) if campaigns else set(campaign_options)

mt_key = tuple(sorted(match_types))
terms_only = bool(match_types or search)

try:
    raw = load_terms(start, end, mt_key, search)
    term_daily_raw = load_term_daily(start, end, mt_key, search)
except Exception:
    load_failed("search terms")
    st.stop()

terms = prep_terms(raw)
term_daily = by_campaign(term_daily_raw)
# Campaign-level numbers: the full ad group report, unless a search-term-only filter is on.
campaign_daily = term_daily if terms_only else by_campaign(campaign_raw)

pg.run()
