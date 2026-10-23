import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

engine = create_engine(st.secret["URL"])
query = """
SELECT
    s.date,
    s.campaign_id,
    ag.campaign_name,
    s.ad_group_id,
    ag.name                    AS ad_group_name,
    s.search_term,
    s.info_text                AS keyword,
    s.search_term_match_type,
    s.status,
    s.impressions,
    s.clicks,
    s.cost_micros / 1e6        AS spend,
    s.conversions,
    s.conversions_value
FROM google_ads.search_term_keyword_stats s
LEFT JOIN google_ads.ad_group_history ag
    ON ag.id = s.ad_group_id
   AND ag.campaign_id = s.campaign_id
   AND ag._fivetran_active
"""


st.set_page_config(page_title="Search Terms Review", layout="wide")
st.title("Search Terms Review")


@st.cache_data
def load():
    d = pd.read_sql(query, engine)
    d["date"] = pd.to_datetime(d["date"])
    return d


try:
    df = load()
except FileNotFoundError:
    st.error(f"Can't find {DATA_FILE}. Export it from your notebook with df.to_pickle('{DATA_FILE}').")
    st.stop()

if "relevant_terms" not in st.session_state:
    st.session_state.relevant_terms = set()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Date range")
    st.caption("Applies to all tabs.")
    dates = st.date_input("Dates", value=(df["date"].min(), df["date"].max()))

    with st.form("thresholds"):
        st.subheader("Recommended negatives")
        st.caption("Zero conversions AND (spend ≥ min spend OR clicks ≥ min clicks). Terms marked relevant are skipped.")
        neg_spend = st.number_input("Min spend ($)", value=100.0)
        neg_clicks = st.number_input("Min clicks", value=10)

        st.subheader("Recommended adds")
        st.caption("Conversions ≥ min conversions AND CPA ≤ max CPA.")
        add_conv = st.number_input("Min conversions", value=1.0)
        add_cpa = st.number_input("Max CPA ($)", value=200.0)

        st.form_submit_button("Apply")

if not isinstance(dates, tuple) or len(dates) != 2:
    st.info("Pick a start and end date.")
    st.stop()

start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
d = df[(df["date"] >= start) & (df["date"] <= end)]

if d.empty:
    st.warning("No search terms in this date range.")
    st.stop()

# ---------- Summarize ----------
terms = (
    d.groupby(["search_term", "keyword", "campaign_name", "ad_group_name", "status"], dropna=False)
    .agg(
        days_seen=("date", "nunique"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        spend=("spend", "sum"),
        conversions=("conversions", "sum"),
    )
    .reset_index()
)
terms["cpc"] = (terms["spend"] / terms["clicks"].replace(0, float("nan"))).round(2)
terms["cpa"] = (terms["spend"] / terms["conversions"].replace(0, float("nan"))).round(2)
terms["spend"] = terms["spend"].round(2)

metrics = st.container()
tab_neg, tab_add, tab_all = st.tabs(["Recommended negatives", "Recommended adds", "All search terms"])

# ---------- All terms + relevance ----------
with tab_all:
    st.caption("Tick 'relevant' to keep a term out of the negatives list.")
    sort_by = st.selectbox("Sort by", ["impressions", "days_seen", "clicks", "spend", "conversions", "cpc", "cpa"])
    terms["relevant"] = terms["search_term"].isin(st.session_state.relevant_terms)
    shown = terms.sort_values(sort_by, ascending=False, na_position="last")[
        ["relevant", "search_term", "keyword", "campaign_name", "ad_group_name",
         "days_seen", "impressions", "clicks", "spend", "conversions", "cpc", "cpa"]
    ]
    edited = st.data_editor(
        shown,
        hide_index=True,
        width="stretch",
        disabled=[c for c in shown.columns if c != "relevant"],
        key="terms_editor",
    )
    st.session_state.relevant_terms = (
        st.session_state.relevant_terms - set(shown["search_term"])
    ) | set(edited.loc[edited["relevant"], "search_term"])

terms["relevant"] = terms["search_term"].isin(st.session_state.relevant_terms)

# Terms already added as keywords or excluded don't need recommendations
open_terms = terms[~terms["status"].isin(["ADDED", "EXCLUDED"])]

# ---------- Recommendations ----------
negatives = open_terms[
    (open_terms["conversions"] == 0)
    & ((open_terms["spend"] >= neg_spend) | (open_terms["clicks"] >= neg_clicks))
    & ~open_terms["relevant"]
].sort_values("spend", ascending=False)

adds = open_terms[
    (open_terms["conversions"] >= add_conv)
    & (open_terms["cpa"].fillna(float("inf")) <= add_cpa)
    & (open_terms["search_term"].str.lower() != open_terms["keyword"].str.lower())
].sort_values("conversions", ascending=False)

with metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Search terms", f"{len(terms):,}")
    c2.metric("Recommended negatives", len(negatives))
    c3.metric("Negatives' spend", f"${negatives['spend'].sum():,.0f}")
    c4.metric("Recommended adds", len(adds))

report_cols = ["search_term", "keyword", "campaign_name", "ad_group_name", "clicks", "spend", "conversions", "cpa"]

with tab_neg:
    st.dataframe(negatives[report_cols], hide_index=True, width="stretch")
    st.download_button("Download CSV", negatives[report_cols].to_csv(index=False), "negatives.csv", key="dl_neg")

with tab_add:
    st.dataframe(adds[report_cols], hide_index=True, width="stretch")
    st.download_button("Download CSV", adds[report_cols].to_csv(index=False), "adds.csv", key="dl_add")
