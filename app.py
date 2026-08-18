"""
CIVIC EVIDENCE STUDIO — functional Streamlit prototype
======================================================
From raw community engagement comments to transparent, human-reviewed evidence
connected to planning decisions.

Workflow:
  RAW ENGAGEMENT FILES -> METADATA + CONTEXT -> QUANTITATIVE ANALYSIS
  -> AI-ASSISTED THEMATIC EXPLORATION -> HUMAN REVIEW -> EVIDENCE LIBRARY
  alongside PROJECT CONSTRAINTS -> CONSTRAINTS LIBRARY
  then EVIDENCE + CONSTRAINTS -> DECISION TRAIL

AI proposes; humans interpret, validate, and decide.
Traceability is the core principle: every theme links back to the exact
comments that produced it; every quote retains its full original comment
and response ID; every constraint retains its source.
"""

import os
import re
import json
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------
# PAGE CONFIG + VISUAL SYSTEM
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Civic Evidence Studio", page_icon="🏛️", layout="wide")

C = {
    "bg": "#FAFAF8", "card": "#FFFFFF", "text": "#2E2E2E", "text2": "#6B7280",
    "border": "#E5E7EB",
    "blue": "#8C9AC3", "purple": "#A88BAD", "green": "#81AF98",
    "yellow": "#9BA174", "orange": "#AE9B7F", "red": "#BA9190",
    # muted tints for pill/card backgrounds
    "blue_t": "#EEF0F6", "purple_t": "#F3EEF4", "green_t": "#EAF2EE",
    "yellow_t": "#F1F1E9", "orange_t": "#F3EEE8", "red_t": "#F3EAEA",
}

REACTION_COLOR = {"approve": C["green"], "disapprove": C["red"], "none": C["yellow"]}

st.markdown(
    f"""
<style>
.stApp {{ background-color: {C['bg']}; }}
html, body, [class*="css"] {{ color: {C['text']}; }}
h1, h2, h3, h4 {{ color: {C['text']}; font-weight: 700; }}
[data-testid="stSidebar"] {{ background-color: {C['card']}; border-right: 1px solid {C['border']}; }}
[data-testid="stMetric"] {{
  background: {C['card']}; border: 1px solid {C['border']}; border-radius: 12px;
  padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}}
[data-testid="stMetricLabel"] p {{ color: {C['text2']}; font-size: 12px;
  text-transform: uppercase; letter-spacing: .5px; font-weight: 700; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {C['card']}; border-radius: 12px;
}}
.stButton > button {{
  border-radius: 8px; border: 1px solid {C['border']}; font-weight: 600;
}}
.stButton > button[kind="primary"] {{
  background: {C['blue']}; border-color: {C['blue']}; color: #fff;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 18px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.ces-pill {{
  display:inline-block; padding:2px 10px; border-radius:100px;
  font-size:11px; font-weight:700; letter-spacing:.2px; margin-right:6px;
  border:1px solid transparent; white-space:nowrap;
}}
.ces-quote {{
  font-size:14px; color:{C['text']}; line-height:1.55; margin:2px 0 6px 0;
}}
.ces-meta {{ font-size:11.5px; color:{C['text2']}; }}
.ces-note-ai {{
  background:{C['purple_t']}; border:1px solid #DDCBE0; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#6d5372; margin:6px 0;
}}
.ces-note-human {{
  background:{C['blue_t']}; border:1px solid #C7CEE0; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#3d4661; margin:6px 0;
}}
.ces-note-warn {{
  background:{C['red_t']}; border:1px solid #DDBCBC; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#7c4f4e; margin:6px 0;
}}
.ces-note-green {{
  background:{C['green_t']}; border:1px solid #BBD9C9; border-radius:8px;
  padding:10px 12px; font-size:12.5px; color:#3f6853; margin:6px 0;
}}
.ces-chain {{
  font-family: ui-monospace, monospace; font-size:12px; color:{C['text2']};
  background:{C['bg']}; border:1px solid {C['border']}; border-radius:8px;
  padding:10px 14px; white-space:pre; overflow-x:auto;
}}
</style>
""",
    unsafe_allow_html=True,
)


def pill(text, kind):
    """Small status pill. kind in {ai, human, validated, review, processing, conflict, gray, approve, disapprove, none}"""
    styles = {
        "ai":         (C["purple_t"], "#7c5f82", "#DDCBE0"),
        "human":      (C["blue_t"],   "#454f74", "#C7CEE0"),
        "validated":  (C["green_t"],  "#3f6853", "#BBD9C9"),
        "review":     (C["yellow_t"], "#5c6140", "#D3D6BE"),
        "processing": (C["orange_t"], "#6b5c45", "#DBCBB6"),
        "conflict":   (C["red_t"],    "#7c4f4e", "#DDBCBC"),
        "gray":       ("#F3F4F6",     C["text2"], C["border"]),
        "approve":    (C["green_t"],  "#3f6853", "#BBD9C9"),
        "disapprove": (C["red_t"],    "#7c4f4e", "#DDBCBC"),
        "none":       (C["yellow_t"], "#5c6140", "#D3D6BE"),
    }
    bg, fg, bd = styles.get(kind, styles["gray"])
    return (f'<span class="ces-pill" style="background:{bg};color:{fg};'
            f'border-color:{bd};">{text}</span>')


def pills(*items):
    return " ".join(pill(t, k) for t, k in items)


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------

DEFAULT_METADATA = {
    "project": "Santa Monica Airport Conversion Project",
    "stage": "Phase 3A — Scenario Testing",
    "activity": "CoMap Scenario Exercise — Housing Comments",
    "date": "July 2025",
    "activity_type": "Interactive Mapping Survey",
    "platform": "CoMap",
    "location": "Santa Monica Airport",
    "scope": "Santa Monica and surrounding communities",
    "stakeholders": "Residents, Workers, Business owners, Other participants",
    "purpose": ("Gather reactions and qualitative feedback about proposed housing and "
                "related design decisions across three alternative scenarios."),
    "topic": "Housing",
    "notes": "",
}


def init_state():
    ss = st.session_state
    ss.setdefault("metadata", dict(DEFAULT_METADATA))
    ss.setdefault("metadata_saved", False)
    ss.setdefault("raw", {1: None, 2: None, 3: None})       # standardized per-slot dfs
    ss.setdefault("raw_problems", {1: [], 2: [], 3: []})
    ss.setdefault("combined", None)                           # combined dataframe
    ss.setdefault("constraints", [])                          # list of dicts
    ss.setdefault("constraint_seq", 0)
    ss.setdefault("clusters", {})                             # cluster_key -> cluster dict
    ss.setdefault("cluster_run_done", False)
    ss.setdefault("tags", {})                                 # record_id -> [{tag, origin}]
    ss.setdefault("themes", [])                               # validated + human themes
    ss.setdefault("theme_seq", 0)
    ss.setdefault("evidence", [])                             # evidence items
    ss.setdefault("evidence_seq", 0)
    ss.setdefault("decisions", [])
    ss.setdefault("decision_seq", 0)
    ss.setdefault("validating_cluster", None)                 # cluster_key being validated
    ss.setdefault("viewing_cluster", {})                      # cluster_key -> bool
    ss.setdefault("cross_reviews", {})                        # pattern_key -> review dict
    ss.setdefault("page", "01 Data + Context")


def next_id(seq_key, prefix, width=4):
    st.session_state[seq_key] += 1
    return f"{prefix}{st.session_state[seq_key]:0{width}d}"


# ----------------------------------------------------------------------------
# DATA STANDARDIZATION
# ----------------------------------------------------------------------------

def _norm_reaction(v):
    s = str(v).strip().lower()
    if s.startswith("dis") or s in ("down", "thumbsdown", "thumbs down", "-1", "dislike"):
        return "disapprove"
    if "approve" in s or s in ("up", "thumbsup", "thumbs up", "+1", "like", "yes"):
        return "approve"
    return "none"


def standardize(df, slot, filename):
    """Normalize an uploaded dataframe. Returns (df, problems)."""
    problems = []
    df = df.copy()
    # normalize column names: strip whitespace, lowercase, remove spaces/underscores
    df.columns = [re.sub(r"[\s_]+", "", str(c).strip().lower()) for c in df.columns]

    colmap = {}
    for c in df.columns:
        if c in ("comment", "comments", "commenttext", "text"):
            colmap[c] = "comment"
        elif c in ("reaction", "reactions", "thumb", "thumbs", "vote"):
            colmap[c] = "reaction"
        elif c in ("responseid", "respid", "responseld", "response", "participantid"):
            colmap[c] = "response_id"
    df = df.rename(columns=colmap)

    for req, label in [("comment", "Comment"), ("reaction", "Reaction"),
                       ("response_id", "Response ID")]:
        if req not in df.columns:
            problems.append(
                f"**{filename}** is missing a required column: **{label}**. "
                f"Columns found: {', '.join(df.columns)}. "
                "Please check the file — the app will not fabricate this field."
            )
    if problems:
        return None, problems

    # never alter the original comment
    df["comment"] = df["comment"].astype(str)
    df["response_id"] = df["response_id"].astype(str).str.strip()
    df["reaction_original"] = df["reaction"]
    df["reaction"] = df["reaction"].apply(_norm_reaction)
    df["scenario"] = f"Scenario {slot}"
    df["source_file"] = filename
    df = df.reset_index(drop=True)
    df["record_id"] = [f"S{slot}-{i + 1:06d}" for i in range(len(df))]

    out = df[["record_id", "response_id", "scenario", "comment", "reaction",
              "reaction_original", "source_file"]]
    return out, []


def reaction_counts(df):
    vc = df["reaction"].value_counts()
    return {r: int(vc.get(r, 0)) for r in ("approve", "disapprove", "none")}


# ----------------------------------------------------------------------------
# LLM INTEGRATION (optional — graceful fallback)
# ----------------------------------------------------------------------------

def _get_secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets.get(name)  # secrets.toml may not exist
    except Exception:
        return None


def llm_provider():
    """Return ('anthropic'|'openai', key) or (None, None)."""
    k = _get_secret("ANTHROPIC_API_KEY")
    if k:
        return "anthropic", k
    k = _get_secret("OPENAI_API_KEY")
    if k:
        return "openai", k
    return None, None


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_interpret_cluster(sample_comments, keywords, scenario):
    """Ask an LLM to name/summarize a cluster from a SAMPLE of its comments.
    Returns dict {name, summary, tags} or None if unavailable/failed."""
    provider, key = llm_provider()
    if provider is None:
        return None
    numbered = "\n".join(f"{i+1}. {c[:400]}" for i, c in enumerate(sample_comments[:12]))
    prompt = (
        "You are helping a city planner label a cluster of public-engagement comments "
        f"about housing on the Santa Monica Airport site ({scenario}). "
        "These are a representative SAMPLE from the cluster, not all comments.\n\n"
        f"Top cluster keywords: {', '.join(keywords[:10])}\n\n"
        f"Sample comments:\n{numbered}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"name": "<short theme name, max 8 words>", '
        '"summary": "<1-2 sentence neutral description of what this cluster of participants is saying>", '
        '"tags": ["<tag1>", "<tag2>", "<tag3>"]}\n'
        "Do not overstate consensus; describe what these particular comments express."
    )
    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-sonnet-4-5", max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
        data = _extract_json(text)
        if data and data.get("name"):
            return {
                "name": str(data.get("name", "")).strip(),
                "summary": str(data.get("summary", "")).strip(),
                "tags": [str(t).strip() for t in (data.get("tags") or [])][:5],
            }
    except Exception as e:
        st.session_state.setdefault("llm_errors", []).append(str(e))
    return None


# ----------------------------------------------------------------------------
# THEMATIC CLUSTERING (local, transparent: TF-IDF + KMeans)
# ----------------------------------------------------------------------------

def cluster_one_scenario(sdf):
    """Cluster comments for one scenario. Returns list of cluster dicts
    (calculated fields only — AI interpretation is attached separately)."""
    sdf = sdf[sdf["comment"].str.strip().astype(bool)]
    sdf = sdf[~sdf["comment"].str.strip().str.lower().isin(["nan", "none", ""])]
    n = len(sdf)
    if n == 0:
        return []
    texts = sdf["comment"].tolist()

    k = 1 if n < 6 else int(min(5, max(2, round(n / 10))))
    vec = TfidfVectorizer(stop_words="english", max_features=2500, ngram_range=(1, 2),
                          min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    terms = np.array(vec.get_feature_names_out())

    if k == 1 or X.shape[0] <= k:
        labels = np.zeros(n, dtype=int)
        centers = np.asarray(X.mean(axis=0))
        k = 1
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_

    clusters = []
    for ci in range(k):
        idx = np.where(labels == ci)[0]
        if len(idx) == 0:
            continue
        sub = sdf.iloc[idx]
        center = np.asarray(centers[ci]).ravel()
        top_terms = terms[np.argsort(center)[::-1][:8]].tolist()

        sims = np.asarray(X[idx].dot(center)).ravel()
        order = np.argsort(sims)[::-1]
        rep_ids = sub.iloc[order[:5]]["record_id"].tolist()

        counts = reaction_counts(sub)
        # majority reaction among approve/disapprove; counter-evidence = the opposite
        if counts["approve"] >= counts["disapprove"]:
            majority, minority = "approve", "disapprove"
        else:
            majority, minority = "disapprove", "approve"
        counter_ids = sub[sub["reaction"] == minority]["record_id"].tolist()

        clusters.append({
            "scenario": sub["scenario"].iloc[0],
            "record_ids": sub["record_id"].tolist(),
            "n_comments": int(len(sub)),
            "n_respondents": int(sub["response_id"].nunique()),
            "counts": counts,
            "majority": majority,
            "counter_ids": counter_ids,
            "keywords": top_terms,
            "rep_ids": rep_ids,
        })
    clusters.sort(key=lambda c: -c["n_comments"])
    return clusters


def run_thematic_analysis():
    """Cluster each scenario independently; attach AI interpretation per cluster."""
    df = st.session_state.combined
    provider, _ = llm_provider()
    st.session_state.clusters = {}
    for scen in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scen]
        found = cluster_one_scenario(sdf)
        for j, cl in enumerate(found):
            key = f"{scen.replace(' ', '')}-C{j + 1}"
            cl["key"] = key
            cl["status"] = "ai-suggested"
            cl["constraint_links"] = {}    # constraint_id -> "ai" | "confirmed" | "dismissed"
            rep_comments = df[df["record_id"].isin(cl["rep_ids"])]["comment"].tolist()
            ai = llm_interpret_cluster(rep_comments, cl["keywords"], scen) if provider else None
            if ai:
                cl["ai"] = {**ai, "source": "llm"}
            else:
                # transparent keyword-based fallback label — clearly not an LLM reading
                kw = [w.title() for w in cl["keywords"][:3]]
                cl["ai"] = {
                    "name": " / ".join(kw) if kw else "Unlabeled cluster",
                    "summary": ("Keyword-based label. AI theme interpretation unavailable — "
                                "configure an API key to enable AI-generated theme names "
                                "and summaries."),
                    "tags": [w.title() for w in cl["keywords"][:3]],
                    "source": "fallback",
                }
            st.session_state.clusters[key] = cl
    st.session_state.cluster_run_done = True


# ----------------------------------------------------------------------------
# TAGS / EVIDENCE / SHARED HELPERS
# ----------------------------------------------------------------------------

def add_tag(record_id, tag, origin):
    tag = tag.strip()
    if not tag:
        return
    entry = st.session_state.tags.setdefault(record_id, [])
    if not any(t["tag"].lower() == tag.lower() and t["origin"] == origin for t in entry):
        entry.append({"tag": tag, "origin": origin})


def all_known_tags():
    tags = set()
    for entries in st.session_state.tags.values():
        for t in entries:
            tags.add(t["tag"])
    for cl in st.session_state.clusters.values():
        for t in cl["ai"].get("tags", []):
            tags.add(t)
    return sorted(tags)


def add_evidence(item):
    item["evidence_id"] = next_id("evidence_seq", "EV-")
    item["created"] = datetime.date.today().isoformat()
    st.session_state.evidence.append(item)
    return item["evidence_id"]


def get_comment_row(record_id):
    df = st.session_state.combined
    rows = df[df["record_id"] == record_id]
    return rows.iloc[0] if len(rows) else None


def comment_card(row, key_prefix, show_actions=True):
    """Render one comment card with provenance + actions."""
    with st.container(border=True):
        st.markdown(f'<div class="ces-quote">&ldquo;{row["comment"]}&rdquo;</div>',
                    unsafe_allow_html=True)
        tag_html = ""
        for t in st.session_state.tags.get(row["record_id"], []):
            tag_html += pill(t["tag"], "ai" if t["origin"] == "ai" else "human")
        st.markdown(
            pills((row["reaction"].title(), row["reaction"]),
                  (row["scenario"], "gray"))
            + tag_html
            + f'<div class="ces-meta" style="margin-top:6px;">Response {row["response_id"]}'
              f' &middot; {row["record_id"]} &middot; {row["source_file"]}</div>',
            unsafe_allow_html=True,
        )
        if not show_actions:
            return
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Add to Evidence", key=f"{key_prefix}-ev-{row['record_id']}"):
                add_evidence({
                    "type": "Direct Comment",
                    "title": row["comment"][:70] + ("…" if len(row["comment"]) > 70 else ""),
                    "record_ids": [row["record_id"]],
                    "response_ids": [row["response_id"]],
                    "scenarios": [row["scenario"]],
                    "reaction": row["reaction"],
                    "source_files": [row["source_file"]],
                    "original_comment": row["comment"],
                    "selected_quote": None,
                    "theme": None,
                    "tags": [t["tag"] for t in st.session_state.tags.get(row["record_id"], [])],
                    "status": "Human Selected",
                })
                st.toast(f"Saved {row['record_id']} to Evidence Library")
        with c2:
            with st.popover("Add Tag"):
                existing = all_known_tags()
                pick = st.selectbox("Existing tag", ["—"] + existing,
                                    key=f"{key_prefix}-tagsel-{row['record_id']}")
                new = st.text_input("Or new tag", key=f"{key_prefix}-tagnew-{row['record_id']}")
                if st.button("Apply tag", key=f"{key_prefix}-tagapply-{row['record_id']}"):
                    chosen = new.strip() or (pick if pick != "—" else "")
                    if chosen:
                        add_tag(row["record_id"], chosen, "human")
                        st.rerun()
        with c3:
            with st.popover("Save Quote"):
                st.caption("Select the passage to save. The complete original comment "
                           "and response ID are always preserved alongside it.")
                q = st.text_area("Quoted passage", value=row["comment"],
                                 key=f"{key_prefix}-quote-{row['record_id']}")
                if st.button("Save highlighted quote as evidence",
                             key=f"{key_prefix}-quotesave-{row['record_id']}"):
                    if q.strip() and q.strip() in row["comment"]:
                        add_evidence({
                            "type": "Highlighted Quote",
                            "title": q.strip()[:70] + ("…" if len(q.strip()) > 70 else ""),
                            "record_ids": [row["record_id"]],
                            "response_ids": [row["response_id"]],
                            "scenarios": [row["scenario"]],
                            "reaction": row["reaction"],
                            "source_files": [row["source_file"]],
                            "original_comment": row["comment"],
                            "selected_quote": q.strip(),
                            "theme": None,
                            "tags": [t["tag"] for t in
                                     st.session_state.tags.get(row["record_id"], [])],
                            "status": "Human Selected",
                        })
                        st.toast("Quote saved with full provenance")
                    else:
                        st.error("The quote must be an exact passage from the original "
                                 "comment — quotes are never fabricated or altered.")


# ----------------------------------------------------------------------------
# PAGE 01 — DATA + CONTEXT
# ----------------------------------------------------------------------------

CONSTRAINT_TYPES = ["Legal / Regulatory", "Voter Mandate", "Financial", "Environmental",
                    "Site / Physical", "Technical", "Timeline", "Other"]


def page_data_context():
    st.title("Data + Context")
    st.markdown(f'<p style="color:{C["text2"]};margin-top:-8px;">Document the engagement '
                'activity, upload the source data, and record the project conditions '
                'surrounding it.</p>', unsafe_allow_html=True)

    tab_data, tab_con = st.tabs(["Engagement Data", "Project Constraints"])

    # ---------------- ENGAGEMENT DATA ----------------
    with tab_data:
        st.subheader("Engagement Activity Metadata")
        md = st.session_state.metadata
        with st.form("metadata_form"):
            c1, c2 = st.columns(2)
            with c1:
                md_project = st.text_input("Project", md["project"])
                md_activity = st.text_input("Activity Name", md["activity"])
                md_type = st.text_input("Engagement Activity Type", md["activity_type"])
                md_location = st.text_input("Location", md["location"])
                md_stake = st.text_input("Stakeholder Groups", md["stakeholders"])
                md_topic = st.text_input("Topic", md["topic"])
            with c2:
                md_stage = st.text_input("Project Stage", md["stage"])
                md_date = st.text_input("Date", md["date"])
                md_platform = st.text_input("Platform", md["platform"])
                md_scope = st.text_input("Geographic Scope", md["scope"])
                md_purpose = st.text_area("Engagement Purpose", md["purpose"], height=88)
            md_notes = st.text_area("Notes", md["notes"], height=68)
            if st.form_submit_button("Save Metadata", type="primary"):
                st.session_state.metadata = {
                    "project": md_project, "stage": md_stage, "activity": md_activity,
                    "date": md_date, "activity_type": md_type, "platform": md_platform,
                    "location": md_location, "scope": md_scope, "stakeholders": md_stake,
                    "purpose": md_purpose, "topic": md_topic, "notes": md_notes,
                }
                st.session_state.metadata_saved = True
        if st.session_state.metadata_saved:
            st.markdown(pill("Metadata saved", "validated"), unsafe_allow_html=True)

        st.divider()
        st.subheader("Upload the Three Scenario Files")
        st.caption("Each file should contain columns: comment, reaction, responseId "
                   "(capitalization and spacing are normalized automatically). "
                   "Original comments are never altered.")

        cols = st.columns(3)
        for slot, col in zip((1, 2, 3), cols):
            with col:
                with st.container(border=True):
                    st.markdown(f"**Scenario {slot}**")
                    up = st.file_uploader(f"Upload Housing Scenario {slot} XLSX",
                                          type=["xlsx"], key=f"upload_{slot}")
                    sig = (up.name, up.size) if up is not None else None
                    if up is not None and st.session_state.get(f"upsig_{slot}") != sig:
                        # only (re)parse when the file actually changes
                        try:
                            raw_df = pd.read_excel(up)
                            sdf, problems = standardize(raw_df, slot, up.name)
                            st.session_state.raw[slot] = sdf
                            st.session_state.raw_problems[slot] = problems
                            st.session_state[f"upsig_{slot}"] = sig
                            st.session_state.combined = None  # require reprocess
                            st.session_state.cluster_run_done = False
                        except Exception as e:
                            st.session_state.raw[slot] = None
                            st.session_state.raw_problems[slot] = [
                                f"Could not read this file as XLSX: {e}"]
                    for p in st.session_state.raw_problems[slot]:
                        st.error(p)
                    sdf = st.session_state.raw[slot]
                    if sdf is not None:
                        cts = reaction_counts(sdf)
                        st.markdown(
                            pills(("Parsed", "validated")) +
                            f'<div class="ces-meta" style="margin-top:6px;">'
                            f'{sdf["source_file"].iloc[0]}<br>'
                            f'{len(sdf)} comments · {sdf["response_id"].nunique()} unique '
                            f'response IDs<br>'
                            f'Approve {cts["approve"]} · Disapprove {cts["disapprove"]} · '
                            f'None {cts["none"]}</div>',
                            unsafe_allow_html=True)
                        with st.expander("Preview data"):
                            st.dataframe(sdf.head(10), use_container_width=True,
                                         hide_index=True)
                        if st.button("Clear", key=f"clear_{slot}"):
                            st.session_state.raw[slot] = None
                            st.session_state.raw_problems[slot] = []
                            st.session_state[f"upsig_{slot}"] = None
                            st.session_state.combined = None
                            st.rerun()

        st.divider()
        loaded = [s for s in (1, 2, 3) if st.session_state.raw[s] is not None]
        missing = [s for s in (1, 2, 3) if st.session_state.raw[s] is None]
        if missing:
            st.markdown(f'<div class="ces-note-human">Waiting for: '
                        f'{", ".join("Scenario " + str(s) for s in missing)}. '
                        'You can process with fewer scenarios, but the cross-scenario '
                        'comparison works best with all three.</div>',
                        unsafe_allow_html=True)
        if st.button("Process All Three Scenarios", type="primary",
                     disabled=(len(loaded) == 0)):
            with st.spinner("Combining and standardizing…"):
                combined = pd.concat([st.session_state.raw[s] for s in loaded],
                                     ignore_index=True)
                st.session_state.combined = combined
        if st.session_state.combined is not None:
            df = st.session_state.combined
            cts = reaction_counts(df)
            st.markdown(pill("Dataset ready", "validated"), unsafe_allow_html=True)
            m = st.columns(4)
            m[0].metric("Total comments", len(df))
            for i, scen in enumerate(["Scenario 1", "Scenario 2", "Scenario 3"]):
                m[i + 1 if i < 3 else 3].metric(
                    f"{scen} comments", int((df["scenario"] == scen).sum()))
            m2 = st.columns(4)
            m2[0].metric("Unique response IDs", df["response_id"].nunique())
            m2[1].metric("Approve", cts["approve"])
            m2[2].metric("Disapprove", cts["disapprove"])
            m2[3].metric("None", cts["none"])
            st.caption("All figures are calculated from the uploaded files. Multiple "
                       "comments can share a response ID — one row is one comment, not "
                       "necessarily one unique participant.")

    # ---------------- PROJECT CONSTRAINTS ----------------
    with tab_con:
        st.subheader("Project Constraints")
        st.markdown('<div class="ces-note-human">Constraints are separate from community '
                    'engagement evidence. A participant mentioning Measure LC in a comment '
                    'is community evidence about how that participant understands the '
                    'constraint — it is <b>not</b> the authoritative constraint itself. '
                    'Document constraints here from their actual sources.</div>',
                    unsafe_allow_html=True)

        with st.expander("＋ Add Constraint", expanded=(len(st.session_state.constraints) == 0)):
            with st.form("constraint_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    cn_name = st.text_input("Constraint Name", placeholder="e.g. Measure LC")
                    cn_type = st.selectbox("Constraint Type", CONSTRAINT_TYPES)
                    cn_source = st.text_input("Source",
                                              placeholder="e.g. Measure LC (2014) ballot text")
                    cn_status = st.selectbox("Status", ["Active", "Pending", "Superseded"])
                with c2:
                    cn_desc = st.text_area("Description", height=88)
                    cn_phase = st.text_input("Relevant Phase", placeholder="e.g. Phase 1–4")
                    cn_notes = st.text_input("Notes")
                if st.form_submit_button("Save Constraint", type="primary"):
                    if cn_name.strip():
                        st.session_state.constraint_seq += 1
                        st.session_state.constraints.append({
                            "id": f"CON-{st.session_state.constraint_seq:03d}",
                            "name": cn_name.strip(), "type": cn_type,
                            "description": cn_desc.strip(), "source": cn_source.strip(),
                            "phase": cn_phase.strip(), "status": cn_status,
                            "notes": cn_notes.strip(),
                        })
                        st.rerun()
                    else:
                        st.error("A constraint needs at least a name.")

        if not st.session_state.constraints:
            st.caption("No constraints documented yet.")
        for con in st.session_state.constraints:
            with st.container(border=True):
                st.markdown(
                    f"**{con['id']} — {con['name']}**  \n" , unsafe_allow_html=False)
                st.markdown(
                    pills((con["type"], "conflict" if ("Legal" in con["type"] or
                                                       "Voter" in con["type"]) else "human"),
                          (con["status"], "validated" if con["status"] == "Active" else "review")),
                    unsafe_allow_html=True)
                if con["description"]:
                    st.write(con["description"])
                st.markdown(f'<div class="ces-meta">Source: {con["source"] or "—"} · '
                            f'Phase: {con["phase"] or "—"}'
                            + (f' · Notes: {con["notes"]}' if con["notes"] else "")
                            + '</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE 02 — INSIGHTS PLAYGROUND
# ----------------------------------------------------------------------------

def reaction_chart(df):
    scens = sorted(df["scenario"].unique())
    fig = go.Figure()
    for reaction in ("approve", "disapprove", "none"):
        vals = [int(((df["scenario"] == s) & (df["reaction"] == reaction)).sum())
                for s in scens]
        fig.add_bar(name=reaction.title(), x=scens, y=vals,
                    marker_color=REACTION_COLOR[reaction],
                    text=vals, textposition="outside")
    fig.update_layout(
        barmode="group", height=360, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color=C["text"], size=13),
        legend=dict(orientation="h", y=1.12, title="Participant Reaction"),
        yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
        xaxis=dict(linecolor=C["border"]),
    )
    return fig


def scenario_pct_table(df):
    rows = []
    for s in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == s]
        cts = reaction_counts(sdf)
        n = max(1, len(sdf))
        rows.append({
            "Scenario": s, "Comments": len(sdf),
            "Unique response IDs": sdf["response_id"].nunique(),
            "Approve": f'{cts["approve"]} ({cts["approve"]/n:.0%})',
            "Disapprove": f'{cts["disapprove"]} ({cts["disapprove"]/n:.0%})',
            "None": f'{cts["none"]} ({cts["none"]/n:.0%})',
        })
    return pd.DataFrame(rows)


def constraint_auto_suggestions(cluster, df):
    """AI-suggested constraint relationships: does a documented constraint's name
    appear inside this cluster's comments? Suggestion only — never an assertion."""
    texts = " ".join(df[df["record_id"].isin(cluster["record_ids"])]["comment"]
                     .str.lower().tolist())
    hits = []
    for con in st.session_state.constraints:
        if con["id"] in cluster["constraint_links"]:
            continue
        name = con["name"].strip().lower()
        if name and name in texts:
            hits.append(con)
    return hits


def render_constraint_links(cluster, df, key_prefix):
    """AI-suggested + confirmed constraint relationships for a cluster."""
    for con in constraint_auto_suggestions(cluster, df):
        cluster["constraint_links"][con["id"]] = "ai"
    links = {cid: s for cid, s in cluster["constraint_links"].items() if s != "dismissed"}
    if not links:
        return
    st.markdown("**Potentially relevant constraints**")
    for cid, state in links.items():
        con = next((c for c in st.session_state.constraints if c["id"] == cid), None)
        if not con:
            continue
        with st.container(border=True):
            if state == "ai":
                st.markdown(pills(("AI Suggested Relationship", "ai")) +
                            f" **{con['id']} — {con['name']}**", unsafe_allow_html=True)
                st.caption("Participants in this cluster mention this constraint by name. "
                           "AI does not determine that a constraint invalidates or "
                           "resolves community evidence — it only surfaces it for review.")
                b1, b2, b3 = st.columns(3)
                if b1.button("Review Constraint", key=f"{key_prefix}-rc-{cid}"):
                    st.info(f"{con['id']} · {con['type']} · {con['status']}  \n"
                            f"{con['description'] or '(no description)'}  \n"
                            f"Source: {con['source'] or '—'}")
                if b2.button("Mark as Relevant", key=f"{key_prefix}-mr-{cid}"):
                    cluster["constraint_links"][cid] = "confirmed"
                    st.rerun()
                if b3.button("Dismiss", key=f"{key_prefix}-dm-{cid}"):
                    cluster["constraint_links"][cid] = "dismissed"
                    st.rerun()
            else:
                st.markdown(pills(("Human Confirmed", "validated")) +
                            f" **{con['id']} — {con['name']}**", unsafe_allow_html=True)


def render_cluster_card(cluster, df):
    key = cluster["key"]
    ai = cluster["ai"]
    cts = cluster["counts"]
    with st.container(border=True):
        src_pill = ("AI Suggested Theme", "ai") if ai["source"] == "llm" \
            else ("Keyword Cluster — AI interpretation unavailable", "review")
        status_pill = []
        if cluster["status"] == "validated":
            status_pill = [("Human Validated", "validated")]
        elif cluster["status"] == "rejected":
            status_pill = [("Rejected", "gray")]
        st.markdown(pills(src_pill, (cluster["scenario"], "gray"), *status_pill),
                    unsafe_allow_html=True)
        st.markdown(f"### {ai['name']}")
        st.markdown(
            f'<div class="ces-meta">{cluster["n_comments"]} related comments · '
            f'{cluster["n_respondents"]} unique response IDs<br>'
            f'Reaction distribution — Approve {cts["approve"]} · '
            f'Disapprove {cts["disapprove"]} · None {cts["none"]}</div>',
            unsafe_allow_html=True)
        note_cls = "ces-note-ai" if ai["source"] == "llm" else "ces-note-warn"
        label = "AI Summary" if ai["source"] == "llm" else "No AI interpretation"
        st.markdown(f'<div class="{note_cls}"><b>{label}:</b> {ai["summary"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("Suggested tags: " +
                    "".join(pill(t, "ai") for t in ai.get("tags", [])),
                    unsafe_allow_html=True)
        st.caption("The title, summary, and tags above are AI-generated suggestions. "
                   "The counts are calculated in Python from the associated record IDs.")

        render_constraint_links(cluster, df, key)

        if cluster["status"] == "ai-suggested":
            b = st.columns(5)
            if b[0].button("View Comments", key=f"vc-{key}"):
                st.session_state.viewing_cluster[key] = \
                    not st.session_state.viewing_cluster.get(key, False)
            with b[1].popover("Edit Theme"):
                new_name = st.text_input("Working name", ai["name"], key=f"edit-name-{key}")
                if st.button("Apply", key=f"edit-apply-{key}"):
                    ai.setdefault("original_name", ai["name"])
                    ai["name"] = new_name
                    st.rerun()
            with b[2].popover("Merge"):
                others = [k for k, c in st.session_state.clusters.items()
                          if k != key and c["status"] == "ai-suggested"
                          and c["scenario"] == cluster["scenario"]]
                if others:
                    target = st.selectbox(
                        "Merge into", others,
                        format_func=lambda k: st.session_state.clusters[k]["ai"]["name"],
                        key=f"merge-sel-{key}")
                    if st.button("Merge clusters", key=f"merge-do-{key}"):
                        tgt = st.session_state.clusters[target]
                        tgt["record_ids"] = sorted(set(tgt["record_ids"])
                                                   | set(cluster["record_ids"]))
                        sub = df[df["record_id"].isin(tgt["record_ids"])]
                        tgt["n_comments"] = len(sub)
                        tgt["n_respondents"] = sub["response_id"].nunique()
                        tgt["counts"] = reaction_counts(sub)
                        maj = "approve" if tgt["counts"]["approve"] >= \
                            tgt["counts"]["disapprove"] else "disapprove"
                        minr = "disapprove" if maj == "approve" else "approve"
                        tgt["majority"] = maj
                        tgt["counter_ids"] = sub[sub["reaction"] == minr]["record_id"].tolist()
                        tgt.setdefault("merged_from", []).append(ai["name"])
                        cluster["status"] = "merged"
                        st.rerun()
                else:
                    st.caption("No other AI clusters in this scenario to merge with.")
            if b[3].button("Reject", key=f"rej-{key}"):
                cluster["status"] = "rejected"
                st.rerun()
            if b[4].button("Validate", key=f"val-{key}", type="primary"):
                st.session_state.validating_cluster = key
                st.rerun()

        # ---- expandable: every original comment in the cluster ----
        if st.session_state.viewing_cluster.get(key):
            sub = df[df["record_id"].isin(cluster["record_ids"])]
            sort = st.selectbox("Sort by", ["Approve first", "Disapprove first",
                                            "None first", "Response ID"],
                                key=f"sort-{key}")
            if sort == "Response ID":
                sub = sub.sort_values("response_id")
            else:
                first = sort.split()[0].lower()
                sub = sub.sort_values("reaction",
                                      key=lambda s: (s != first).astype(int))
            st.markdown("**Representative comments** (closest to cluster center)")
            for rid in cluster["rep_ids"][:3]:
                r = get_comment_row(rid)
                if r is not None:
                    comment_card(r, f"rep-{key}")
            if cluster["counter_ids"]:
                st.markdown(f'<div class="ces-note-warn"><b>Potential counter-evidence '
                            f'({len(cluster["counter_ids"])}):</b> comments in this '
                            f'cluster whose reaction opposes the cluster majority '
                            f'({cluster["majority"]}). A common theme is not '
                            'automatically consensus — disagreement is preserved.</div>',
                            unsafe_allow_html=True)
                for rid in cluster["counter_ids"][:5]:
                    r = get_comment_row(rid)
                    if r is not None:
                        comment_card(r, f"ctr-{key}")
            with st.expander(f"All {len(sub)} comments in this cluster"):
                for _, r in sub.iterrows():
                    comment_card(r, f"all-{key}")


def render_validation_form(cluster, df):
    """The human review form that turns an AI cluster into a validated theme."""
    key = cluster["key"]
    ai = cluster["ai"]
    st.markdown("---")
    st.subheader(f"Validate Theme — {ai['name']}")
    st.markdown('<div class="ces-note-human">Review the underlying comments before '
                'validating. The AI original name and summary are preserved unchanged '
                'alongside your interpretation.</div>', unsafe_allow_html=True)
    sub = df[df["record_id"].isin(cluster["record_ids"])]
    label_map = {rid: f'{rid} · {sub[sub["record_id"] == rid]["comment"].iloc[0][:80]}'
                 for rid in cluster["record_ids"]}
    with st.form(f"validate-{key}"):
        name = st.text_input("Theme Name", ai["name"])
        interp = st.text_area("Human Interpretation",
                              placeholder="What does this pattern mean, in your own words?")
        include = st.multiselect("Comments to Include", cluster["record_ids"],
                                 default=cluster["record_ids"],
                                 format_func=lambda r: label_map[r])
        tags = st.multiselect("Tags", sorted(set(all_known_tags())
                                             | set(ai.get("tags", []))),
                              default=ai.get("tags", []))
        new_tag = st.text_input("Add a new tag (optional)")
        notes = st.text_area("Optional Notes", height=68)
        rel_cons = st.multiselect(
            "Relevant Constraints",
            [c["id"] for c in st.session_state.constraints],
            default=[cid for cid, s in cluster["constraint_links"].items()
                     if s == "confirmed"],
            format_func=lambda cid: f"{cid} — "
            f"{next(c['name'] for c in st.session_state.constraints if c['id'] == cid)}")
        c1, c2 = st.columns(2)
        submitted = c1.form_submit_button("Validate Theme", type="primary")
        cancelled = c2.form_submit_button("Cancel")
    if cancelled:
        st.session_state.validating_cluster = None
        st.rerun()
    if submitted:
        if not interp.strip():
            st.error("A human interpretation is required — that is the point of review.")
            return
        if new_tag.strip():
            tags = tags + [new_tag.strip()]
        excluded = [r for r in cluster["record_ids"] if r not in include]
        inc_sub = df[df["record_id"].isin(include)]
        theme_id = next_id("theme_seq", "TH-")
        theme = {
            "theme_id": theme_id,
            "origin": "ai-validated",
            "ai_original_name": ai.get("original_name", ai["name"]),
            "ai_original_summary": ai["summary"],
            "ai_source": ai["source"],
            "name": name.strip(), "interpretation": interp.strip(),
            "record_ids": include, "excluded_record_ids": excluded,
            "scenarios": sorted(inc_sub["scenario"].unique().tolist()),
            "counts": reaction_counts(inc_sub),
            "n_respondents": int(inc_sub["response_id"].nunique()),
            "tags": tags, "notes": notes.strip(),
            "counter_ids": [r for r in cluster["counter_ids"] if r in include],
            "constraints": rel_cons,
            "validated": datetime.date.today().isoformat(),
            "status": "HUMAN VALIDATED",
            "cluster_key": key,
        }
        st.session_state.themes.append(theme)
        for rid in include:
            for t in tags:
                add_tag(rid, t, "ai" if t in ai.get("tags", []) else "human")
        cluster["status"] = "validated"
        cluster["theme_id"] = theme_id
        add_evidence({
            "type": "Validated Theme",
            "title": theme["name"],
            "theme_id": theme_id,
            "record_ids": include,
            "response_ids": sorted(inc_sub["response_id"].unique().tolist()),
            "scenarios": theme["scenarios"],
            "reaction": None,
            "counts": theme["counts"],
            "source_files": sorted(inc_sub["source_file"].unique().tolist()),
            "original_comment": None, "selected_quote": None,
            "theme": theme["name"], "tags": tags,
            "interpretation": interp.strip(),
            "ai_original_name": theme["ai_original_name"],
            "ai_original_summary": theme["ai_original_summary"],
            "counter_ids": theme["counter_ids"],
            "constraints": rel_cons,
            "status": "Human Validated",
        })
        st.session_state.validating_cluster = None
        st.toast(f"Theme validated and saved to Evidence Library as {theme_id}")
        st.rerun()


def render_create_human_theme(df):
    with st.expander("＋ Create Theme (human-created, no AI)"):
        label_map = {rid: f'{rid} · {c[:80]}' for rid, c in
                     zip(df["record_id"], df["comment"])}
        with st.form("human-theme"):
            name = st.text_input("Theme Name")
            interp = st.text_area("Interpretation")
            scens = st.multiselect("Scenarios", sorted(df["scenario"].unique()),
                                   default=sorted(df["scenario"].unique()))
            pool = df[df["scenario"].isin(scens)]["record_id"].tolist() if scens else []
            include = st.multiselect("Select Comments", pool,
                                     format_func=lambda r: label_map[r])
            tags = st.multiselect("Tags", all_known_tags())
            new_tag = st.text_input("Add a new tag (optional)")
            rel_cons = st.multiselect(
                "Relevant Constraints",
                [c["id"] for c in st.session_state.constraints],
                format_func=lambda cid: f"{cid} — "
                f"{next(c['name'] for c in st.session_state.constraints if c['id'] == cid)}")
            if st.form_submit_button("Save Theme to Evidence Library", type="primary"):
                if not name.strip() or not include:
                    st.error("A human theme needs a name and at least one comment.")
                else:
                    if new_tag.strip():
                        tags = tags + [new_tag.strip()]
                    inc_sub = df[df["record_id"].isin(include)]
                    theme_id = next_id("theme_seq", "TH-")
                    theme = {
                        "theme_id": theme_id, "origin": "human",
                        "ai_original_name": None, "ai_original_summary": None,
                        "ai_source": None,
                        "name": name.strip(), "interpretation": interp.strip(),
                        "record_ids": include, "excluded_record_ids": [],
                        "scenarios": sorted(inc_sub["scenario"].unique().tolist()),
                        "counts": reaction_counts(inc_sub),
                        "n_respondents": int(inc_sub["response_id"].nunique()),
                        "tags": tags, "notes": "",
                        "counter_ids": [], "constraints": rel_cons,
                        "validated": datetime.date.today().isoformat(),
                        "status": "HUMAN VALIDATED", "cluster_key": None,
                    }
                    st.session_state.themes.append(theme)
                    for rid in include:
                        for t in tags:
                            add_tag(rid, t, "human")
                    add_evidence({
                        "type": "Validated Theme", "title": theme["name"],
                        "theme_id": theme_id, "record_ids": include,
                        "response_ids": sorted(inc_sub["response_id"].unique().tolist()),
                        "scenarios": theme["scenarios"], "reaction": None,
                        "counts": theme["counts"],
                        "source_files": sorted(inc_sub["source_file"].unique().tolist()),
                        "original_comment": None, "selected_quote": None,
                        "theme": theme["name"], "tags": tags,
                        "interpretation": interp.strip(),
                        "ai_original_name": None, "ai_original_summary": None,
                        "counter_ids": [], "constraints": rel_cons,
                        "status": "Human Validated",
                    })
                    st.toast(f"Human theme saved as {theme_id}")
                    st.rerun()


def cross_scenario_patterns():
    """Compare scenario-level clusters by keyword overlap. Proposals only."""
    clusters = [c for c in st.session_state.clusters.values()
                if c["status"] in ("ai-suggested", "validated")]
    by_scen = {}
    for c in clusters:
        by_scen.setdefault(c["scenario"], []).append(c)
    scens = sorted(by_scen)
    patterns = []
    seen = set()
    for i, s1 in enumerate(scens):
        for c1 in by_scen[s1]:
            group = [c1]
            kws1 = set(c1["keywords"])
            for s2 in scens[i + 1:]:
                best, best_j = None, 0.0
                for c2 in by_scen[s2]:
                    inter = len(kws1 & set(c2["keywords"]))
                    union = len(kws1 | set(c2["keywords"]))
                    j = inter / union if union else 0
                    if j > best_j:
                        best, best_j = c2, j
                if best is not None and best_j >= 0.15:
                    group.append(best)
            gkey = tuple(sorted(c["key"] for c in group))
            if len(group) >= 2 and gkey not in seen:
                seen.add(gkey)
                n_scen = len({c["scenario"] for c in group})
                if n_scen == len(scens) and len(scens) >= 3:
                    rel = "Appears Across All Scenarios"
                else:
                    dis = {c["scenario"]: (c["counts"]["disapprove"] /
                                           max(1, c["n_comments"])) for c in group}
                    strongest = max(dis, key=dis.get)
                    rel = (f"Stronger in {strongest}"
                           if max(dis.values()) - min(dis.values()) > 0.2
                           else "Shared Across Scenarios")
                patterns.append({"key": "|".join(gkey), "relationship": rel,
                                 "clusters": group,
                                 "shared_keywords": sorted(
                                     set.intersection(*[set(c["keywords"])
                                                        for c in group]))})
    for c in clusters:  # scenario-specific clusters
        if not any(c in p["clusters"] for p in patterns):
            patterns.append({"key": c["key"], "relationship": "Mostly Scenario-Specific",
                             "clusters": [c], "shared_keywords": c["keywords"][:4]})
    return patterns


def page_insights():
    st.title("Insights Playground")
    st.markdown(
        pills(("AI-assisted analysis", "ai")) +
        f'<span style="color:{C["text2"]};font-size:13px;"> AI suggestions are starting '
        'points for human interpretation.</span>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{C["text2"]};margin-top:-2px;">Explore the comments, '
                'compare scenarios, and develop themes.</p>', unsafe_allow_html=True)

    df = st.session_state.combined
    if df is None:
        st.markdown('<div class="ces-note-human">No processed dataset yet. Go to '
                    '<b>01 Data + Context</b>, upload the scenario XLSX files, and press '
                    '<b>Process All Three Scenarios</b>.</div>', unsafe_allow_html=True)
        return

    # constraints drawer — available from anywhere in the playground
    with st.sidebar.expander("☷ Project Constraints", expanded=False):
        if not st.session_state.constraints:
            st.caption("No constraints documented yet — add them in Data + Context.")
        for con in st.session_state.constraints:
            st.markdown(f"**{con['id']} — {con['name']}**")
            st.markdown(pills((con["type"], "gray"), (con["status"], "validated")),
                        unsafe_allow_html=True)
            if con["description"]:
                st.caption(con["description"][:160])

    tab_over, tab_comments, tab_themes, tab_compare = st.tabs(
        ["Overview", "Comments", "Themes", "Compare"])

    # ---------------- OVERVIEW ----------------
    with tab_over:
        m = st.columns(4)
        m[0].metric("Total comments", len(df))
        for i, scen in enumerate(sorted(df["scenario"].unique())):
            if i < 3:
                m[i + 1].metric(f"{scen}", int((df["scenario"] == scen).sum()))
        st.subheader("Approve vs Disapprove by Scenario")
        st.caption("Participant Reaction — the reaction field supplied in the dataset is "
                   "the authoritative reaction variable. This is not AI sentiment.")
        st.plotly_chart(reaction_chart(df), use_container_width=True)
        st.dataframe(scenario_pct_table(df), use_container_width=True, hide_index=True)
        if st.button("Save this reaction breakdown as a Quantitative Finding"):
            add_evidence({
                "type": "Quantitative Finding",
                "title": "Participant reaction breakdown by scenario",
                "record_ids": df["record_id"].tolist(),
                "response_ids": sorted(df["response_id"].unique().tolist()),
                "scenarios": sorted(df["scenario"].unique().tolist()),
                "reaction": None,
                "counts": reaction_counts(df),
                "per_scenario": {s: reaction_counts(df[df["scenario"] == s])
                                 for s in sorted(df["scenario"].unique())},
                "source_files": sorted(df["source_file"].unique().tolist()),
                "original_comment": None, "selected_quote": None,
                "theme": None, "tags": [], "status": "Human Selected",
            })
            st.toast("Quantitative finding saved to Evidence Library")

    # ---------------- COMMENTS ----------------
    with tab_comments:
        f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
        f_scen = f1.selectbox("Scenario", ["All"] + sorted(df["scenario"].unique()))
        f_reac = f2.selectbox("Reaction", ["All", "approve", "disapprove", "none"])
        theme_opts = ["All"] + [f'{c["key"]} · {c["ai"]["name"]}'
                                for c in st.session_state.clusters.values()
                                if c["status"] in ("ai-suggested", "validated")] \
                     + [f"tag:{t}" for t in all_known_tags()]
        f_theme = f3.selectbox("Theme / Tag", theme_opts)
        f_kw = f4.text_input("Keyword", placeholder="Search comments...")

        view = df
        if f_scen != "All":
            view = view[view["scenario"] == f_scen]
        if f_reac != "All":
            view = view[view["reaction"] == f_reac]
        if f_theme != "All":
            if f_theme.startswith("tag:"):
                t = f_theme[4:]
                rids = [rid for rid, entries in st.session_state.tags.items()
                        if any(e["tag"] == t for e in entries)]
                view = view[view["record_id"].isin(rids)]
            else:
                ckey = f_theme.split(" · ")[0]
                cl = st.session_state.clusters.get(ckey)
                if cl:
                    view = view[view["record_id"].isin(cl["record_ids"])]
        if f_kw.strip():
            view = view[view["comment"].str.contains(re.escape(f_kw.strip()),
                                                     case=False, na=False)]

        st.caption(f"{len(view)} comments · {view['response_id'].nunique()} unique "
                   f"response IDs match the current filters (of {len(df)} total).")

        with st.expander("Bulk tagging (select multiple comments)"):
            label_map = {rid: f'{rid} · {c[:70]}'
                         for rid, c in zip(view["record_id"], view["comment"])}
            sel = st.multiselect("Comments", view["record_id"].tolist(),
                                 format_func=lambda r: label_map.get(r, r))
            bt1, bt2 = st.columns(2)
            btag = bt1.selectbox("Tag", ["—"] + all_known_tags(), key="bulk-tag-sel")
            bnew = bt2.text_input("Or new tag", key="bulk-tag-new")
            if st.button("Apply tag to selected"):
                chosen = bnew.strip() or (btag if btag != "—" else "")
                if chosen and sel:
                    for rid in sel:
                        add_tag(rid, chosen, "human")
                    st.rerun()

        page_size = 15
        n_pages = max(1, (len(view) - 1) // page_size + 1)
        pg = st.number_input("Page", 1, n_pages, 1) if n_pages > 1 else 1
        for _, r in view.iloc[(pg - 1) * page_size: pg * page_size].iterrows():
            comment_card(r, "cx")
            memberships = [c["ai"]["name"] for c in st.session_state.clusters.values()
                           if r["record_id"] in c["record_ids"]
                           and c["status"] in ("ai-suggested", "validated")]
            if memberships:
                st.caption("In themes: " + " · ".join(memberships))

    # ---------------- THEMES ----------------
    with tab_themes:
        provider, _ = llm_provider()
        if provider is None:
            st.markdown('<div class="ces-note-warn">AI theme interpretation unavailable. '
                        'Configure an API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) to '
                        'enable AI-generated theme names and summaries. Local text '
                        'clustering still runs, and all human workflows remain '
                        'available.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 3])
        if c1.button("Run Thematic Analysis", type="primary"):
            with st.spinner("Clustering comments per scenario (TF-IDF + KMeans)"
                            + (" and asking the LLM to interpret each cluster…"
                               if provider else "…")):
                run_thematic_analysis()
            st.rerun()
        c2.caption("Comments are clustered independently for each scenario using "
                   "TF-IDF + KMeans. A sample of representative comments from each "
                   "cluster is sent to the LLM for a suggested name and summary. "
                   "Counts always come from the data, never from the LLM.")

        render_create_human_theme(df)

        if st.session_state.validating_cluster:
            cl = st.session_state.clusters.get(st.session_state.validating_cluster)
            if cl:
                render_validation_form(cl, df)
        if st.session_state.cluster_run_done:
            for scen in sorted(df["scenario"].unique()):
                scen_clusters = [c for c in st.session_state.clusters.values()
                                 if c["scenario"] == scen
                                 and c["status"] in ("ai-suggested", "validated")]
                if scen_clusters:
                    st.subheader(scen)
                    for cl in scen_clusters:
                        render_cluster_card(cl, df)
        elif not st.session_state.themes:
            st.caption("No thematic analysis yet — press Run Thematic Analysis.")

        if st.session_state.themes:
            st.subheader("Validated Themes")
            for th in st.session_state.themes:
                with st.container(border=True):
                    st.markdown(pills(("Human Validated", "validated"),
                                      *[(s, "gray") for s in th["scenarios"]]),
                                unsafe_allow_html=True)
                    st.markdown(f"**{th['theme_id']} — {th['name']}**")
                    st.markdown(f'<div class="ces-note-human"><b>Human interpretation:'
                                f'</b> {th["interpretation"] or "—"}</div>',
                                unsafe_allow_html=True)
                    if th["ai_original_name"]:
                        st.markdown(f'<div class="ces-note-ai"><b>AI original preserved:'
                                    f'</b> “{th["ai_original_name"]}” — '
                                    f'{th["ai_original_summary"]}</div>',
                                    unsafe_allow_html=True)
                    cts = th["counts"]
                    st.markdown(f'<div class="ces-meta">{len(th["record_ids"])} comments '
                                f'· {th["n_respondents"]} unique response IDs · Approve '
                                f'{cts["approve"]} · Disapprove {cts["disapprove"]} · '
                                f'None {cts["none"]} · Validated {th["validated"]}</div>',
                                unsafe_allow_html=True)

    # ---------------- COMPARE ----------------
    with tab_compare:
        st.subheader("Approve vs Disapprove")
        st.plotly_chart(reaction_chart(df), use_container_width=True,
                        key="compare-chart")
        st.dataframe(scenario_pct_table(df), use_container_width=True, hide_index=True)

        st.subheader("Themes by Scenario")
        if not st.session_state.cluster_run_done:
            st.caption("Run the thematic analysis in the Themes tab first.")
        else:
            scen_cols = st.columns(3)
            for i, scen in enumerate(["Scenario 1", "Scenario 2", "Scenario 3"]):
                with scen_cols[i]:
                    st.markdown(f"**{scen}**")
                    scl = [c for c in st.session_state.clusters.values()
                           if c["scenario"] == scen
                           and c["status"] in ("ai-suggested", "validated")]
                    if not scl:
                        st.caption("No comments / clusters.")
                    for c in scl:
                        n = max(1, c["n_comments"])
                        with st.container(border=True):
                            kind = "validated" if c["status"] == "validated" else "ai"
                            st.markdown(pills((c["ai"]["name"][:38], kind)),
                                        unsafe_allow_html=True)
                            st.markdown(
                                f'<div class="ces-meta">{c["n_comments"]} comments<br>'
                                f'Approve {c["counts"]["approve"]/n:.0%} · '
                                f'Disapprove {c["counts"]["disapprove"]/n:.0%}<br>'
                                f'{" · ".join(c["keywords"][:4])}</div>',
                                unsafe_allow_html=True)

            st.subheader("Patterns Across Scenarios")
            st.caption("AI-computed proposals based on keyword overlap between "
                       "scenario-level clusters. Cross-scenario insights remain "
                       "proposals until reviewed by a human.")
            for p in cross_scenario_patterns():
                reviewed = st.session_state.cross_reviews.get(p["key"])
                with st.container(border=True):
                    st.markdown(pills(
                        (p["relationship"], "ai"),
                        *[("Reviewed", "validated")] if reviewed else []),
                        unsafe_allow_html=True)
                    names = " + ".join(f'{c["ai"]["name"]} ({c["scenario"]})'
                                       for c in p["clusters"])
                    st.markdown(f"**{names}**")
                    st.markdown(f'<div class="ces-meta">Shared keywords: '
                                f'{", ".join(p["shared_keywords"][:6]) or "—"}</div>',
                                unsafe_allow_html=True)
                    with st.expander("Supporting comments"):
                        for c in p["clusters"]:
                            for rid in c["rep_ids"][:2]:
                                r = get_comment_row(rid)
                                if r is not None:
                                    comment_card(r, f"xs-sup-{p['key']}",
                                                 show_actions=False)
                    ctr = [rid for c in p["clusters"] for rid in c["counter_ids"][:2]]
                    if ctr:
                        with st.expander("Contradictory comments"):
                            for rid in ctr:
                                r = get_comment_row(rid)
                                if r is not None:
                                    comment_card(r, f"xs-ctr-{p['key']}",
                                                 show_actions=False)
                    if not reviewed:
                        if st.button("Review — mark as human-reviewed",
                                     key=f"xsrev-{p['key']}"):
                            st.session_state.cross_reviews[p["key"]] = {
                                "date": datetime.date.today().isoformat()}
                            st.rerun()
                    else:
                        rc1, rc2 = st.columns([1, 1])
                        if rc1.button("Save as Cross-Scenario Finding",
                                      key=f"xssave-{p['key']}"):
                            rids = sorted({rid for c in p["clusters"]
                                           for rid in c["record_ids"]})
                            sub = df[df["record_id"].isin(rids)]
                            add_evidence({
                                "type": "Cross-Scenario Finding",
                                "title": f'{p["relationship"]}: ' + " + ".join(
                                    c["ai"]["name"] for c in p["clusters"]),
                                "record_ids": rids,
                                "response_ids": sorted(sub["response_id"]
                                                       .unique().tolist()),
                                "scenarios": sorted(sub["scenario"].unique().tolist()),
                                "reaction": None,
                                "counts": reaction_counts(sub),
                                "source_files": sorted(sub["source_file"]
                                                       .unique().tolist()),
                                "original_comment": None, "selected_quote": None,
                                "theme": None,
                                "tags": p["shared_keywords"][:5],
                                "status": "Human Reviewed",
                            })
                            st.toast("Cross-scenario finding saved to Evidence Library")


# ----------------------------------------------------------------------------
# PAGE 03 — LIBRARIES
# ----------------------------------------------------------------------------

def render_traceability(ev):
    """The provenance chain for one evidence item."""
    chain = ("SOURCE XLSX FILE\n"
             "   ↓\n"
             "ORIGINAL COMMENT(S)\n"
             "   ↓\n"
             "RECORD ID(S)\n"
             "   ↓\n"
             "AI CLUSTER / TAG\n"
             "   ↓\n"
             "HUMAN REVIEW\n"
             "   ↓\n"
             "VALIDATED THEME / EVIDENCE")
    st.markdown(f'<div class="ces-chain">{chain}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ces-meta" style="margin-top:8px;">'
                f'<b>Evidence ID:</b> {ev["evidence_id"]} · <b>Type:</b> {ev["type"]} '
                f'· <b>Status:</b> {ev["status"]} · <b>Created:</b> {ev["created"]}<br>'
                f'<b>Source file(s):</b> {", ".join(ev.get("source_files", []) or ["—"])}<br>'
                f'<b>Scenario(s):</b> {", ".join(ev.get("scenarios", []) or ["—"])}<br>'
                f'<b>Record IDs:</b> {len(ev.get("record_ids", []))} · '
                f'<b>Response IDs:</b> {len(ev.get("response_ids", []))}</div>',
                unsafe_allow_html=True)
    if ev.get("selected_quote"):
        st.markdown(f'<div class="ces-note-human"><b>Selected quote:</b> '
                    f'“{ev["selected_quote"]}”</div>', unsafe_allow_html=True)
    if ev.get("original_comment"):
        st.markdown(f'<div class="ces-meta"><b>Complete original comment (always '
                    f'preserved):</b> “{ev["original_comment"]}”</div>',
                    unsafe_allow_html=True)
    if ev.get("ai_original_name"):
        st.markdown(f'<div class="ces-note-ai"><b>AI original preserved:</b> '
                    f'“{ev["ai_original_name"]}” — {ev["ai_original_summary"]}</div>',
                    unsafe_allow_html=True)
    if ev.get("interpretation"):
        st.markdown(f'<div class="ces-note-human"><b>Human interpretation:</b> '
                    f'{ev["interpretation"]}</div>', unsafe_allow_html=True)
    # return to the original comments
    df = st.session_state.combined
    if df is not None and ev.get("record_ids"):
        with st.expander(f"View source comments ({len(ev['record_ids'])})"):
            sub = df[df["record_id"].isin(ev["record_ids"])]
            for _, r in sub.head(50).iterrows():
                comment_card(r, f"trace-{ev['evidence_id']}", show_actions=False)
            if len(sub) > 50:
                st.caption(f"Showing 50 of {len(sub)}.")


def page_libraries():
    st.title("Libraries")
    tab_ev, tab_con = st.tabs(["Evidence Library", "Constraints Library"])

    with tab_ev:
        st.markdown(f'<p style="color:{C["text2"]};">Human-reviewed engagement evidence '
                    'ready to support planning decisions.</p>', unsafe_allow_html=True)
        if not st.session_state.evidence:
            st.caption("No evidence yet. Build evidence in the Insights Playground — "
                       "save comments, quotes, validated themes, quantitative findings, "
                       "or cross-scenario findings.")
        types = sorted({e["type"] for e in st.session_state.evidence})
        f_type = st.selectbox("Filter by type", ["All"] + types) if types else "All"
        for ev in st.session_state.evidence:
            if f_type != "All" and ev["type"] != f_type:
                continue
            with st.container(border=True):
                kind = "validated" if "Validated" in ev["status"] else "human"
                st.markdown(pills((ev["evidence_id"], "gray"),
                                  (ev["type"].upper(), kind),
                                  *[(s, "gray") for s in ev.get("scenarios", [])]),
                            unsafe_allow_html=True)
                st.markdown(f"**{ev['title']}**")
                if ev.get("counts"):
                    cts = ev["counts"]
                    st.markdown(f'<div class="ces-meta">{len(ev.get("record_ids", []))} '
                                f'related comments · Approve {cts["approve"]} · '
                                f'Disapprove {cts["disapprove"]} · None {cts["none"]}'
                                '</div>', unsafe_allow_html=True)
                if ev.get("tags"):
                    st.markdown("".join(pill(t, "human") for t in ev["tags"]),
                                unsafe_allow_html=True)
                with st.expander("View traceability"):
                    render_traceability(ev)
                if ev.get("constraints"):
                    st.markdown('<div class="ces-meta"><b>Linked constraints:</b> '
                                + ", ".join(ev["constraints"]) + "</div>",
                                unsafe_allow_html=True)

    with tab_con:
        st.markdown(f'<p style="color:{C["text2"]};">Documented conditions that shape '
                    'what is possible.</p>', unsafe_allow_html=True)
        if not st.session_state.constraints:
            st.caption("No constraints documented yet — add them in Data + Context.")
        for con in st.session_state.constraints:
            related_themes = [t["theme_id"] + " " + t["name"]
                              for t in st.session_state.themes
                              if con["id"] in t.get("constraints", [])]
            related_dec = [d["id"] for d in st.session_state.decisions
                           if con["id"] in d.get("constraints", [])]
            with st.container(border=True):
                st.markdown(pills((con["id"], "gray"),
                                  (con["type"], "conflict" if "Legal" in con["type"] or
                                   "Voter" in con["type"] else "human"),
                                  (con["status"], "validated")),
                            unsafe_allow_html=True)
                st.markdown(f"**{con['name']}**")
                if con["description"]:
                    st.write(con["description"])
                st.markdown(f'<div class="ces-meta">Source: {con["source"] or "—"} · '
                            f'Phase: {con["phase"] or "—"}<br>'
                            f'Related themes: {", ".join(related_themes) or "none"} · '
                            f'Related decisions: {", ".join(related_dec) or "none"}'
                            '</div>', unsafe_allow_html=True)
                with st.expander("Traceability"):
                    chain = ("SOURCE DOCUMENT\n   ↓\nDOCUMENTED CONSTRAINT\n   ↓\n"
                             "HUMAN VALIDATION\n   ↓\nRELEVANT DECISION")
                    st.markdown(f'<div class="ces-chain">{chain}</div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div class="ces-meta"><b>Source:</b> '
                                f'{con["source"] or "—"}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE 04 — DECISION TRAILS
# ----------------------------------------------------------------------------

DECISION_DIAGRAM = (
    "COMMUNITY EVIDENCE ─────┐\n"
    "                        │\n"
    "PROJECT CONSTRAINTS ────┼→ CONSIDERATION → DECISION\n"
    "                        │\n"
    "CONFLICTING INPUT ──────┘"
)


def page_decisions():
    st.title("Decision Trails")
    st.markdown(f'<p style="color:{C["text2"]};">Document how evidence, constraints, '
                'trade-offs, and judgment shaped a planning decision. Inputs are '
                '<b>considered</b> — evidence does not automatically cause a '
                'decision.</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="ces-chain">{DECISION_DIAGRAM}</div>',
                unsafe_allow_html=True)

    ev_opts = {e["evidence_id"]: f'{e["evidence_id"]} · {e["type"]} · {e["title"][:60]}'
               for e in st.session_state.evidence}
    con_opts = {c["id"]: f'{c["id"]} · {c["name"]}'
                for c in st.session_state.constraints}

    with st.expander("＋ New Decision", expanded=(len(st.session_state.decisions) == 0)):
        with st.form("decision-form", clear_on_submit=True):
            d_name = st.text_input("Decision Name")
            d_desc = st.text_area("Decision Description")
            c1, c2 = st.columns(2)
            d_date = c1.text_input("Date", datetime.date.today().isoformat())
            d_maker = c2.text_input("Decision Maker")
            d_alts = st.text_area("Alternatives Considered", height=68)
            d_rationale = st.text_area("Rationale", height=68)
            st.markdown("**Community Evidence**")
            d_evidence = st.multiselect("Add Evidence", list(ev_opts),
                                        format_func=lambda k: ev_opts[k])
            st.markdown("**Project Constraints**")
            d_constraints = st.multiselect("Add Constraint", list(con_opts),
                                           format_func=lambda k: con_opts[k])
            st.markdown("**Trade-offs / Conflicts**")
            d_conflict_ev = st.multiselect("Conflicting or trade-off input "
                                           "(from Evidence Library)",
                                           list(ev_opts),
                                           format_func=lambda k: ev_opts[k],
                                           key="dec-conflict")
            d_tradeoffs = st.text_area("Trade-off notes", height=68)
            if st.form_submit_button("Save Decision", type="primary"):
                if not d_name.strip():
                    st.error("The decision needs a name.")
                else:
                    st.session_state.decision_seq += 1
                    st.session_state.decisions.append({
                        "id": f"DEC-{st.session_state.decision_seq:03d}",
                        "name": d_name.strip(), "description": d_desc.strip(),
                        "date": d_date, "maker": d_maker.strip(),
                        "alternatives": d_alts.strip(),
                        "rationale": d_rationale.strip(),
                        "evidence": d_evidence, "constraints": d_constraints,
                        "conflicts": d_conflict_ev, "tradeoffs": d_tradeoffs.strip(),
                    })
                    st.rerun()

    for dec in st.session_state.decisions:
        with st.container(border=True):
            st.markdown(pills((dec["id"], "gray"), ("DECISION", "human")),
                        unsafe_allow_html=True)
            st.markdown(f"### {dec['name']}")
            if dec["description"]:
                st.write(dec["description"])
            st.markdown(f'<div class="ces-meta">Date: {dec["date"]} · '
                        f'Decision maker: {dec["maker"] or "—"}</div>',
                        unsafe_allow_html=True)
            col_e, col_c, col_x = st.columns(3)
            with col_e:
                st.markdown("**Community Evidence**")
                for eid in dec["evidence"]:
                    st.markdown(pill(eid, "validated") +
                                f'<span class="ces-meta"> '
                                f'{ev_opts.get(eid, eid)[len(eid) + 3:]}</span>',
                                unsafe_allow_html=True)
                if not dec["evidence"]:
                    st.caption("none attached")
            with col_c:
                st.markdown("**Project Constraints**")
                for cid in dec["constraints"]:
                    st.markdown(pill(cid, "review") +
                                f'<span class="ces-meta"> '
                                f'{con_opts.get(cid, cid)[len(cid) + 3:]}</span>',
                                unsafe_allow_html=True)
                if not dec["constraints"]:
                    st.caption("none attached")
            with col_x:
                st.markdown("**Trade-offs / Conflicts**")
                for eid in dec["conflicts"]:
                    st.markdown(pill(eid, "conflict"), unsafe_allow_html=True)
                if dec["tradeoffs"]:
                    st.caption(dec["tradeoffs"])
                if not dec["conflicts"] and not dec["tradeoffs"]:
                    st.caption("none documented")
            if dec["alternatives"]:
                st.markdown(f'<div class="ces-meta"><b>Alternatives considered:</b> '
                            f'{dec["alternatives"]}</div>', unsafe_allow_html=True)
            if dec["rationale"]:
                st.markdown(f'<div class="ces-note-human"><b>Rationale:</b> '
                            f'{dec["rationale"]}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------

def main():
    init_state()
    with st.sidebar:
        st.markdown("## Civic Evidence Studio")
        st.caption(st.session_state.metadata["project"])
        page = st.radio("Navigate", ["01 Data + Context", "02 Insights Playground",
                                     "03 Libraries", "04 Decision Trails"],
                        label_visibility="collapsed")
        st.divider()
        df = st.session_state.combined
        st.caption(
            f"Dataset: {'ready — ' + str(len(df)) + ' comments' if df is not None else 'not processed'}\n\n"
            f"Evidence items: {len(st.session_state.evidence)}\n\n"
            f"Constraints: {len(st.session_state.constraints)}\n\n"
            f"Decisions: {len(st.session_state.decisions)}")
        provider, _ = llm_provider()
        st.markdown(pill("AI: " + (provider if provider else "not configured"),
                         "ai" if provider else "review"), unsafe_allow_html=True)
        st.caption("Traceability is preserved end-to-end — from source XLSX row to "
                   "validated evidence to decision.")

    if page == "01 Data + Context":
        page_data_context()
    elif page == "02 Insights Playground":
        page_insights()
    elif page == "03 Libraries":
        page_libraries()
    else:
        page_decisions()


main()
