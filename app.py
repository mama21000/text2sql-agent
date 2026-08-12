"""
Streamlit frontend.

    streamlit run app.py

Shows every attempt the agent made, not just the final answer - the repair loop
is the interesting part, so it should be visible.
"""

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import Text2SQLAgent

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")


@st.cache_resource
def get_agent() -> Text2SQLAgent:
    return Text2SQLAgent()


def available_databases() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.sqlite"))


def run(coro):
    """Streamlit has no running loop, so drive the coroutine on a fresh one."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


st.title("Text-to-SQL Agent")
st.caption(
    "Asks a question, writes SQL, runs it, and repairs itself based on what "
    "kind of failure came back."
)

databases = available_databases()
if not databases:
    st.error(f"No .sqlite files in {DATA_DIR}/. See data/README.md for setup.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    db_name = st.selectbox("Database", databases)
    show_schema = st.checkbox("Show schema", value=False)

    if show_schema:
        agent = get_agent()
        schema = run(agent._get_schema(db_name))
        st.code(schema, language="text")

question = st.text_input(
    "Question",
    placeholder="How many schools in Alameda County have an average math score above 600?",
)

if st.button("Ask", type="primary") and question:
    agent = get_agent()

    with st.spinner("Working..."):
        final = run(agent.ask(question, db_name))

    history = final.get("history", [])
    attempts = len(history)

    if final.get("finished"):
        if attempts == 1:
            st.success("Answered on the first attempt.")
        else:
            st.success(f"Answered after {attempts} attempts — "
                       f"the repair loop recovered this one.")
    else:
        st.error(final.get("give_up_reason", "Could not answer."))

    result = final.get("result", {})
    if result.get("success") and result.get("rows"):
        st.subheader("Result")
        st.dataframe(
            pd.DataFrame(result["rows"], columns=result.get("columns")),
            use_container_width=True,
        )
        if result.get("truncated"):
            st.caption("Showing the first 50 rows.")
    elif result.get("success"):
        st.info("The query ran but returned no rows.")

    st.subheader("Final SQL")
    st.code(final.get("sql", ""), language="sql")

    if attempts > 1:
        st.subheader("How it got there")
        for step in history:
            label = (f"Attempt {step['attempt']} — "
                     f"{'succeeded' if step['success'] else 'failed'}")
            with st.expander(label, expanded=not step["success"]):
                st.code(step["sql"], language="sql")
                if step.get("error"):
                    st.error(step["error"])
                elif step.get("row_count") is not None:
                    st.caption(f"{step['row_count']} rows")

        if final.get("findings"):
            with st.expander("What the agent learned between attempts"):
                for f in final["findings"]:
                    st.markdown(f"- {f}")
