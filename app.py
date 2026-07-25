import json
import os
import re
import sys
from datetime import datetime
from typing import Any

import streamlit as st


def extract_json_object(text: str) -> dict:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {"route": "direct", "rationale": "Fallback route due to non-text output."}

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace:
            text = brace.group(0)

    try:
        return json.loads(text)
    except Exception:
        return {"route": "direct", "rationale": "Fallback route due to JSON parse error."}


@st.cache_resource(show_spinner=False)
def load_ai_dependencies() -> dict[str, Any]:
    from crewai import Agent, Crew, Task
    from crewai_tools import PDFSearchTool
    from tavily import TavilyClient

    return {
        "Agent": Agent,
        "Crew": Crew,
        "Task": Task,
        "PDFSearchTool": PDFSearchTool,
        "TavilyClient": TavilyClient,
    }


@st.cache_resource(show_spinner=False)
def build_tools(pdf_path: str, deps: dict[str, Any], tavily_key: str):
    PDFSearchTool = deps["PDFSearchTool"]
    TavilyClient = deps["TavilyClient"]
    pdf_tool = PDFSearchTool(pdf=pdf_path)
    web_tool = TavilyClient(api_key=tavily_key)
    return pdf_tool, web_tool


def build_agents(pdf_tool, web_tool, deps: dict[str, Any]):
    Agent = deps["Agent"]
    router_agent = Agent(
        role="Router Agent",
        goal="Classify a user question into pdf, web, or direct retrieval strategy.",
        backstory=(
            "You decide the best information source for each query. "
            "Always return strict JSON with keys: route, rationale."
        ),
        verbose=False,
        allow_delegation=False,
        tools=[],
    )

    retriever_agent = Agent(
        role="Retriever Agent",
        goal="Use the selected tool path and return a grounded concise answer with source notes.",
        backstory=(
            "You retrieve information from the selected tool or answer directly, "
            "then provide a concise final response."
        ),
        verbose=False,
        allow_delegation=False,
        tools=[pdf_tool],
    )
    return router_agent, retriever_agent


def log_event(step: str, actor: str, content):
    st.session_state.trace_log.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "step": step,
            "actor": actor,
            "content": content,
        }
    )


def route_question(question: str, router_agent, deps: dict[str, Any]):
    Task = deps["Task"]
    Crew = deps["Crew"]
    routing_prompt = f"""
Classify the question into one of: pdf, web, direct.

Rules:
- pdf: if answer likely appears in the research PDF
- web: if question needs current/external information
- direct: if conceptual reasoning is sufficient

Return only valid JSON:
{{"route": "pdf|web|direct", "rationale": "short reason"}}

Question: {question}
"""

    task = Task(
        description=routing_prompt,
        expected_output='A strict JSON object with keys "route" and "rationale".',
        agent=router_agent,
    )
    crew = Crew(agents=[router_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    decision = extract_json_object(str(result))

    route = decision.get("route", "direct").strip().lower()
    if route not in {"pdf", "web", "direct"}:
        route = "direct"
    rationale = decision.get("rationale", "No rationale provided.")

    log_event("route", "Router Agent", {"question": question, "route": route, "rationale": rationale})
    return route, rationale


def retrieve_answer(question: str, route: str, retriever_agent, pdf_tool, web_tool, deps: dict[str, Any]):
    Task = deps["Task"]
    Crew = deps["Crew"]
    if route == "pdf":
        description = f"""
Use the PDF search tool to answer the question.
Cite evidence from retrieved PDF content.

Question: {question}
"""
        tools = [pdf_tool]
    elif route == "web":
        web_results = web_tool.search(
            query=question,
            max_results=5,
            include_answer=True,
            include_raw_content=False,
        )
        answer_hint = web_results.get("answer", "")
        result_lines = []
        for idx, item in enumerate(web_results.get("results", []), start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            result_lines.append(f"{idx}. {title} ({url})\\n{content}")
        web_context = "\\n\\n".join(result_lines)

        description = f"""
Use the provided web search evidence to answer the question.
Prefer reliable sources and include source references from the evidence below.

Question: {question}

Web Answer Hint:
{answer_hint}

Web Evidence:
{web_context}
"""
        tools = []
    else:
        description = f"""
Answer directly using reasoning.
If uncertain, state assumptions clearly.

Question: {question}
"""
        tools = []

    task = Task(
        description=description,
        expected_output=(
            "A concise answer and a short Source Notes section. "
            "If no tool is used, write: Source Notes: Direct LLM reasoning."
        ),
        agent=retriever_agent,
    )
    retriever_agent.tools = tools

    crew = Crew(agents=[retriever_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    answer = str(result)

    log_event("retrieve", "Retriever Agent", {"route": route, "answer_preview": answer[:400]})
    return answer


def main():
    st.set_page_config(page_title="Agentic RAG Demo", page_icon="📚", layout="wide")
    st.title("Agentic RAG: Router-Retriever Demo")
    st.caption("CrewAI multi-agent routing across PDF search, web search, and direct generation")

    if "trace_log" not in st.session_state:
        st.session_state.trace_log = []

    default_pdf = os.path.abspath(os.path.join(os.path.dirname(__file__), "trasformer_research_paper-dataset.pdf"))

    if sys.version_info >= (3, 12):
        st.error(
            "This app requires Python 3.11 for CrewAI/Chroma compatibility. "
            "In Streamlit app settings, set Python to 3.11 and reboot."
        )
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        pdf_path = st.text_input("PDF path", value=default_pdf)

        env_openai = os.getenv("OPENAI_API_KEY", "")
        env_tavily = os.getenv("TAVILY_API_KEY", "")
        sec_openai = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
        sec_tavily = st.secrets.get("TAVILY_API_KEY", "") if hasattr(st, "secrets") else ""

        openai_key = sec_openai or env_openai
        tavily_key = sec_tavily or env_tavily

        st.markdown("Optional override keys for local testing:")
        openai_input = st.text_input("OPENAI_API_KEY", type="password", value="")
        tavily_input = st.text_input("TAVILY_API_KEY", type="password", value="")

        if openai_input:
            os.environ["OPENAI_API_KEY"] = openai_input
            openai_key = openai_input
        if tavily_input:
            os.environ["TAVILY_API_KEY"] = tavily_input
            tavily_key = tavily_input

        st.write("OpenAI key detected:", bool(openai_key))
        st.write("Tavily key detected:", bool(tavily_key))

    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at: {pdf_path}")
        st.stop()

    if not openai_key:
        st.error("Missing OPENAI_API_KEY. Add it to environment or Streamlit secrets.")
        st.stop()

    try:
        deps = load_ai_dependencies()
        pdf_tool, web_tool = build_tools(pdf_path, deps, tavily_key)
        router_agent, retriever_agent = build_agents(pdf_tool, web_tool, deps)
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        question = st.text_area(
            "Ask a question",
            value="According to the transformer research paper, what is novel about the architecture?",
            height=120,
        )
        run = st.button("Run Agentic RAG", type="primary")

    with col2:
        st.markdown("### Suggested prompts")
        st.markdown("- PDF: Summarize the paper's core contribution")
        st.markdown("- Web: What are recent LLM releases this week?")
        st.markdown("- Direct: When is direct generation better than retrieval?")

    if run:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        log_event("input", "User", question)
        with st.spinner("Routing and retrieving answer..."):
            route, rationale = route_question(question, router_agent, deps)
            answer = retrieve_answer(question, route, retriever_agent, pdf_tool, web_tool, deps)

        st.subheader("Result")
        st.markdown(f"**Route:** {route}")
        st.markdown(f"**Rationale:** {rationale}")
        st.markdown("**Answer:**")
        st.write(answer)

    st.subheader("Interaction Trace")
    if st.session_state.trace_log:
        st.dataframe(st.session_state.trace_log, use_container_width=True)
        trace_json = json.dumps(st.session_state.trace_log, indent=2)
        st.download_button(
            "Download trace_log.json",
            data=trace_json,
            file_name="trace_log.json",
            mime="application/json",
        )
    else:
        st.info("No trace entries yet. Run a question to populate logs.")


if __name__ == "__main__":
    main()
