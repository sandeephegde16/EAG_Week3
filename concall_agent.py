import json
import re
import os
import argparse
from functools import partial
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from logger import LLMCallLogger
from prompts import SYSTEM_PROMPT
from tools import (
    extract_company_info,
    extract_kpis,
    derive_thesis,
    analyze_what_changed,
    analyze_thesis_status,
    identify_red_flags,
    make_investment_decision,
    generate_watchlist,
)

load_dotenv()

# --- Configuration ---
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL  = "claude-haiku-4-5-20251001"

# --- Logger Setup ---
logger = LLMCallLogger("llm_calls.json")
logger.clear()

# --- Tools Registration ---
tools = {
    "extract_company_info":     extract_company_info,
    "extract_kpis":             partial(extract_kpis,             logger=logger, client=client),
    "derive_thesis":            partial(derive_thesis,            logger=logger, client=client),
    "analyze_what_changed":     partial(analyze_what_changed,     logger=logger, client=client),
    "analyze_thesis_status":    partial(analyze_thesis_status,    logger=logger, client=client),
    "identify_red_flags":       partial(identify_red_flags,       logger=logger, client=client),
    "make_investment_decision":  make_investment_decision,
    "generate_watchlist":       partial(generate_watchlist,       logger=logger, client=client),
}


# Maps tool name → key used to store its result in partial_results
TOOL_RESULT_KEYS = {
    "extract_company_info":    "company_info",
    "extract_kpis":            "kpis",
    "derive_thesis":           "derived_thesis",
    "analyze_what_changed":    "what_changed",
    "analyze_thesis_status":   "thesis_status",
    "identify_red_flags":      "red_flags",
    "make_investment_decision": "decision",
    "generate_watchlist":      "watchlist",
}
ALL_TOOLS = set(TOOL_RESULT_KEYS.keys())


# --- Response Parser ---
def parse_llm_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # LLM returned multiple JSON objects — take only the first one
        if "Extra data" in str(e):
            return json.loads(text[:e.pos])
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse: {text[:300]}")


# --- Prompt Builder ---
def build_prompt(messages: list) -> str:
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += msg["content"] + "\n\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n\n"
        elif msg["role"] == "tool":
            prompt += f"Tool Result: {msg['content']}\n\n"
    return prompt


# --- Agent Loop ---
def run_agent(transcript: str, company_name: str, max_iterations: int = 20):
    print(f"\n{'='*60}")
    print(f"CONCALL AGENT: {company_name}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Company: {company_name}\n\nTranscript:\n{transcript}"},
    ]

    partial_results = {}
    called_tools    = set()

    for iteration in range(max_iterations):

        # All 8 tools done — compile report directly without another LLM call
        if called_tools == ALL_TOOLS:
            print(f"\n[AGENT] All {len(ALL_TOOLS)} tools executed. Compiling report.")
            format_report(partial_results, company_name)
            print(f"\n{logger.summary()}")
            return partial_results

        # Token management: transcript consumed by iteration 5, replace with placeholder
        if iteration == 5:
            messages[1]["content"] = (
                f"Company: {company_name}\n\n"
                f"[Transcript processed. All extractions are in the tool results above.]"
            )

        print(f"\n--- Iteration {iteration + 1} | Tools done: {len(called_tools)}/{len(ALL_TOOLS)} ---")
        prompt = build_prompt(messages)

        # LLM decides what to do next
        response      = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text
        print(f"[LLM] {response_text[:200]}{'...' if len(response_text) > 200 else ''}")

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens":  response.usage.input_tokens + response.usage.output_tokens,
        }
        print(f"[TOKENS] {usage}")

        # Parse LLM response
        try:
            parsed = parse_llm_response(response_text)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[PARSE ERROR] {e}")
            logger.log(call_type="main_loop", context=f"iteration_{iteration+1}",
                       prompt=prompt, raw_response=response_text, error=str(e), usage=usage)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": "Please respond with valid JSON only. No markdown."})
            continue

        logger.log(call_type="main_loop", context=f"iteration_{iteration+1}",
                   prompt=prompt, raw_response=response_text, parsed_response=parsed, usage=usage)

        # LLM returned final answer early — use it if present
        if "answer" in parsed:
            print(f"\n[AGENT] LLM returned final answer at iteration {iteration+1}.")
            format_report(parsed["answer"], company_name)
            print(f"\n{logger.summary()}")
            return parsed["answer"]

        # LLM decided on a tool call
        if "tool_name" in parsed:
            tool_name = parsed["tool_name"]
            tool_args = parsed.get("tool_arguments", {})
            print(f"[TOOL CALL] {tool_name}({list(tool_args.keys())})")

            if tool_name not in tools:
                err = json.dumps({"error": f"Unknown tool '{tool_name}'. Valid: {list(tools.keys())}"})
                print(f"[TOOL ERROR] Unknown tool: {tool_name}")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "tool",      "content": err})
                continue

            # Inject transcript automatically — LLM never generates it
            if "transcript" in tool_args or tool_name in {
                "extract_company_info", "extract_kpis", "derive_thesis",
                "analyze_what_changed", "analyze_thesis_status",
                "identify_red_flags", "generate_watchlist",
            }:
                tool_args["transcript"] = transcript

            try:
                tool_result = tools[tool_name](**tool_args)
                print(f"[TOOL RESULT] {tool_result[:200]}{'...' if len(tool_result) > 200 else ''}")
                called_tools.add(tool_name)
                result_key = TOOL_RESULT_KEYS.get(tool_name, tool_name)
                partial_results[result_key] = tool_result
            except Exception as e:
                tool_result = json.dumps({"error": f"{tool_name} failed: {e}"})
                print(f"[TOOL ERROR] {e}")

            logger.log(call_type="main_loop", context=f"iteration_{iteration+1}",
                       prompt=prompt, raw_response=response_text, parsed_response=parsed,
                       tool_name=tool_name, tool_args=tool_args, tool_result=tool_result, usage=usage)

            # Result feeds back into conversation — LLM reads it next iteration
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "tool",      "content": tool_result})
            continue

        # Unexpected format
        print(f"[WARN] Unexpected response format.")
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": "Respond ONLY with a tool call JSON or final answer JSON."})

    print(f"\n[AGENT] Max iterations reached. Returning partial results.")
    format_report(partial_results, company_name)
    print(logger.summary())
    return partial_results


# --- Report Formatter ---
def format_report(answer: dict, company_name: str):
    info           = answer.get("company_info",   {})
    kpis           = answer.get("kpis",           {})
    derived_thesis = answer.get("derived_thesis", {})
    changed        = answer.get("what_changed",   {})
    thesis         = answer.get("thesis_status",  {})
    flags          = answer.get("red_flags",      {})
    decision       = answer.get("decision",       {})
    watchlist      = answer.get("watchlist",      {})

    if isinstance(info,           str): info           = json.loads(info)
    if isinstance(kpis,           str): kpis           = json.loads(kpis)
    if isinstance(derived_thesis, str): derived_thesis = json.loads(derived_thesis)
    if isinstance(changed,        str): changed        = json.loads(changed)
    if isinstance(thesis,         str): thesis         = json.loads(thesis)
    if isinstance(flags,          str): flags          = json.loads(flags)
    if isinstance(decision,       str): decision       = json.loads(decision)
    if isinstance(watchlist,      str): watchlist      = json.loads(watchlist)

    ticker = info.get("ticker", "N/A")
    quarter= info.get("quarter","N/A")
    price  = info.get("current_price", "N/A")

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CONCALL ANALYSIS REPORT: {company_name} [{ticker}]")
    print(f"  Quarter: {quarter}  |  CMP: {price}")
    print(sep)

    print("\nDERIVED INVESTMENT THESIS")
    print(f"  Engine : {derived_thesis.get('engine', 'N/A')}")
    print(f"  Moat   : {derived_thesis.get('moat', 'N/A')}")
    print(f"  Breaks : {derived_thesis.get('thesis_break_conditions', 'N/A')}")

    print("\nWHAT CHANGED THIS QUARTER")
    print(f"  Revenue  : {changed.get('revenue_change', 'N/A')}")
    print(f"  Margins  : {changed.get('margin_change',  'N/A')}")
    initiatives = changed.get("new_initiatives", [])
    if initiatives:
        print(f"  New Items: {' | '.join(initiatives)}")
    print(f"  Beat/Miss: {changed.get('beat_or_miss',   'N/A')}")
    print(f"  Tone     : {changed.get('management_tone','N/A')}")

    print("\nTHESIS STATUS")
    for pillar in ["engine", "moat", "management"]:
        p = thesis.get(pillar, {})
        rating   = p.get("rating",   "N/A")
        evidence = p.get("evidence", "N/A")
        print(f"  {pillar.title():<12} [{rating:<14}] {evidence}")

    print("\nKEY PERFORMANCE INDICATORS")
    rev  = kpis.get("revenue", {})
    pat  = kpis.get("pat",     {})
    ebit = kpis.get("ebitda",  {})
    if isinstance(rev,  str): rev  = {"value": rev}
    if isinstance(pat,  str): pat  = {"value": pat}
    if isinstance(ebit, str): ebit = {"value": ebit}
    print(f"  Revenue      : {rev.get('value','N/A')} {rev.get('unit','')}   YoY: {rev.get('yoy_growth','N/A')}")
    print(f"  PAT          : {pat.get('value','N/A')} {pat.get('unit','')}   YoY: {pat.get('yoy_growth','N/A')}")
    print(f"  EBITDA       : {ebit.get('value','N/A')} {ebit.get('unit','')}   Margin: {ebit.get('margin','N/A')}")
    print(f"  Guidance     : {kpis.get('guidance','N/A')}")

    severity   = flags.get("severity",   "N/A")
    flag_count = flags.get("flag_count", 0)
    print(f"\nRED FLAGS  [Severity: {severity} | Count: {flag_count}]")
    flag_keys = ["guidance_cut", "margin_compression", "auditor_concern",
                 "management_deflection", "customer_concentration"]
    for k in flag_keys:
        f = flags.get(k, {})
        detected = f.get("detected", False)
        detail   = f.get("detail",   "")
        icon     = "[!]" if detected else "[ ]"
        label    = k.replace("_", " ").title()
        print(f"  {icon} {label:<28} {detail or ''}")

    dec_val   = decision.get("decision", "N/A")
    score     = decision.get("score",    "N/A")
    breakdown = decision.get("score_breakdown", {})
    rationale = decision.get("rationale", "N/A")
    print(f"\nINVESTMENT DECISION: {dec_val}  (Score: {score})")
    print(f"  Thesis : {breakdown.get('thesis_score','N/A')}")
    print(f"  Flags  : {breakdown.get('flag_penalty','N/A')}")
    print(f"  KPI    : {breakdown.get('kpi_momentum','N/A')}")
    print(f"  Rationale: {rationale}")

    print("\nNEXT QUARTER WATCHLIST")
    for item in watchlist.get("metrics_to_watch", []):
        print(f"  Metric  : {item}")
    for item in watchlist.get("events_to_monitor", []):
        print(f"  Event   : {item}")
    for item in watchlist.get("milestones", []):
        print(f"  Milestone: {item}")
    for item in watchlist.get("catalyst_dates", []):
        print(f"  Date    : {item}")

    print(f"\n{sep}\n")


# --- Transcript Loader ---
def load_transcript(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages  = [page.extract_text() or "" for page in reader.pages]
        text   = "\n".join(pages)
        print(f"[INFO] PDF loaded: {len(reader.pages)} pages, {len(text):,} characters")
        return text

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[INFO] TXT loaded: {len(text):,} characters")
    return text


# --- Run it! ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Company Concall Analyser Agent")
    parser.add_argument("--transcript", required=True, help="Path to the concall transcript (.pdf or .txt)")
    parser.add_argument("--company",    required=True, help="Company name e.g. 'Infosys Ltd'")
    args = parser.parse_args()

    transcript_text = load_transcript(args.transcript)

    run_agent(
        transcript   = transcript_text,
        company_name = args.company,
    )
