import json
import re
from logger import LLMCallLogger
from prompts import (
    DERIVE_THESIS_PROMPT,
    WHAT_CHANGED_PROMPT,
    THESIS_STATUS_PROMPT,
    RED_FLAGS_PROMPT,
    WATCHLIST_PROMPT,
)


# ---------------------------------------------------------------------------
# Shared sub-LLM helper
# ---------------------------------------------------------------------------

def _call_sub_llm(prompt: str, context: str, logger: LLMCallLogger, client) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text

    usage = {
        "prompt_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens":  response.usage.input_tokens + response.usage.output_tokens,
    }

    logger.log(
        call_type="tool_sub_call",
        context=context,
        prompt=prompt,
        raw_response=raw,
        usage=usage,
    )
    return raw


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
        if text.startswith("json"):
            text = text[4:].strip()
    return text


def _parse_json_safe(text: str) -> dict:
    text = _strip_markdown(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from sub-LLM response: {text[:200]}")


# ---------------------------------------------------------------------------
# Tool 1 — extract_company_info  [DETERMINISTIC]
# ---------------------------------------------------------------------------

def extract_company_info(transcript: str) -> str:
    lines = transcript[:3000]

    # Ticker symbol: NSE: XXXX or BSE: XXXXXX
    ticker_match = re.search(
        r'\b(?:NSE|BSE)\s*[:/]\s*([A-Z]{2,10})\b', transcript, re.IGNORECASE
    )
    ticker = ticker_match.group(1).upper() if ticker_match else "N/A"

    # Current market price / CMP
    price_match = re.search(
        r'(?:CMP|current\s+market\s+price|trading\s+at|price)[^\d]*?([\d,]+(?:\.\d+)?)',
        transcript, re.IGNORECASE
    )
    price = price_match.group(1).replace(",", "") if price_match else "N/A"

    # Quarter: Q1/Q2/Q3/Q4 FY24 / FY2024
    quarter_match = re.search(
        r'\b(Q[1-4]\s*(?:FY|FY\s*)?\s*(?:20)?\d{2}(?:\s*[-–]\s*(?:20)?\d{2})?)\b',
        transcript, re.IGNORECASE
    )
    quarter = quarter_match.group(1).strip() if quarter_match else "N/A"

    # Company name: look for "Limited", "Ltd", "Corporation", "Corp" in first 3000 chars
    name_match = re.search(
        r'([A-Z][A-Za-z\s&]+(?:Limited|Ltd|Corporation|Corp|Technologies|Infosystems|Industries))',
        lines
    )
    company_name = name_match.group(1).strip() if name_match else "N/A"

    result = {
        "company_name": company_name,
        "ticker": ticker,
        "current_price": price,
        "quarter": quarter,
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 2 — extract_kpis  [DETERMINISTIC + sub-LLM fallback]
# ---------------------------------------------------------------------------

def extract_kpis(transcript: str, logger: LLMCallLogger, client) -> str:
    t = transcript.lower()

    def find_amount(patterns):
        for pat in patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                return m.group(1).replace(",", ""), m.group(2).upper() if len(m.groups()) > 1 else ""
        return None, None

    def find_pct(patterns):
        for pat in patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    revenue_val, revenue_unit = find_amount([
        r'revenue[^₹\d]*?([\d,]+(?:\.\d+)?)\s*(cr|crore|mn|million|bn|billion)',
        r'total\s+income[^₹\d]*?([\d,]+(?:\.\d+)?)\s*(cr|crore|mn|million)',
        r'([\d,]+(?:\.\d+)?)\s*(cr|crore|mn)\s+(?:in\s+)?revenue',
    ])

    pat_val, pat_unit = find_amount([
        r'(?:pat|profit\s+after\s+tax|net\s+profit)[^₹\d]*?([\d,]+(?:\.\d+)?)\s*(cr|crore|mn|million)',
        r'net\s+profit[^₹\d]*?([\d,]+(?:\.\d+)?)\s*(cr|crore|mn)',
    ])

    ebitda_val, ebitda_unit = find_amount([
        r'ebitda[^₹\d]*?([\d,]+(?:\.\d+)?)\s*(cr|crore|mn|million)',
    ])

    ebitda_margin = find_pct([
        r'ebitda\s+margin[^\d]*?(\d+\.?\d*)\s*%',
        r'operating\s+margin[^\d]*?(\d+\.?\d*)\s*%',
    ])

    yoy_growth = find_pct([
        r'(?:revenue|income)[^\d]*?grew[^\d]*?(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*%\s*(?:yoy|year[\s-]on[\s-]year)',
        r'growth\s+of\s+(\d+\.?\d*)\s*%',
    ])

    guidance = None
    guidance_match = re.search(
        r'guid(?:ance|e)[^\d]*?([\d,]+(?:\.\d+)?(?:\s*(?:cr|crore|mn|%|billion))?[^.]{0,60})',
        t, re.IGNORECASE
    )
    if guidance_match:
        guidance = guidance_match.group(1).strip()

    kpis = {
        "revenue": {"value": revenue_val, "unit": revenue_unit, "yoy_growth": yoy_growth},
        "pat":     {"value": pat_val,     "unit": pat_unit},
        "ebitda":  {"value": ebitda_val,  "unit": ebitda_unit, "margin": ebitda_margin},
        "guidance": guidance,
    }

    # Sub-LLM fallback only for fields that regex could not find
    missing = [k for k, v in kpis.items() if k != "guidance" and all(
        x is None for x in (v.values() if isinstance(v, dict) else [v])
    )]
    if kpis["guidance"] is None:
        missing.append("guidance")

    if missing:
        fallback_prompt = (
            f"Extract ONLY these financial KPIs from the transcript below. "
            f"Fields needed: {', '.join(missing)}.\n"
            f"Respond in JSON with exactly those keys. No markdown.\n\n"
            f"Transcript:\n{transcript[:6000]}"
        )
        try:
            raw = _call_sub_llm(fallback_prompt, "tool:extract_kpis_fallback", logger, client)
            extra = _parse_json_safe(raw)
            for field in missing:
                if field in extra:
                    kpis[field] = extra[field]
        except Exception as e:
            kpis["_fallback_error"] = str(e)

    return json.dumps(kpis)


# ---------------------------------------------------------------------------
# Tool 3 — derive_thesis  [SUB-LLM]
# ---------------------------------------------------------------------------

def derive_thesis(transcript: str, company_name: str,
                  logger: LLMCallLogger, client) -> str:
    prompt = DERIVE_THESIS_PROMPT.format(
        company_name=company_name,
        transcript=transcript[:8000],
    )
    raw = _call_sub_llm(prompt, "tool:derive_thesis", logger, client)
    try:
        result = _parse_json_safe(raw)
    except Exception:
        result = {"raw": raw, "_parse_error": True}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 4 — analyze_what_changed  [SUB-LLM]
# ---------------------------------------------------------------------------

def analyze_what_changed(transcript: str, company_name: str,
                          logger: LLMCallLogger, client) -> str:
    prompt = WHAT_CHANGED_PROMPT.format(
        company_name=company_name,
        transcript=transcript[:8000],
    )
    raw = _call_sub_llm(prompt, "tool:what_changed", logger, client)
    try:
        result = _parse_json_safe(raw)
    except Exception:
        result = {"raw": raw, "_parse_error": True}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 5 — analyze_thesis_status  [SUB-LLM]
# ---------------------------------------------------------------------------

def analyze_thesis_status(transcript: str, derived_thesis: str,
                           logger: LLMCallLogger, client) -> str:
    prompt = THESIS_STATUS_PROMPT.format(
        derived_thesis=derived_thesis,
        transcript=transcript[:8000],
    )
    raw = _call_sub_llm(prompt, "tool:thesis_status", logger, client)
    try:
        result = _parse_json_safe(raw)
    except Exception:
        result = {"raw": raw, "_parse_error": True}
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 5 — identify_red_flags  [HYBRID]
# ---------------------------------------------------------------------------

DEFLECTION_KEYWORDS = [
    "discuss offline", "not in a position", "circle back",
    "separately", "detailed note", "we'll get back",
    "i'd rather not", "cannot comment", "won't be able to share",
]

AUDITOR_KEYWORDS = [
    "qualification", "qualified opinion", "restatement",
    "auditor resigned", "going concern", "material weakness",
]

GUIDANCE_CUT_KEYWORDS = [
    "revised guidance", "lowered guidance", "cut guidance",
    "guidance revision", "reduce our outlook", "lower our forecast",
]


def identify_red_flags(transcript: str, kpis_json: str,
                        logger: LLMCallLogger, client) -> str:
    t_lower = transcript.lower()

    # --- Deterministic checks ---
    guidance_cut = any(kw in t_lower for kw in GUIDANCE_CUT_KEYWORDS)

    auditor_concern = any(kw in t_lower for kw in AUDITOR_KEYWORDS)

    # Margin compression: look for explicit compression statements
    margin_compression_match = re.search(
        r'margin[s]?\s+(?:compressed?|contracted?|declined?|fell?|lower)[^\d]*?(\d+\.?\d*)\s*(?:bps|basis\s+points|%)',
        t_lower
    )
    margin_compression = bool(margin_compression_match)
    margin_detail = (
        f"Margin {margin_compression_match.group(0).strip()}"
        if margin_compression_match else None
    )

    # --- Sub-LLM checks (nuanced language) ---
    deflection_hits = sum(1 for kw in DEFLECTION_KEYWORDS if kw in t_lower)
    llm_flags = {"management_deflection": {"detected": False, "detail": None},
                 "customer_concentration": {"detected": False, "detail": None}}

    # Only call LLM for deflection if keyword pre-pass finds suspicious signals
    if deflection_hits >= 1:
        prompt = RED_FLAGS_PROMPT.format(transcript=transcript[:8000])
        try:
            raw = _call_sub_llm(prompt, "tool:red_flags", logger, client)
            llm_flags = _parse_json_safe(raw)
        except Exception as e:
            llm_flags["_llm_error"] = str(e)
    else:
        # Still run LLM for customer concentration even if no deflection keywords
        prompt = RED_FLAGS_PROMPT.format(transcript=transcript[:8000])
        try:
            raw = _call_sub_llm(prompt, "tool:red_flags", logger, client)
            parsed = _parse_json_safe(raw)
            llm_flags["customer_concentration"] = parsed.get(
                "customer_concentration", {"detected": False, "detail": None}
            )
        except Exception as e:
            llm_flags["_llm_error"] = str(e)

    # --- Assemble and score ---
    flags = {
        "guidance_cut":          {"detected": guidance_cut,      "detail": "Guidance revision language found" if guidance_cut else None},
        "margin_compression":    {"detected": margin_compression, "detail": margin_detail},
        "auditor_concern":       {"detected": auditor_concern,    "detail": "Auditor concern keywords found" if auditor_concern else None},
        "management_deflection": llm_flags.get("management_deflection", {"detected": False, "detail": None}),
        "customer_concentration": llm_flags.get("customer_concentration", {"detected": False, "detail": None}),
    }

    flag_count = sum(1 for v in flags.values() if isinstance(v, dict) and v.get("detected"))
    severity = "NONE" if flag_count == 0 else "LOW" if flag_count == 1 else "MEDIUM" if flag_count <= 3 else "HIGH"

    flags["severity"] = severity
    flags["flag_count"] = flag_count
    return json.dumps(flags)


# ---------------------------------------------------------------------------
# Tool 6 — make_investment_decision  [DETERMINISTIC]
# ---------------------------------------------------------------------------

RATING_SCORE = {"STRENGTHENING": 2, "INTACT": 1, "WEAKENING": -1}


def make_investment_decision(thesis_json: str, red_flags_json: str, kpis_json: str) -> str:
    try:
        thesis    = json.loads(thesis_json)
        red_flags = json.loads(red_flags_json)
        kpis      = json.loads(kpis_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Could not parse inputs: {e}"})

    # Thesis score
    pillars = ["engine", "moat", "management"]
    thesis_score = 0
    pillar_breakdown = {}
    for p in pillars:
        rating = thesis.get(p, {}).get("rating", "INTACT").upper()
        score = RATING_SCORE.get(rating, 1)
        thesis_score += score
        pillar_breakdown[p] = f"{rating} ({'+' if score > 0 else ''}{score})"

    # Red flag penalty
    flag_count   = red_flags.get("flag_count", 0)
    flag_penalty = flag_count * -2

    # KPI momentum from YoY revenue growth
    try:
        growth_str = kpis.get("revenue", {}).get("yoy_growth") or "0"
        growth_pct = float(re.sub(r'[^0-9.\-]', '', str(growth_str)))
    except (ValueError, TypeError):
        growth_pct = 0.0

    kpi_momentum = 2 if growth_pct > 15 else 1 if growth_pct > 0 else -2

    total_score = thesis_score + flag_penalty + kpi_momentum

    decision = "ACCUMULATE" if total_score >= 5 else "HOLD" if total_score >= 1 else "TRIM"

    # Build rationale programmatically
    thesis_parts = ", ".join(f"{p.title()} {pillar_breakdown[p]}" for p in pillars)
    rationale = (
        f"Thesis: {thesis_parts}. "
        f"Red flags: {flag_count} detected (penalty {flag_penalty}). "
        f"Revenue growth {growth_pct:.1f}% → KPI momentum {'+' if kpi_momentum > 0 else ''}{kpi_momentum}. "
        f"Total score {total_score} → {decision}."
    )

    result = {
        "decision": decision,
        "score": total_score,
        "score_breakdown": {
            "thesis_score":  thesis_score,
            "flag_penalty":  flag_penalty,
            "kpi_momentum":  kpi_momentum,
        },
        "pillar_detail": pillar_breakdown,
        "rationale": rationale,
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 7 — generate_watchlist  [SUB-LLM]
# ---------------------------------------------------------------------------

def generate_watchlist(transcript: str, thesis_json: str, red_flags_json: str,
                        logger: LLMCallLogger, client) -> str:
    prompt = WATCHLIST_PROMPT.format(
        thesis_json=thesis_json,
        red_flags_json=red_flags_json,
        transcript=transcript[:6000],
    )
    raw = _call_sub_llm(prompt, "tool:watchlist", logger, client)
    try:
        result = _parse_json_safe(raw)
    except Exception:
        result = {"raw": raw, "_parse_error": True}
    return json.dumps(result)
