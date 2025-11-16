import json
import os
from typing import Dict, Any, Tuple

from openai import OpenAI, AuthenticationError, OpenAIError


def _format_analysis_text(analysis: Dict[str, Any]) -> str:
    """
    Turn the analysis dict into a plain-text summary (fallback if no LLM).
    """
    lines = []
    meta = analysis.get("meta", {})
    cols = analysis.get("columns", {})

    lines.append(
        f"Rows: {meta.get('num_rows', '?')}, "
        f"Columns: {meta.get('num_columns', '?')}"
    )
    lines.append(
        f"Numeric columns: {', '.join(meta.get('numeric_columns', [])) or 'None'}"
    )
    lines.append("")

    for col_name, stats in cols.items():
        lines.append(f"Column: {col_name}")
        lines.append(
            f"  mean={stats['mean']:.3f}, "
            f"min={stats['min']:.3f}, "
            f"max={stats['max']:.3f}, "
            f"std={stats['std']:.3f}"
        )
        lines.append(
            f"  first={stats['first_val']:.3f}, "
            f"last={stats['last_val']:.3f}, "
            f"trend={stats['trend']:.3f}"
        )
        monotonic = "none"
        if stats["is_monotonic_increasing"]:
            monotonic = "increasing"
        elif stats["is_monotonic_decreasing"]:
            monotonic = "decreasing"
        lines.append(
            f"  monotonic={monotonic}, "
            f"spikes={stats['num_spikes']}, "
            f"count={stats['count']}"
        )
        lines.append("")

    return "\n".join(lines)


def _try_init_client() -> Tuple[OpenAI | None, str | None]:
    """
    Try to initialize an OpenAI client. Returns (client, error_message).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "OPENAI_API_KEY not set."

    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except AuthenticationError as e:
        return None, f"Authentication error: {e}"
    except Exception as e:  # noqa: BLE001
        return None, f"Unexpected error initializing OpenAI client: {e}"


def summarize(analysis: Dict[str, Any]) -> str:
    """
    Summarize the analysis using OpenAI if available, otherwise fallback.

    Returns a plain-text summary.
    """
    # Fallback summary text in any case
    fallback_summary = _format_analysis_text(analysis)

    client, error = _try_init_client()
    if client is None:
        # No client → just return fallback
        return (
            "LLM summary unavailable "
            f"({error}). Here is a basic analysis summary:\n\n{fallback_summary}"
        )

    prompt = f"""
You are an engineering simulation analyst.

Given this JSON analysis of one or more simulation time-series, write a clear,
concise technical summary for an engineer.

JSON analysis:
{json.dumps(analysis, indent=2)}

Your summary should:
- Describe overall behavior of key variables.
- Mention trends (rising, falling, stable).
- Mention any anomalies or spikes.
- Suggest 1–3 possible next steps or parameter checks.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content or fallback_summary
    except OpenAIError as e:
        return (
            "LLM summary failed "
            f"({e}). Here is a basic analysis summary instead:\n\n{fallback_summary}"
        )
    except Exception as e:  # noqa: BLE001
        return (
            "Unexpected error during LLM summary "
            f"({e}). Here is a basic analysis summary:\n\n{fallback_summary}"
        )
