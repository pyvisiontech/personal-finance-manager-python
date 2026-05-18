"""
PDF bank statement extractor using get_text("words") + LLM.

Step 1: _get_column_order   — one LLM call on first 200 words to detect column names
Step 2: _extract_page_transactions — one LLM call per page to extract transaction rows
"""

import json
import re
import logging

logger = logging.getLogger("app")


def _format_words(words: list) -> str:
    """Sort words by (y0, x0) and format as coordinate string for LLM."""
    sorted_words = sorted(words, key=lambda w: (round(w[1]), w[0]))
    return "\n".join(
        f'({w[0]:.1f}, {w[1]:.1f}, {w[2]:.1f}, {w[3]:.1f}, "{w[4]}")'
        for w in sorted_words
    )


def _parse_json_response(raw: str) -> list | None:
    """Strip markdown fences and parse JSON array from LLM response."""
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"```$", "", raw).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        # Try extracting embedded JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
    return None


def _get_column_order(doc, client, model) -> list | None:
    """
    Detect column names from the first 200 words of the document.
    Returns ordered list of column name strings, or None on failure.
    """
    words = doc[0].get_text("words")
    if not words:
        logger.warning("_get_column_order: no words on page 0")
        return None

    word_data = _format_words(words[:200])

    prompt = f"""You are processing a bank statement PDF. Below are the first words extracted from the PDF
with their bounding box coordinates.

Data format: (x0, y0, x1, y1, "text")
- x0, y0 = top-left corner of the word; x1, y1 = bottom-right corner
- x increases left→right; y increases top→bottom
- Words with similar y0 (within ~5px) are on the same visual row
- A word's column position is indicated by its x0 value

The PDF typically starts with bank name, address, account info, and date range — these
appear before the actual table header. Ignore them.

The table header row contains column labels like Date, Narration, Debit, Credit, Balance.
Some column names span multiple lines (incrementing y0) — combine them into one string.
Multi-line parts of the same column header may have slightly different x0 values due to
center-alignment — use horizontal center to group them into the same column.

Return ONLY a JSON array of column names in left-to-right order.
No explanation, no markdown — just the JSON array.

Actual data:
{word_data}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
    )

    raw = response.choices[0].message.content
    logger.info(f"_get_column_order raw response: {raw}")

    column_order = _parse_json_response(raw)
    if column_order and len(column_order) >= 2:
        logger.info(f"Detected column order: {column_order}")
        return column_order

    logger.warning(f"_get_column_order: failed to parse valid column list from response")
    return None


def _extract_page_transactions(page, column_order: list, client, model) -> list:
    """
    Extract transaction rows from a single page using coordinate data + LLM.
    Returns list of {date, narration, debit_amount, credit_amount} dicts.
    """
    words = page.get_text("words")
    if not words:
        logger.info("_extract_page_transactions: no words on page")
        return []

    word_data = _format_words(words)

    prompt = f"""You are processing one page of a bank statement PDF. Below are all words extracted
from this page with their bounding box coordinates.

Data format: (x0, y0, x1, y1, "text")
- x0, y0 = top-left corner of the word; x1, y1 = bottom-right corner
- x increases left→right; y increases top→bottom
- Words with similar y0 belong to the same visual row — infer row boundaries
  from natural gaps in y0 values
- A word's column is determined by its x position

Column order (left to right): {column_order}
The header row may be absent on this page — use the column order above as reference
for assigning words to the correct column.

Some narrations span multiple lines — merge them into a single narration string.

Extract every transaction and return ONLY a JSON array:
[{{"date": "...", "narration": "...", "debit_amount": "500.00 or null", "credit_amount": "250.00 or null"}}]

Rules:
- Skip header rows, opening/closing balance rows, and totals rows
- debit_amount and credit_amount are mutually exclusive per row — one must always be null
- Preserve the original date format as-is
- Every row is a unique transaction — do NOT merge or skip rows even if amounts or narrations look similar to adjacent rows
- Payment reversals and refunds are valid transactions — include them

Actual data:
{word_data}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0,
    )

    raw = response.choices[0].message.content
    logger.info(f"_extract_page_transactions response (first 300 chars): {raw[:300]}")

    rows = _parse_json_response(raw)
    if rows is not None:
        logger.info(f"Extracted {len(rows)} transactions from page")
        return rows

    logger.warning("_extract_page_transactions: failed to parse JSON response")
    return []
