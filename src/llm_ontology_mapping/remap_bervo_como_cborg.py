#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
remap_bervo_como_cborg.py

Re‑map the COMO columns of a previously‑merged BERVO file by querying the CBORG
LLM.  Rows whose Match? flag is “Y” are refreshed; rows whose flag is “N” (or
any other value) are cleared.

Usage
-----

    CBORG_API_KEY=<your‑key> python remap_bervo_como_cborg.py \
        merged_file.tsv updated_como_vars.tsv remapped_output.tsv
"""

# ----------------------------------------------------------------------
# 1️⃣ Imports
# ----------------------------------------------------------------------
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Literal

import pandas as pd
from openai import AsyncOpenAI           # ✅ Correct OpenAI client class
from pydantic import BaseModel, ValidationError

# ----------------------------------------------------------------------
# 2️⃣ System prompt (unchanged)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an ontology‑matching assistant. "
    "Given one ESS‑DIVE variable (its id, name and definition) and a list of "
    "COMO variables (each with id, name and definition), select the single "
    "COMO variable that best corresponds to the ESS‑DIVE variable. "
    "Return a JSON object with the keys `como_id`, `como_name` and `match` "
    "(where `match` must be either \"Y\" or \"N\"). "
    "If none of the COMO entries are appropriate, still return a JSON object "
    "but set `match` to \"N\" and pick the most related term."
)

# ----------------------------------------------------------------------
# 3️⃣ Initialise the CBORG model (OpenAI‑compatible client)
# ----------------------------------------------------------------------
api_key = os.getenv("CBORG_API_KEY")
if not api_key:
    raise EnvironmentError("Please set the CBORG_API_KEY environment variable.")

MODEL_NAME = "lbl/cborg-deepthought:latest"
openai_client = AsyncOpenAI(
    base_url="https://api.cborg.lbl.gov/v1",
    api_key=api_key,
)

# ----------------------------------------------------------------------
# 4️⃣ Pydantic schemas (unchanged)
# ----------------------------------------------------------------------
class MatchResult(BaseModel):
    como_id: str
    como_name: str
    match: Literal["Y", "N"]

class ComoEntry(BaseModel):
    id: str
    name: str
    definition: str

class MatchRequest(BaseModel):
    ess_id: str
    ess_name: str
    ess_definition: str
    como_catalogue: List[ComoEntry]

# ----------------------------------------------------------------------
# 5️⃣ Helper functions
# ----------------------------------------------------------------------
def load_tsv(path: Path) -> pd.DataFrame:
    """Load a three‑column TSV (id, name, definition) as a DataFrame."""
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if df.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns.")
    df = df.iloc[:, :3]
    df.columns = ["id", "name", "definition"]
    return df


def build_como_entries(df: pd.DataFrame) -> List[ComoEntry]:
    """Convert the COMO master DataFrame into a list of Pydantic objects."""
    return [
        ComoEntry(
            id=row["id"],
            name=row["name"],
            definition=row["definition"] or "No definition provided",
        )
        for _, row in df.iterrows()
    ]


def make_match_request(bervo_row: pd.Series, como_entries: List[ComoEntry]) -> MatchRequest:
    """Create the request payload for a single BERVO row."""
    return MatchRequest(
        ess_id=bervo_row["BERVO ID"],
        ess_name=bervo_row["BERVO Variable"],
        ess_definition=bervo_row["BERVO Definition"] or "No definition provided",
        como_catalogue=como_entries,
    )


async def query_cborg(request_obj: MatchRequest) -> MatchResult:
    """
    Send a single matching request to the CBORG LLM and return a validated
    ``MatchResult``.  Retries are performed on JSON/validation errors and on
    transient API failures (same logic as the original script).
    """
    user_message = json.dumps(request_obj.model_dump(), indent=2)

    attempts = 0
    while True:
        attempts += 1
        try:
            response = await openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                timeout=60.0,
            )
            # ------------------------------------------------------------------
            # Extract & clean LLM output
            # ------------------------------------------------------------------
            try:
                assistant_content = response.choices[0].message.content.strip()
            except (IndexError, AttributeError) as e:
                raise RuntimeError(f"Invalid LLM response format: {response}") from e

            # Strip possible markdown fences
            if assistant_content.startswith("```"):
                assistant_content = assistant_content.strip("` ").split("\n", 1)[-1].strip("` ")

            raw_json = json.loads(assistant_content)
            result = MatchResult.model_validate(raw_json)
            return result

        except (json.JSONDecodeError, ValidationError) as exc:
            if attempts >= 2:
                raise RuntimeError(
                    f"Invalid JSON / validation from LLM for BERVO ID {request_obj.ess_id}: {exc}\nRaw:\n{assistant_content}"
                )
            logging.warning(
                "JSON decode / validation failed (attempt %d). Retrying…\nResponse:\n%s",
                attempts,
                assistant_content,
            )
            await asyncio.sleep(1)

        except Exception as exc:   # network / API errors
            if attempts >= 3:
                raise
            wait = 2 ** attempts
            logging.warning("API call failed (%s). Waiting %.1f s", exc, wait)
            await asyncio.sleep(wait)


# ----------------------------------------------------------------------
# 6️⃣ Core coroutine – re‑mapping the merged file
# ----------------------------------------------------------------------
async def remap_all(
    mapped_path: Path,
    como_path: Path,
    out_path: Path,
    pause_seconds: float = 0.4,
) -> None:
    """
    Read *mapped_path* (the merged TSV with 7 columns), update the COMO columns
    via the CBORG API where appropriate, and write the result to *out_path*.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    # 1) The merged file – keep original column order
    merged_df = pd.read_csv(
        mapped_path, sep="\t", dtype=str, keep_default_na=False
    )
    # Ensure required columns exist (case‑sensitive as in the spec)
    required_cols = [
        "BERVO ID",
        "BERVO Variable",
        "Match?",
        "COMO Variable",
        "COMO ID",
        "BERVO Definition",
        "BERVO Units",
    ]
    missing = [c for c in required_cols if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Input file {mapped_path} is missing columns: {missing}")

    # 2) The updated COMO catalogue
    como_df = load_tsv(como_path)
    como_entries = build_como_entries(como_df)

    rows_out = []  # will store dicts preserving column order

    for i, row in merged_df.iterrows():
        match_val = row["Match?"].strip()
        match_upper = match_val.upper()

        # ------------------------------------------------------------------
        # Case A – original flag is Y → ask the LLM
        # ------------------------------------------------------------------
        if match_upper == "Y":
            request_obj = make_match_request(row, como_entries)
            try:
                result = await query_cborg(request_obj)
            except Exception as exc:
                logging.error(
                    "LLM call failed for BERVO ID %s – treating as no‑match: %s",
                    row["BERVO ID"],
                    exc,
                )
                result = MatchResult(como_id="", como_name="", match="N")

            if result.match == "Y":
                row["COMO Variable"] = result.como_name
                row["COMO ID"] = result.como_id
                row["Match?"] = "Y"
            else:
                # LLM reported no suitable match
                row["COMO Variable"] = ""
                row["COMO ID"] = ""
                row["Match?"] = "N"

        # ------------------------------------------------------------------
        # Case B – flag is N / any other value → wipe COMO fields
        # ------------------------------------------------------------------
        else:
            row["COMO Variable"] = ""
            row["COMO ID"] = ""
            row["Match?"] = "N"

        # Append the (now‑clean) row as a plain dict preserving order
        rows_out.append({col: row[col] for col in required_cols})

        # Small pause to be nice to the API
        await asyncio.sleep(pause_seconds)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(rows_out, columns=required_cols)
    logging.info("Writing %d rows to %s", len(out_df), out_path)
    out_df.to_csv(out_path, sep="\t", index=False, header=True)
    logging.info("✅ Finished – file saved as %s", out_path)


# ----------------------------------------------------------------------
# 7️⃣ Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Re‑map the COMO columns of a merged BERVO file using the CBORG LLM. "
            "Rows with Match? = Y are refreshed; rows with Match? = N are cleared."
        )
    )
    parser.add_argument("merged_tsv", type=Path, help="Path to the merged file (7 columns).")
    parser.add_argument("como_tsv", type=Path, help="Path to the updated como_vars.tsv.")
    parser.add_argument(
        "output_tsv",
        type=Path,
        help="Path for the remapped output TSV (default: remapped.tsv).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.4,
        help="Pause in seconds between successive LLM calls (default: 0.4).",
    )

    args = parser.parse_args()

    # Basic sanity checks
    for p in (args.merged_tsv, args.como_tsv):
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")

    asyncio.run(
        remap_all(
            mapped_path=args.merged_tsv,
            como_path=args.como_tsv,
            out_path=args.output_tsv,
            pause_seconds=args.pause,
        )
    )
