#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
match_essdive_como_pydantic_ai.py

Usage:
    CBORG_API_KEY=<your-key> python match_essdive_como_pydantic_ai.py \
        essdive_vars_complete_v3.tsv como_vars.tsv matched_vars_v3.tsv
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
from pydantic import BaseModel, Field, ValidationError

# ----------------------------------------------------------------------
# 2️⃣ System prompt
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
# 3️⃣ Initialise the CBORG model (OpenAI-compatible client)
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
# 4️⃣ Pydantic schemas
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
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if df.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns.")
    df = df.iloc[:, :3]
    df.columns = ["id", "name", "definition"]
    return df

def build_como_entries(df: pd.DataFrame) -> List[ComoEntry]:
    return [
        ComoEntry(
            id=row["id"],
            name=row["name"],
            definition=row["definition"] or "No definition provided"
        )
        for _, row in df.iterrows()
    ]

def make_match_request(ess_row: pd.Series, como_entries: List[ComoEntry]) -> MatchRequest:
    return MatchRequest(
        ess_id=ess_row["id"],
        ess_name=ess_row["name"],
        ess_definition=ess_row["definition"] or "No definition provided",
        como_catalogue=como_entries,
    )

# ----------------------------------------------------------------------
# 6️⃣ Core matching coroutine
# ----------------------------------------------------------------------
async def match_all(
    ess_path: Path,
    como_path: Path,
    out_path: Path,
    pause_seconds: float = 0.4,
    limit: int = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")

    ess_df = load_tsv(ess_path)
    como_df = load_tsv(como_path)
    como_entries = build_como_entries(como_df)

    if limit:
        ess_df = ess_df.head(limit)

    rows_out = []

    for i, ess_row in ess_df.iterrows():
        logging.info("Matching %s (%s) – %d / %d",
                     ess_row["name"], ess_row["id"], i + 1, len(ess_df))

        request_obj = make_match_request(ess_row, como_entries)
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

                try:
                    assistant_content = response.choices[0].message.content.strip()
                except (IndexError, AttributeError) as e:
                    raise RuntimeError(f"Invalid LLM response format: {response}") from e

                if assistant_content.startswith("```"):
                    assistant_content = assistant_content.strip("` ").split("\n", 1)[-1].strip("` ")

                raw_json = json.loads(assistant_content)
                result = MatchResult.model_validate(raw_json)
                break  # success

            except (json.JSONDecodeError, ValidationError) as exc:
                if attempts >= 2:
                    raise RuntimeError(f"Invalid JSON from LLM for {ess_row['id']}: {exc}\nRaw:\n{assistant_content}")
                logging.warning("JSON decode / validation failed. Retrying…\nResponse:\n%s", assistant_content)
                await asyncio.sleep(1)

            except Exception as exc:
                if attempts >= 3:
                    raise
                wait = 2 ** attempts
                logging.warning("API call failed (%s). Waiting %.1f s", exc, wait)
                await asyncio.sleep(wait)

        rows_out.append((
            ess_row["name"],
            result.como_name,
            result.match,
            ess_row["id"],
            result.como_id,
        ))

        await asyncio.sleep(pause_seconds)

    out_df = pd.DataFrame(rows_out, columns=[
        "ESS‑DIVE Variable", "COMO Variable", "Match?", "ESS‑DIVE ID", "COMO ID"
    ])
    logging.info("Writing %d rows to %s", len(out_df), out_path)
    out_df.to_csv(out_path, sep="\t", index=False, header=True)
    logging.info("✅ Finished – file saved as %s", out_path)

# ----------------------------------------------------------------------
# 8️⃣ Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Match ESS‑DIVE variables to COMO variables using CBORG LLM."
    )
    parser.add_argument("essdive_tsv", type=Path)
    parser.add_argument("como_tsv", type=Path)
    parser.add_argument("output_tsv", type=Path)
    parser.add_argument("--pause", type=float, default=0.4, help="Pause (sec) between LLM calls.")
    parser.add_argument("--limit", type=int, help="Limit number of rows processed (for testing).")

    args = parser.parse_args()

    asyncio.run(
        match_all(
            ess_path=args.essdive_tsv,
            como_path=args.como_tsv,
            out_path=args.output_tsv,
            pause_seconds=args.pause,
            limit=args.limit,
        )
    )
