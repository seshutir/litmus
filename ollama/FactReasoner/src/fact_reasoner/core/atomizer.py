# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Decompose the input string into atomic units. Use the same Mellea session
# (context) to revise or decontextualize the atomc units if needed.

import json
import mellea.stdlib.functional as mfuncs

from typing import Dict, List
from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import MelleaLogger

# Local imports
from fact_reasoner.utils import (
    validate_json_code_block,
    strip_code_fences,
    run_throttled,
    LOOP_BUDGET,
)

INSTRUCTION_ATOMIZER = """
Instructions:
Your task is to produce a given paragraph into a set of two atomic units without adding any new information.

Rules:
- An atomic unit is the smallest sentence containing a singular piece of information extracted from the provided paragraph.
- Atomic units may contradict one another.
- The first atomic unit must check the risk for the domain.
- The other atomic units must check the risk for the ai tasks.
- The first atomic unit must answer the question "What is the domain of this paragraph"
- The other atomic units must answer the question "What are the AI tasks of this paragraph"
- Where possible, avoid paraphrasing and instead try to only use language used in the paragraph without introducing new words. 
- The output must be a JSON dictionary with the following format and markdown code fences such that each atomic unit has a unique ID:

```json
{
    "id1": "<first atomic unit>",
    "id2": "<second atomic unit>",
    "id3": ....
}
```

Use the provided examples to learn your task.

Example 1:
INPUT: Hallucination is a risk in Gen AI automates legal billing processes by analyzing time spent on various tasks, generating invoices, and identifying potential billing errors, improving efficiency and reducing administrative burdens for legal teams.
OUTPUT:
```json
{
    "id1": "Hallucination is a risk in legal billing",
    "id2": "Hallucination is a risk in analyzing time‑tracking data to calculate billable hours per task",
    "id3": "Hallucination is a risk in automatically generating and formatting invoices,
    "id4": "Hallucination is a risk in detecting and flagging potential billing errors or inconsistencies",
    "id5": "Hallucination is a risk in identifying patterns and opportunities for process optimization"
}
```

Example 2:
INPUT: Security is a risk in Gen AI analyzes customer purchase history, browsing behavior, and demographic data to recommend relevant products, leading to increased customer satisfaction and sales.
OUTPUT:
```json
{
    "id1": "Security is a risk in e-commerce Recommendation",
    "id2": "Security is a risk in analyzing customer purchase history",
    "id3": "Security is a risk in analyzing browsing behavior",
    "id4": "Security is a risk in analyzing demographic data",
    "id5": "Security is a risk in generating product recommendations"
}
```

Your task:
INPUT: {{response}}
OUTPUT:
"""


class Atomizer(object):
    """
    The Atomizer class implements the atomic decomposition of the response.
    For our purpose, an atomic unit or atom is either a fact or a claim.

    Design note (Mellea backend vs. session):
    This class holds a raw Mellea ``Backend`` and issues requests via the
    ``mellea.stdlib.functional`` free functions with a fresh ``SimpleContext()``
    per call, rather than using a ``MelleaSession``. Atomization is a batch of
    independent, stateless requests fanned out concurrently (see ``run_batch``),
    and this is the pattern the Mellea authors recommend for such workloads:
      - A ``MelleaSession`` threads a single mutable context through calls. Under
        concurrency that shared context races: harmless with ``SimpleContext``
        (its view is always empty) but pointless, and incorrect with
        ``ChatContext`` (Mellea even logs a stale-context warning for async +
        non-SimpleContext). We have no multi-turn conversation to thread.
      - A ``Backend`` holds only connection/config and no per-request mutable
        state, so it is safe to share across all concurrent calls, while each
        call's immutable ``SimpleContext()`` guarantees per-request isolation.
      - Mellea's own guidance ("for parallel generation, use SimpleContext") and
        its session docstring both steer stateless/high-concurrency use away
        from sessions and toward Backend + functional.
    If a genuinely sequential, multi-turn sub-flow is ever needed, introduce a
    session locally there (with ``ChatContext``, awaiting between calls); keep
    the batch path on Backend + per-call ``SimpleContext``.
    """

    def __init__(
        self,
        backend: Backend,
    ):
        """
        Initialize the Atomizer.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
        """

        # Safety checks
        if backend is None:
            raise ValueError(
                "Mellea backend is None. Please provide a valid Mellea backend."
            )

        # Initialize the extractor
        self.backend = backend

        # Print info
        print(f"[Atomizer] Using Mellea backend: {self.backend.model_id}")

        # Disable Mellea logging
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    def run(self, response: str) -> Dict[str, str]:
        """
        Extract atomic units from a single response.

        Args:
            response: str
                The response from which to extract atomic units.
        Returns:
            Dict[str, str]: A dictionary containing the atomic units, each with
            a unique identifier.
        """
        # Perform the instruction with validation. A backend/network error is
        # raised out of mfuncs.instruct (validation failures instead come back
        # as a result with success=False), so guard the whole generation.
        try:
            output = mfuncs.instruct(
                INSTRUCTION_ATOMIZER,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid JSON dictionary with markdown code fences",
                        validation_fn=simple_validate(
                            lambda s: validate_json_code_block(s)
                        ),
                    )
                ],
                user_variables={"response": response},
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True,
            )
        except Exception as e:
            print(f"[Atomizer] Generation failed: {e}")
            return {}  # empty dict on failure

        if not output.success:
            return {}  # empty dict on validation failure

        # The output is a validated JSON string; parse it defensively.
        try:
            cleaned = strip_code_fences(str(output))
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Atomizer] Failed to parse output: {e}")
            return {}

    async def run_batch(self, responses: List[str]) -> List[Dict[str, str]]:
        """
        Extract atomic units from a list of responses.

        Args:
            responses: List[str]
                The list of response from which to extract atomic units.
        Returns:
            dict: A dictionary containing the number of atomic units, the units themselves,
            all atomic units as dictionaries, and all facts as dictionaries.
        """

        # Build a fresh coroutine per response. run_throttled applies bounded
        # concurrency plus a per-minute rate limit, and captures per-item
        # exceptions so a single backend failure does not drop the rest.
        def factory(response: str):
            return mfuncs.ainstruct(
                INSTRUCTION_ATOMIZER,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid JSON dictionary with markdown code fences",
                        validation_fn=simple_validate(
                            lambda s: validate_json_code_block(s)
                        ),
                    )
                ],
                user_variables={"response": response},
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True,
            )

        print(f"[Atomizer] Running throttled batch of {len(responses)} requests ...")
        outputs = await run_throttled(factory, responses)

        # Results are positionally aligned with responses; map every failure
        # (raised exception, unsuccessful sampling, or unparsable output) to {}.
        results: List[Dict[str, str]] = []
        for output in outputs:
            if isinstance(output, Exception):
                print(f"[Atomizer] Batch item failed: {output}")
                results.append({})
                continue
            if not getattr(output, "success", False):
                results.append({})
                continue
            try:
                cleaned = strip_code_fences(str(output))
                results.append(json.loads(cleaned))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Atomizer] Failed to parse batch item: {e}")
                results.append({})

        return results

    def __str__(self) -> str:
        return "This is the atomizer"
