"""
DeepEval-based automated annotator for the "Risk Validation Task".

This mirrors the three Label Studio questions from the annotation UI:

  Q1  Intent <-> Atom consistency   -> consistent / inconsistent / unclear
  Q2  Atom  <-> Risk text relevance -> relevant   / irrelevant   / unclear
  Q3  Entailment type verification  -> correct    / incorrect    / unsure

Each question is implemented as its own GEval metric, but instead of a
score-range Rubric, each metric uses explicit `evaluation_steps` plus
few-shot anchor examples (low-score / high-score) baked into the steps --
the same pattern used in the "Paraphrase quality" metric. This tends to
give the judge a much more concrete, calibrated sense of what a 0.1 vs a
1.0 looks like than a bare rubric description does.

IMPORTANT: The few-shot examples below are placeholders. Swap them out for
real low/high scoring examples pulled from your own annotated data -- that
is what actually calibrates the judge, not the fact that examples exist.

Usage
-----
    export OPENAI_API_KEY=...        # or set model= to a local/other judge
    python deepeval_risk_annotator.py --dataset data.json --outdir results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

from deepeval.metrics.g_eval import Rubric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

os.environ["OPENAI_API_KEY"] = "..."

# ---------------------------------------------------------------------------
# 1. Metric definitions -- one GEval metric per annotation question
# ---------------------------------------------------------------------------

def build_q1_metric(model: Optional[str] = None) -> GEval:
    """Q1: Is the risk scenario (atom) logically consistent with the intent?"""
    return GEval(
        name="Intent-Atom Consistency",
        criteria=(
            "Determine whether the RISK SCENARIO in the Actual Output could plausibly "
            "relate to, arise from, or be associated with the INTENT in the Input. "
            "DEFAULT TO CONSISTENT: annotators treat almost every risk scenario that "
            "shares the same system, capability, user population, or domain as the "
            "intent as consistent, even when the connection is indirect, tangential, "
            "second-order, or an edge case. Only mark something inconsistent when the "
            "risk scenario is about a genuinely different system, capability, or "
            "domain with essentially no plausible link to the intent."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Compare the INTENT in the Input with the RISK SCENARIO in the Actual Output.",
            "Ask a low, permissive bar: is there ANY plausible way this risk could "
            "arise from, or be associated with, the system/capability/use-case "
            "described in the intent? This includes indirect effects, second-order "
            "consequences, misuse by a third party, edge cases, and low-probability "
            "but coherent scenarios -- all of these still count as consistent.",
            "Default to CONSISTENT (high score, e.g. 0.8-1.0) whenever such a "
            "connection exists. Do NOT require the risk scenario to be the most "
            "obvious, most likely, or primary risk of the intent -- being one of many "
            "possible risks is enough.",
            "Do NOT penalize a risk scenario merely for being extreme, unlikely, "
            "narrow, or indirectly connected. Penalize (score low) ONLY when the risk "
            "scenario describes a fundamentally different system, capability, domain, "
            "or actor than the intent, such that connecting them would require an "
            "unsupported, arbitrary leap.",
            "When genuinely torn between a low score and a high score, prefer the "
            "high score -- treat 'plausibly related' as the default and reserve low "
            "scores for clear mismatches.",
            "Use the following FEW-SHOT EXAMPLES to anchor your scoring alignment "
            "(REPLACE these with real examples from your labeled dataset):",

            # High Score Example -- direct connection
            "[EXAMPLE 1 - HIGH SCORE, direct connection]"
            "Input: 'Intent: A system for scheduling employee shifts based on availability and demand forecasts.'"
            "Actual Output: 'Risk scenario: The scheduling model systematically assigns fewer hours to employees who requested religious accommodations, effectively penalizing them.'"
            "Score: 0.95"
            "Reasoning: This risk arises directly from the scheduling capability "
            "described in the intent.",

            # High Score Example -- indirect/tangential but still consistent
            "[EXAMPLE 2 - HIGH SCORE, indirect but still consistent]"
            "Input: 'Intent: A system for scheduling employee shifts based on availability and demand forecasts.'"
            "Actual Output: 'Risk scenario: Managers use the scheduling tool's exported data to build informal, undocumented performance rankings that influence promotion decisions.'"
            "Score: 0.85"
            "Reasoning: This is a second-order, downstream misuse of the system's "
            "output rather than a risk of the core scheduling function itself, but it "
            "still plausibly arises from the same system and user population -- so it "
            "should be scored as consistent, not penalized for being indirect.",

            # Low Score Example -- genuinely unrelated domain
            "[EXAMPLE 3 - LOW SCORE, no plausible link]"
            "Input: 'Intent: A system for scheduling employee shifts based on availability and demand forecasts.'"
            "Actual Output: 'Risk scenario: The model leaks patients' private medical records to unauthorized third parties.'"
            "Score: 0.05"
            "Reasoning: The risk scenario concerns medical-record privacy, an entirely "
            "different domain, capability, and data type than shift scheduling. There "
            "is no plausible chain connecting the intent to this risk.",

            "Compare the target case against these examples to assign a final score "
            "from 0.0 to 1.0. Make sure not to confuse the Input (intent) with the "
            "Actual Output (risk scenario) when scoring.",
        ],
        # IMPORTANT: without an explicit rubric, GEval scores "strength of
        # alignment with the evaluation_steps text" rather than "which
        # category applies" -- so hedging words like "indirect" or "tenuous"
        # drag the score down even when the steps say to treat them as
        # consistent. The rubric below forces the score to reflect the
        # category, matching how lenient human annotators actually are here.
        rubric=[
            Rubric(
                score_range=(0, 2),
                expected_outcome=(
                    "Inconsistent - the risk scenario has NO plausible connection "
                    "to the intent: it concerns a genuinely different system, "
                    "capability, domain, or actor."
                ),
            ),
            Rubric(
                score_range=(3, 4),
                expected_outcome=(
                    "Unclear - a connection is conceivable but is highly "
                    "speculative or the case is genuinely ambiguous even after "
                    "considering indirect/second-order links."
                ),
            ),
            Rubric(
                score_range=(5, 10),
                expected_outcome=(
                    "Consistent - there is a plausible connection between the "
                    "risk scenario and the intent, INCLUDING indirect, "
                    "tangential, second-order, or low-probability-but-coherent "
                    "connections. This should be the default outcome whenever "
                    "any plausible link exists, even a weak one."
                ),
            ),
        ],
        model=model,
        threshold=0.5,
    )


def build_q2_metric(model: Optional[str] = None) -> GEval:
    """Q2: Is the supporting risk text relevant to the risk scenario (atom)?"""
    return GEval(
        name="Risk Text Relevance",
        criteria=(
            "Determine whether the SUPPORTING TEXT in the Actual Output provides "
            "relevant evidence or context for the RISK SCENARIO in the Input. "
            "Relevant text should discuss the same risk topic, mechanism, or domain "
            "as the risk scenario, even if it is only partial evidence."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Compare the RISK SCENARIO in the Input with the SUPPORTING TEXT in the "
            "Actual Output.",
            "Ask: does the supporting text discuss the same risk topic, mechanism, or "
            "domain as the risk scenario -- even partially -- such that it could serve "
            "as evidence or context for that scenario?",
            "Text that is on-topic but only weakly or partially supportive should "
            "still score reasonably well. Text that is about an unrelated risk, "
            "domain, or mechanism should score low.",
            "Use the following FEW-SHOT EXAMPLES to anchor your scoring alignment "
            "(REPLACE these with real examples from your labeled dataset):",

            # Low Score Example -- placeholder, replace with a real one
            "[EXAMPLE 1 - LOW SCORE]"
            "Input: 'Risk scenario: An automated resume-screening model systematically "
            "downranks candidates from certain universities associated with a minority group.'"
            "Actual Output: 'Supporting text: The company reported a 12% increase in "
            "quarterly cloud infrastructure costs due to higher model-serving traffic.'"
            "Score: 0.05"
            "Reasoning: The supporting text is about infrastructure cost, which has no "
            "bearing on the discrimination risk scenario described.",

            # High Score Example -- placeholder, replace with a real one
            "[EXAMPLE 2 - HIGH SCORE]"
            "Input: 'Risk scenario: An automated resume-screening model systematically "
            "downranks candidates from certain universities associated with a minority group.'"
            "Actual Output: 'Supporting text: An internal audit found that candidates "
            "from the flagged universities received scores 18% lower on average, even "
            "after controlling for stated qualifications.'"
            "Score: 0.95"
            "Reasoning: The supporting text directly documents the disparity described "
            "in the risk scenario and provides concrete evidence for it.",

            "Compare the target case against these examples to assign a final score "
            "from 0.0 to 1.0. Make sure not to confuse the Input (risk scenario) with "
            "the Actual Output (supporting text) when scoring.",
        ],
        # Same fix as Q1: anchor the score to the category via a rubric,
        # since without one GEval scores "strength of alignment with the
        # steps text" rather than "which label applies" -- causing hedging
        # words like "partial" or "indirect" to drag the score down even
        # when the steps say to treat those as relevant.
        rubric=[
            Rubric(
                score_range=(0, 2),
                expected_outcome=(
                    "Irrelevant - the supporting text has NO bearing on the risk "
                    "scenario: different topic, mechanism, or domain entirely."
                ),
            ),
            Rubric(
                score_range=(3, 4),
                expected_outcome=(
                    "Unclear - the text's relevance is genuinely ambiguous even "
                    "after considering partial/indirect support."
                ),
            ),
            Rubric(
                score_range=(5, 10),
                expected_outcome=(
                    "Relevant - the text discusses the same risk topic, "
                    "mechanism, or domain as the risk scenario, even if only "
                    "partial, indirect, or weak evidence. This should be the "
                    "default outcome whenever the text is on-topic at all."
                ),
            ),
        ],
        model=model,
        threshold=0.5,
    )


def build_q3_metric(model: Optional[str] = None) -> GEval:
    """Q3: Is the marked entailment/contradiction conclusion valid given the evidence?"""
    return GEval(
        name="Entailment Validity",
        criteria=(
            "You are given a RISK SCENARIO and SUPPORTING EVIDENCE as context, a "
            "SYNTHETIC SUMMARY as the Actual Output, and a MARKED RELATIONSHIP LABEL "
            "as the Expected Output (one of 'entailment', 'contradiction', or "
            "'neutral'). Determine whether the synthetic summary's conclusion about "
            "the risk scenario, given the supporting evidence, actually matches the "
            "marked relationship label."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.CONTEXT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Read the RISK SCENARIO (Input), the SUPPORTING EVIDENCE (Context), and "
            "the SYNTHETIC SUMMARY (Actual Output).",
            "Independently judge the true relationship: 'entailment' if the evidence "
            "supports/implies the risk scenario's conclusion as stated in the "
            "synthetic summary; 'contradiction' if the evidence undermines or "
            "contradicts it; 'neutral' if the evidence is unrelated to it.",
            "Compare your independent judgment to the MARKED RELATIONSHIP LABEL "
            "(Expected Output). A high score means they match; a low score means "
            "they don't.",
            "Use the following FEW-SHOT EXAMPLES to anchor your scoring alignment "
            "(REPLACE these with real examples from your labeled dataset):",

            # Low Score Example -- placeholder, replace with a real one
            "[EXAMPLE 1 - LOW SCORE]"
            "Input: 'Risk scenario: The recommendation model amplifies extremist "
            "content to maximize engagement.'"
            "Context: 'Internal metrics showed no measurable increase in extremist "
            "content exposure after the recommendation model was deployed.'"
            "Actual Output: 'The evidence confirms the model amplifies extremist "
            "content as described in the risk scenario.'"
            "Expected Output: 'entailment'"
            "Score: 0.05"
            "Reasoning: The context explicitly reports no increase in exposure, which "
            "contradicts the risk scenario -- the correct label should be "
            "'contradiction', not 'entailment', so the marked label is wrong.",

            # High Score Example -- placeholder, replace with a real one
            "[EXAMPLE 2 - HIGH SCORE]"
            "Input: 'Risk scenario: The recommendation model amplifies extremist "
            "content to maximize engagement.'"
            "Context: 'An internal audit found that flagged extremist content was "
            "recommended 3x more often than baseline after the engagement-optimization "
            "update shipped.'"
            "Actual Output: 'The evidence confirms the model amplifies extremist "
            "content as described in the risk scenario.'"
            "Expected Output: 'entailment'"
            "Score: 0.95"
            "Reasoning: The context directly supports the risk scenario's conclusion, "
            "so 'entailment' is the correct label and the marked label matches.",

            "Compare the target case against these examples to assign a final score "
            "from 0.0 to 1.0. Make sure not to confuse the Context (evidence) with the "
            "Actual Output (synthetic summary) or the Expected Output (marked label) "
            "when scoring.",
        ],
        # Same fix as Q1/Q2: anchor the score to the category via a rubric.
        # Unlike Q1/Q2 there's no reason to bias this one toward a default
        # label -- it's a straight match/mismatch check between your
        # independently-derived relationship and the marked label -- but it
        # still needs the rubric so the score reflects "does the label
        # match" rather than "how strong is the evidence in general".
        rubric=[
            Rubric(
                score_range=(0, 2),
                expected_outcome=(
                    "Incorrect - your independently-derived relationship "
                    "(entailment/contradiction/neutral) clearly does NOT match "
                    "the marked relationship label."
                ),
            ),
            Rubric(
                score_range=(3, 4),
                expected_outcome=(
                    "Unsure - it's genuinely difficult to determine the true "
                    "relationship, or the evidence is too thin/ambiguous to "
                    "confirm or refute the marked label with confidence."
                ),
            ),
            Rubric(
                score_range=(5, 10),
                expected_outcome=(
                    "Correct - your independently-derived relationship clearly "
                    "matches the marked relationship label."
                ),
            ),
        ],
        model=model,
        threshold=0.5,
    )


# ---------------------------------------------------------------------------
# 2. Score -> categorical label mapping (per question, matching the UI Choices)
#    GEval scores are normalized to 0-1, so buckets are defined on that scale.
# ---------------------------------------------------------------------------

Q1_LABELS = {"neg": "inconsistent", "mid": "unclear", "pos": "consistent"}
Q2_LABELS = {"neg": "irrelevant", "mid": "unclear", "pos": "relevant"}
Q3_LABELS = {"neg": "incorrect", "mid": "unsure", "pos": "correct"}


def score_to_label(score: float, labels: dict, neg_max: float = 0.25, mid_max: float = 0.45) -> str:
    """Map a 0-1 GEval score back to a category.

    Defaults match the standard rubric used by all three metrics: 0-2/10 ->
    neg, 3-4/10 -> mid, 5-10/10 -> pos, i.e. 0-0.2 / 0.3-0.4 / 0.5-1.0
    normalized. neg_max/mid_max are exposed in case a question's rubric
    ever diverges from that standard shape.
    """
    if score <= neg_max:
        return labels["neg"]
    if score <= mid_max:
        return labels["mid"]
    return labels["pos"]


# ---------------------------------------------------------------------------
# 3. Test-case builders -- map dataset fields onto LLMTestCase fields
# ---------------------------------------------------------------------------

def make_q1_case(item: dict) -> LLMTestCase:
    d = item["data"]
    return LLMTestCase(
        input=f"Intent: {d['intent']}",
        actual_output=f"Risk scenario: {d['atom']}",
    )


def make_q2_case(item: dict) -> LLMTestCase:
    d = item["data"]
    return LLMTestCase(
        input=f"Risk scenario: {d['atom']}",
        actual_output=f"Supporting text: {d['risk_text']}",
    )


def make_q3_case(item: dict) -> LLMTestCase:
    d = item["data"]
    return LLMTestCase(
        input=f"Risk scenario: {d['atom']}",
        context=[d["risk_text"]],
        actual_output=d["synthetic_summary"],
        expected_output=d["entailment_type"],
    )


# ---------------------------------------------------------------------------
# 4. Result container
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    id: Any
    source_file: str
    question: str
    raw_score: float
    predicted_label: str
    reason: str


# ---------------------------------------------------------------------------
# 5. Runner -- one loop per question
# ---------------------------------------------------------------------------

def run_question(
    dataset: list[dict],
    question_name: str,
    metric: GEval,
    case_builder: Callable[[dict], LLMTestCase],
    labels: dict,
    neg_max: float = 0.25,
    mid_max: float = 0.45,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for _, item in tqdm(enumerate(dataset)):
        d = item["data"]
        test_case = case_builder(item)
        metric.measure(test_case)
        results.append(
            QuestionResult(
                id=item.get("id"),
                source_file=d.get("source_file", ""),
                question=question_name,
                raw_score=metric.score,
                predicted_label=score_to_label(metric.score, labels, neg_max=neg_max, mid_max=mid_max),
                reason=metric.reason,
            )
        )
    return results


def write_csv(path: str, results: list[QuestionResult]) -> None:
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval GEval as an automated annotator over the risk dataset.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset JSON file (list of {id, data} records).")
    parser.add_argument("--outdir", default="results_deepeval_v4", help="Directory to write CSV/JSON reports to.")
    parser.add_argument("--model", default=None, help="Optional judge model name/instance to pass to GEval (defaults to deepeval's default OpenAI judge).")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    # --- Loop 1: Intent <-> Atom consistency -------------------------------
    # The rubric maps 0-2/10 -> inconsistent, 3-4/10 -> unclear, 5-10/10 ->
    # consistent, i.e. 0-0.2 / 0.3-0.4 / 0.5-1.0 once normalized. Cutoffs
    # below sit in the gaps between those bands.
    q1_metric = build_q1_metric(model=args.model)
    q1_results = run_question(
        dataset, "consistency", q1_metric, make_q1_case, Q1_LABELS,
        neg_max=0.25, mid_max=0.45,
    )
    write_csv(os.path.join(args.outdir, "q1_consistency.csv"), q1_results)

    # --- Loop 2: Atom <-> Risk text relevance -------------------------------
    q2_metric = build_q2_metric(model=args.model)
    q2_results = run_question(dataset, "relevance", q2_metric, make_q2_case, Q2_LABELS)
    write_csv(os.path.join(args.outdir, "q2_relevance.csv"), q2_results)

    # --- Loop 3: Entailment type verification -------------------------------
    q3_metric = build_q3_metric(model=args.model)
    q3_results = run_question(dataset, "entailment_check", q3_metric, make_q3_case, Q3_LABELS)
    write_csv(os.path.join(args.outdir, "q3_entailment.csv"), q3_results)

    # --- Combined report, one row per item with all three verdicts ---------
    by_id = {}
    for r in q1_results + q2_results + q3_results:
        row = by_id.setdefault(r.id, {"id": r.id, "source_file": r.source_file})
        row[f"{r.question}_label"] = r.predicted_label
        row[f"{r.question}_score"] = r.raw_score
        row[f"{r.question}_reason"] = r.reason

    combined_path = os.path.join(args.outdir, "combined.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(list(by_id.values()), f, indent=2)

    print(f"Wrote per-question CSVs and {combined_path} to {args.outdir}/")


if __name__ == "__main__":
    main()