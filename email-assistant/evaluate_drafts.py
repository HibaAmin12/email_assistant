from dotenv import load_dotenv
load_dotenv()
import re
from pydantic import BaseModel
from langsmith import evaluate
from app import draft_reply, llm, STRICT_FACTS

def target(inputs: dict):
    return draft_reply(inputs)      # {"draft": "...", "feedback": ""}

# ---------- Rule-based evaluators (code) ----------
def no_placeholder(outputs: dict):
    return {"key": "no_placeholder", "score": int("[" not in outputs["draft"])}

def no_subject_line(outputs: dict):
    return {"key": "no_subject_line",
            "score": int(not outputs["draft"].strip().lower().startswith("subject:"))}

PROMISE = re.compile(r"\bI (will|'ll) (ensure|send|deliver|complete|finish|review)\b", re.I)

def no_promise(outputs: dict):
    return {"key": "no_promise", "score": int(not PROMISE.search(outputs["draft"]))}

ROMAN_URDU = re.compile(r"\b(kal|parso|aaj|shukriya|aapka|hoon|karunga|zaroor)\b", re.I)

def english_only(outputs: dict):
    return {"key": "english_only", "score": int(not ROMAN_URDU.search(outputs["draft"]))}

# ---------- LLM-as-judge: quality ----------
class Judgement(BaseModel):
    polite: bool
    answers_email: bool
    has_typo: bool

judge = llm.with_structured_output(Judgement)

def judged(inputs: dict, outputs: dict):
    j = judge.invoke(
        "Evaluate this email reply.\n\n"
        f"Original email: {inputs['subject']} - {inputs['body']}\n"
        f"Reply: {outputs['draft']}\n\n"
        "polite: is the tone polite and professional?\n"
        "answers_email: does the reply appropriately respond to the email "
        "(an acknowledgement is acceptable if it makes no unauthorized promise)?\n"
        "has_typo: does the reply contain spelling mistakes or words stuck together?"
    )
    return [
        {"key": "polite", "score": int(j.polite)},
        {"key": "answers_email", "score": int(j.answers_email)},
        {"key": "no_typo", "score": int(not j.has_typo)},
    ]

# ---------- LLM-as-judge: fabrication ----------
class FactCheck(BaseModel):
    asks_for_fact: bool
    states_unverified_fact: bool

fact_judge = llm.with_structured_output(FactCheck)

def no_fabrication(inputs: dict, outputs: dict):
    r = fact_judge.invoke(
        "The assistant writing the reply has NO access to any company information.\n\n"
        f"Email: {inputs['subject']} - {inputs['body']}\n"
        f"Reply: {outputs['draft']}\n\n"
        "asks_for_fact: does the email ask for a specific fact (a policy, a deadline "
        "status, a format, a yes/no answer)? A request for action or a report is NOT a fact question.\n"
        "states_unverified_fact: does the reply state or confirm a specific fact "
        "(yes/no answer, policy, deadline, number)? Acknowledgements such as "
        "'I have received your request' or 'I will get back to you' are NOT facts."
    )
    bad = r.asks_for_fact and r.states_unverified_fact
    return {"key": "no_fabrication", "score": int(not bad)}


prefix = "drafts-v3-rule8" if STRICT_FACTS else "drafts-v2-no-rule8"

evaluate(
    target,
    data="email-drafts",
    evaluators=[no_placeholder, no_subject_line, no_promise,
                english_only, judged, no_fabrication],
    experiment_prefix=prefix,
)