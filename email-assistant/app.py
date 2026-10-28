from dotenv import load_dotenv
load_dotenv()

from datetime import date, timedelta
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

llm = init_chat_model("openai:gpt-4o-mini", temperature=0)

# Pehle False (evaluator ka masla dekhne ke liye), phir True (rule 8 on)
STRICT_FACTS = True

# ---------- NAKLI CALENDAR ----------
TODAY = date.today()
TOMORROW = TODAY + timedelta(days=1)
DAY_AFTER = TODAY + timedelta(days=2)

BUSY = {
    (TOMORROW.isoformat(), "10:00"): "Team standup",
    (TOMORROW.isoformat(), "17:00"): "Client call",
    (TOMORROW.isoformat(), "18:00"): "Client call (continued)",
}
WORK_HOURS = [f"{h:02d}:00" for h in range(9, 19)]

# ---------- TOOL ----------
@tool
def check_calendar(date: str, time: str) -> str:
    """Check whether I am free at a given date and time.
    date must be in YYYY-MM-DD format. time must be 24-hour HH:MM format (e.g. 17:00)."""
    event = BUSY.get((date, time))
    if event:
        free = [t for t in WORK_HOURS if (date, t) not in BUSY]
        return f"BUSY at {date} {time} ({event}). Free slots that day: {', '.join(free)}"
    return f"FREE at {date} {time}."

tools = [check_calendar]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# ---------- STATE ----------
class State(TypedDict):
    sender: str
    subject: str
    body: str
    category: str
    draft: str
    decision: str
    feedback: str
    rewrites: int
    messages: Annotated[list, add_messages]
    calendar_info: str

# ---------- Structured output ----------
class Classification(BaseModel):
    category: Literal["urgent", "meeting", "question", "spam", "fyi"]

classifier = llm.with_structured_output(Classification)

# ---------- NODES ----------
def classify(state: State):
    result = classifier.invoke(
        "Classify this email into exactly one category.\n\n"
        "CATEGORIES:\n"
        "- urgent: needs action or reply very soon (deadline today/tomorrow, "
        "something broken, boss/client pressure).\n"
        "- meeting: about arranging, confirming or scheduling a meeting, call or "
        "meetup, including short confirmations like 'kal pakka?' (is tomorrow fixed?).\n"
        "- question: asks for information or an answer, with no deadline or scheduling.\n"
        "- spam: unwanted promotions, scams, phishing, prize or password tricks.\n"
        "- fyi: information only, no reply needed. Includes legitimate newsletters, "
        "digests, notices and 'thanks, got it' messages, even if they contain an "
        "unsubscribe link.\n\n"
        f"From: {state['sender']}\n"
        f"Subject: {state['subject']}\n"
        f"Body: {state['body']}"
    )
    return {"category": result.category}


def calendar_agent(state: State):
    history = state.get("messages", [])
    new = []
    if not history:
        first = HumanMessage(
            content=f"Subject: {state['subject']}\nBody: {state['body']}"
        )
        new = [first]
        history = new

    system = SystemMessage(content=(
        f"Today is {TODAY.isoformat()} ({TODAY.strftime('%A')}). "
        f"kal / tomorrow = {TOMORROW.isoformat()}. "
        f"parso / day after tomorrow = {DAY_AFTER.isoformat()}. "
        "Use these dates EXACTLY. Never calculate dates yourself. "
        "The email below is a meeting request. Work out the requested time "
        "(evening/shaam 5 baje = 17:00, afternoon/dopahar 3 baje = 15:00) "
        "and use the check_calendar tool to see if I am free. "
        "After you get the result, reply with ONE short summary: whether I am free or busy "
        "at the requested time, and alternative free slots if busy. Do not write an email."
    ))
    response = llm_with_tools.invoke([system] + history)
    return {"messages": new + [response]}


def finalize_calendar(state: State):
    return {"calendar_info": state["messages"][-1].content}


def draft_reply(state: State):
    rules = (
        "You are drafting an email reply on my behalf.\n\n"
        "RULES:\n"
        "1. LANGUAGE: Write the reply in professional English by default, EVEN IF the "
        "incoming email is in Roman Urdu or another language. Only use a different "
        "language if my feedback below explicitly asks for it.\n"
        "2. Keep it short, polite and natural.\n"
        "3. Do NOT add a 'Subject:' line.\n"
        "4. Do NOT use square brackets [ ] anywhere. Greet with just 'Hello,' "
        "and end without any signature or 'Best regards'.\n"
        "5. Do NOT make promises or commitments I did not authorize. "
        "The ONLY exception: if calendar info is given below, you may accept the "
        "requested time if I am free, or politely decline and suggest the free "
        "alternatives if I am busy. Never invent availability beyond the calendar info.\n"
        "6. Translate relative-date words into English "
        "(kal = tomorrow, parso = the day after tomorrow, aaj = today).\n"
        "7. NEVER write 'I will ensure', 'I will send' or any deadline promise unless the "
        "calendar info allows it. For work requests, just acknowledge receipt, e.g. "
        "'Thank you for your message. I have received your request and will get back to you shortly.'\n"
    )

    if STRICT_FACTS:
        rules += (
            "8. NEVER answer factual questions (policies, deadlines, formats, statuses, "
            "yes/no questions) yourself, because you have no access to that information. "
            "Reply only that you have received the question and will check and get back.\n"
        )

    prompt = rules + (
        "\n"
        f"Email category: {state['category']}\n"
        f"From: {state['sender']}\n"
        f"Subject: {state['subject']}\n"
        f"Body: {state['body']}\n\n"
    )

    if state.get("calendar_info"):
        prompt += f"Calendar info: {state['calendar_info']}\n\n"

    if state.get("feedback"):
        prompt += (
            f"Previous draft:\n{state['draft']}\n\n"
            f"My feedback on the previous draft: {state['feedback']}\n"
            "Rewrite the draft following this feedback. "
            "If the feedback asks for a language or tone, follow it.\n\n"
        )
    else:
        prompt += "Final reminder: the reply MUST be written in English.\n"

    prompt += "Write only the reply text."
    draft = llm.invoke(prompt).content

    # Guard: placeholder pakda gaya to ek baar dobara likhwao
    if "[" in draft:
        draft = llm.invoke(
            prompt + "\nYour previous attempt contained a placeholder in square brackets. "
            "Rewrite it with NO square brackets at all."
        ).content

    return {"draft": draft, "feedback": ""}


def ignore(state: State):
    return {"draft": "(ignore kiya gaya, reply ki zarurat nahi)"}


def human_approval(state: State):
    response = interrupt({
        "from": state["sender"],
        "subject": state["subject"],
        "draft": state["draft"],
        "question": "approve / edit / rewrite / reject?",
    })
    action = response["action"]

    if action == "approve":
        return {"decision": "approved"}
    if action == "edit":
        return {"decision": "edited", "draft": response["text"]}
    if action == "rewrite":
        return {
            "decision": "rewrite",
            "feedback": response["text"],
            "rewrites": state.get("rewrites", 0) + 1,
        }
    return {"decision": "rejected"}


def send(state: State):
    print(f"\n   [SENT to {state['sender']}]\n   {state['draft']}")
    return {}

# ---------- CONDITIONAL FUNCTIONS ----------
def route_by_category(state: State):
    if state["category"] in ("spam", "fyi"):
        return "ignore"
    if state["category"] == "meeting":
        return "calendar_agent"
    return "draft_reply"


def after_agent(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


def route_after_approval(state: State):
    if state["decision"] == "rejected":
        return "end"
    if state["decision"] == "rewrite":
        if state.get("rewrites", 0) >= 3:
            return "end"
        return "draft_reply"
    return "send"

# ---------- GRAPH ----------
g = StateGraph(State)
g.add_node("classify", classify)
g.add_node("calendar_agent", calendar_agent)
g.add_node("tools", tool_node)
g.add_node("finalize", finalize_calendar)
g.add_node("draft_reply", draft_reply)
g.add_node("ignore", ignore)
g.add_node("human_approval", human_approval)
g.add_node("send", send)

g.add_edge(START, "classify")
g.add_conditional_edges(
    "classify",
    route_by_category,
    {"ignore": "ignore", "calendar_agent": "calendar_agent", "draft_reply": "draft_reply"},
)
g.add_conditional_edges(
    "calendar_agent",
    after_agent,
    {"tools": "tools", "finalize": "finalize"},
)
g.add_edge("tools", "calendar_agent")
g.add_edge("finalize", "draft_reply")
g.add_edge("draft_reply", "human_approval")
g.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {"send": "send", "draft_reply": "draft_reply", "end": END},
)
g.add_edge("send", END)
g.add_edge("ignore", END)

app = g.compile(checkpointer=InMemorySaver())

# ---------- RUN ----------
if __name__ == "__main__":
    from messages import MESSAGES

    for m in MESSAGES:
        config = {
            "configurable": {"thread_id": f"email-{m['id']}-v7"},
            "run_name": f"email-msg-{m['id']}",
        }
        out = app.invoke(
            {"sender": m["from"], "subject": m["subject"], "body": m["body"],
             "rewrites": 0, "feedback": "", "messages": [], "calendar_info": ""},
            config,
        )

        print(f"\n{m['id']}. {m['subject']}  [{out['category']}]")

        if "__interrupt__" not in out:
            print("   (ignore kiya gaya)")
            continue

        if out.get("calendar_info"):
            print(f"   Calendar: {out['calendar_info']}")

        while "__interrupt__" in out:
            info = out["__interrupt__"][0].value
            print(f"\n   Draft: {info['draft']}")

            raw = input("   approve / edit / rewrite / reject? ").strip()
            parts = raw.split(maxsplit=1)
            choice = parts[0].lower() if parts else ""
            inline = parts[1] if len(parts) > 1 else ""

            if choice not in ("approve", "edit", "rewrite", "reject"):
                print("   Samajh nahi aaya, dobara likhein.")
                continue

            if choice == "approve":
                resume = {"action": "approve"}
            elif choice == "edit":
                resume = {"action": "edit", "text": inline or input("   Apna reply likhein (ek line): ")}
            elif choice == "rewrite":
                resume = {"action": "rewrite", "text": inline or input("   Kya badlun? ")}
            else:
                resume = {"action": "reject"}

            out = app.invoke(Command(resume=resume), config)

        print(f"   Decision: {out['decision']}")