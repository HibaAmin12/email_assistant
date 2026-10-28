from dotenv import load_dotenv
load_dotenv()
from langsmith import evaluate
from app import classify

def target(inputs: dict):
    # humara asli classify node, bina graph ke
    return classify(inputs)          # {"category": "..."}

def correct(outputs: dict, reference_outputs: dict):
    return {
        "key": "correct",
        "score": int(outputs["category"] == reference_outputs["category"]),
    }

results = evaluate(
    target,
    data="email-classify",
    evaluators=[correct],
    experiment_prefix="classify-v1",
)

df = results.to_pandas()
print(df.columns.tolist())   # pehle dekhein ke columns ke naam kya hain