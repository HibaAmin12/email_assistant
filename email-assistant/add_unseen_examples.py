from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

client = Client()

unseen = [
    ("ali@friend.com", "Dinner?", "Aaj raat khana khane chalein? 8 baje?", "meeting"),
    ("boss@company.com", "Fix this now", "Dashboard mein numbers ghalat hain, board meeting 1 ghante mein hai!", "urgent"),
    ("zain@colleague.com", "Password policy", "Kya humari company ka password har 90 din mein badalna zaroori hai?", "question"),
    ("offers@cheapmeds.ru", "Cheap pills", "Buy now no prescription needed, limited offer, click here!!!", "spam"),
    ("admin@company.com", "Parking update", "Next week se basement parking band rahegi, bahar parking use karein.", "fyi"),
    ("client@abc.com", "Call tomorrow", "Can we hop on a call tomorrow 4 PM to review the proposal?", "meeting"),
]

client.create_examples(
    dataset_id=client.read_dataset(dataset_name="email-classify").id,
    examples=[
        {"inputs": {"sender": s, "subject": sub, "body": b}, "outputs": {"category": c}}
        for s, sub, b, c in unseen
    ],
)
print(f"{len(unseen)} unseen examples add hue")