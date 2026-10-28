from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

client = Client()
NAME = "email-drafts"

if client.has_dataset(dataset_name=NAME):
    print("Dataset pehle se maujood hai")
    raise SystemExit

ds = client.create_dataset(NAME, description="Draft reply quality")

data = [
    ("boss@company.com", "Report kab tak milegi?", "Q3 ki report aaj shaam tak chahiye. Ready ho jayegi?", "urgent"),
    ("client@abc.com", "URGENT: server down", "Our website is down since morning, please fix ASAP.", "urgent"),
    ("manager@company.com", "Contract kal tak sign karo", "Client contract kal subah tak sign hona zaroori hai warna deal khatam.", "urgent"),
    ("zain@colleague.com", "Invoice ka format", "Bhai invoice mai GST alag likhna hai ya total mai? Tumhe pata hai?", "question"),
    ("student@uni.edu", "Doubt in assignment", "Sir, is the deadline for assignment 2 extended? Could you confirm?", "question"),
    ("zain@colleague.com", "Password policy", "Kya humari company ka password har 90 din mein badalna zaroori hai?", "question"),
]

client.create_examples(
    dataset_id=ds.id,
    examples=[
        {
            "inputs": {
                "sender": s, "subject": sub, "body": b, "category": c,
                "feedback": "", "calendar_info": "", "draft": "",
            },
            "outputs": {},
        }
        for s, sub, b, c in data
    ],
)
print(f"{len(data)} examples ban gaye")