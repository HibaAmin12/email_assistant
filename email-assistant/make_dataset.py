from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

client = Client()
NAME = "email-classify"

if client.has_dataset(dataset_name=NAME):
    print("Dataset pehle se maujood hai")
    raise SystemExit

ds = client.create_dataset(NAME, description="Email category classification")

data = [
    ("boss@company.com", "Report kab tak milegi?", "Q3 ki report aaj shaam tak chahiye. Ready ho jayegi?", "urgent"),
    ("client@abc.com", "URGENT: server down", "Our website is down since morning, please fix ASAP.", "urgent"),
    ("ali@friend.com", "Kal milte hain?", "Yaar kal shaam 5 baje chai pe milte hain?", "meeting"),
    ("sara@partner.com", "Parso meeting?", "Kya hum parso dopahar 3 baje 30 minute ki meeting kar sakte hain?", "meeting"),
    ("hr@company.com", "Interview schedule", "Can we set up a call on Friday at 11 AM to discuss your application?", "meeting"),
    ("deals@shopnow.com", "50% OFF!!!", "Sirf aaj ke liye mega sale. Abhi click karein aur jeetein inaam!", "spam"),
    ("lottery@win.biz", "You won $1,000,000", "Click this link to claim your prize now!", "spam"),
    ("hr@company.com", "Office holiday notice", "Yaad dahani: Jumma ko office band rahega.", "fyi"),
    ("it@company.com", "Maintenance tonight", "FYI, the VPN will be down from 2 AM to 3 AM tonight. No action needed.", "fyi"),
    ("zain@colleague.com", "Invoice ka format", "Bhai invoice mai GST alag likhna hai ya total mai? Tumhe pata hai?", "question"),
    ("student@uni.edu", "Doubt in assignment", "Sir, is the deadline for assignment 2 extended? Could you confirm?", "question"),
    ("manager@company.com", "Contract kal tak sign karo", "Client contract kal subah tak sign hona zaroori hai warna deal khatam.", "urgent"),
]

client.create_examples(
    dataset_id=ds.id,
    examples=[
        {
            "inputs": {"sender": s, "subject": sub, "body": b},
            "outputs": {"category": cat},
        }
        for s, sub, b, cat in data
    ],
)
print(f"{len(data)} examples ban gaye")