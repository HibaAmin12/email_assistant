from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

client = Client()

hard = [
    # urgent vs meeting
    ("boss@company.com", "Abhi call karo", "Mujhe abhi 10 minute ke liye call chahiye, bohat zaroori hai.", "urgent"),
    ("client@abc.com", "Quick sync?", "Whenever you get a chance this week, let's sync on the project. No rush.", "meeting"),
    # question vs urgent
    ("ali@friend.com", "Ek sawal", "Bhai urgent nahi hai, bas ye batao ke kal ki party ka time kya hai?", "question"),
    ("cto@company.com", "Deploy question", "Production deploy ruka hua hai, kya hum rollback karein? Jaldi batao.", "urgent"),
    # spam vs fyi
    ("newsletter@tech.com", "Weekly digest", "Is hafte ki top 5 tech khabrain. Unsubscribe karne ke liye yahan click karein.", "fyi"),
    ("promo@bank-secure.xyz", "Account suspended", "Your account is suspended. Verify your password here immediately.", "spam"),
    # fyi vs meeting
    ("hr@company.com", "Town hall Friday", "Reminder: company town hall Friday 3 PM. Attendance optional, recording will be shared.", "fyi"),
    ("team@company.com", "Planning meeting", "Please confirm if you can join the sprint planning on Monday at 10 AM.", "meeting"),
    # mixed language / short
    ("ali@friend.com", "ok?", "kal pakka?", "meeting"),
    ("sara@partner.com", "Thanks", "Thanks for the update, got it.", "fyi"),
]

client.create_examples(
    dataset_id=client.read_dataset(dataset_name="email-classify").id,
    examples=[
        {"inputs": {"sender": s, "subject": sub, "body": b}, "outputs": {"category": cat}}
        for s, sub, b, cat in hard
    ],
)
print(f"{len(hard)} mushkil examples add hue")