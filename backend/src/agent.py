

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal, Optional, List
from dataclasses import dataclass, asdict


print("💡 bikeya_agent.py LOADED SUCCESSFULLY!")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

# 🔌 PLUGINS
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📂 1. KNOWLEDGE BASE (FAQ)
# ======================================================

FAQ_FILE = "store_faq.json"
LEADS_FILE = "leads_db.json"

# Default FAQ data for "Bikeya Company Profile"
DEFAULT_FAQ = [
    {
        "question": "What kind of bikes does Bikeya sell?",
        "answer": "Bikeya specializes in custom-built, high-performance road, gravel, and time trial bikes. We focus on lightweight carbon frames and personalized component selection."
    },
    {
        "question": "How much does a custom bike fitting session cost?",
        "answer": "Our comprehensive 3D dynamic bike fitting session, essential for all custom orders, is priced at $350. This fee is often waived if you purchase a complete bike package."
    },
    {
        "question": "Do you offer financing or payment plans?",
        "answer": "Yes, we partner with specialized cycling finance companies to offer 6 and 12-month payment plans, subject to approval."
    },
    {
        "question": "What is the typical lead time for a custom bike?",
        "answer": "Depending on component availability, a custom Bikeya frame and build takes approximately 8 to 12 weeks from the final fit session to delivery."
    }
]

def load_knowledge_base():
    """Generates FAQ file if missing, then loads it."""
    try:
        # Note: We'll generate the new store_faq.json below, but the logic remains the same.
        path = os.path.join(os.path.dirname(__file__), FAQ_FILE)
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_FAQ, f, indent=4)
        with open(path, "r", encoding='utf-8') as f:
            return json.dumps(json.load(f)) # Return as string for the Prompt
    except Exception as e:
        print(f"⚠️ Error loading FAQ: {e}")
        return ""

STORE_FAQ_TEXT = load_knowledge_base()

# ======================================================
# 💾 2. LEAD DATA STRUCTURE
# ======================================================

@dataclass
class LeadProfile:
    name: str | None = None
    company: str | None = None
    email: str | None = None
    role: str | None = None # e.g., 'Amateur Racer', 'Weekend Rider'
    use_case: str | None = None # e.g., 'Custom Road Bike', 'Gravel Setup', 'Coaching'
    team_size: str | None = None # e.g., 'Individual', 'Small Racing Team'
    timeline: str | None = None
    
    def is_qualified(self):
        """Returns True if we have the minimum info (Name + Email + Use Case)"""
        return all([self.name, self.email, self.use_case])

@dataclass
class Userdata:
    lead_profile: LeadProfile

# ======================================================
# 🛠️ 3. SDR TOOLS (Functionality remains the same, adjusted context)
# ======================================================

@function_tool
async def update_lead_profile(
    ctx: RunContext[Userdata],
    name: Annotated[Optional[str], Field(description="Customer's name")] = None,
    company: Annotated[Optional[str], Field(description="Customer's cycling club or current bike brand")] = None,
    email: Annotated[Optional[str], Field(description="Customer's email address")] = None,
    role: Annotated[Optional[str], Field(description="Customer's main riding category (e.g., weekend warrior, racer, commuter)")] = None,
    use_case: Annotated[Optional[str], Field(description="What specific type of bike or service they are interested in (e.g., custom road bike, gravel wheels, performance coaching)")] = None,
    team_size: Annotated[Optional[str], Field(description="Individual or size of their cycling team")] = None,
    timeline: Annotated[Optional[str], Field(description="When they plan to acquire the bike/service (e.g., Spring 2026, Now, within 6 months)")] = None,
) -> str:
    """
    ✍️ Captures lead details provided by the user during conversation for Bikeya's custom products.
    Only call this when the user explicitly provides information.
    """
    profile = ctx.userdata.lead_profile
    
    # Update only fields that are provided (not None)
    if name: profile.name = name
    if company: profile.company = company
    if email: profile.email = email
    if role: profile.role = role
    if use_case: profile.use_case = use_case
    if team_size: profile.team_size = team_size
    if timeline: profile.timeline = timeline
    
    print(f"📝 UPDATING LEAD: {profile}")
    return "Lead profile updated. Continue the conversation."

@function_tool
async def submit_lead_and_end(
    ctx: RunContext[Userdata],
) -> str:
    """
    💾 Saves the lead to the database and signals the end of the call.
    Call this when the user says goodbye or 'that's all'.
    """
    profile = ctx.userdata.lead_profile
    
    # Save to JSON file (Append mode)
    db_path = os.path.join(os.path.dirname(__file__), LEADS_FILE)
    
    entry = asdict(profile)
    entry["timestamp"] = datetime.now().isoformat()
    
    # Read existing, append, write back (Simple JSON DB)
    existing_data = []
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                existing_data = json.load(f)
        except: pass
    
    existing_data.append(entry)
    
    with open(db_path, "w") as f:
        json.dump(existing_data, f, indent=4)
        
    print(f"✅ LEAD SAVED TO {LEADS_FILE}")
    return f"Lead saved. Summarize the call for the user: 'Thanks {profile.name}, I have your info regarding a {profile.use_case} build. We will send the consultation schedule to {profile.email}. Happy cycling, goodbye!'"

# ======================================================
# 🧠 4. AGENT DEFINITION
# ======================================================

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""
            You are 'Liam', a friendly and professional Custom Bike Consultant for 'Bikeya Company'.
            
            📘 **YOUR KNOWLEDGE BASE (FAQ):**
            {STORE_FAQ_TEXT}
            
            🎯 **YOUR GOAL:**
            1. Answer questions about our custom bikes, fittings, and services using the FAQ.
            2. **QUALIFY THE LEAD:** Naturally ask for the following details during the chat:
                - Name
                - Company / Role (Riding style or club)
                - Email
                - What specific bike or service are they looking for? (Use Case)
                - Timeline (When do they want the new bike?)
            
            ⚙️ **BEHAVIOR:**
            - **Be Conversational:** Don't interrogate the user. Answer a question, THEN ask for a detail.
            - *Example:* "Our custom fit session is $350. It ensures the perfect geometry. By the way, what kind of riding are you primarily focused on (road, gravel, or racing)?"
            - **Capture Data:** Use `update_lead_profile` immediately when you hear new info.
            - **Closing:** When the user is done, use `submit_lead_and_end`.
            
            🚫 **RESTRICTIONS:**
            - If you don't know an answer, say "I'll check with our Master Builder and email you the details." (Don't hallucinate prices).
            """,
            tools=[update_lead_profile, submit_lead_and_end],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    print("\n" + "🚴‍♂️" * 25)
    print("🚀 STARTING BIKEYA CONSULTANT SESSION")
    
    # 1. Initialize State
    userdata = Userdata(lead_profile=LeadProfile())

    # 2. Setup Agent
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-alicia", # Warm, professional male voice for a consultant
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    # 3. Start
    await session.start(
        agent=SDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))