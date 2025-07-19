import os
import json
import discord
import aiohttp
from discord.ext import commands
from discord import app_commands
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
LRS_ENDPOINT = os.getenv("LRS_ENDPOINT")
LRS_USERNAME = os.getenv("LRS_USERNAME")
LRS_PASSWORD = os.getenv("LRS_PASSWORD")

CONFIG_FILE = "channel_configs.json"

# ---------- Discord Setup ----------
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.members = True

class MyBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="/", intents=intents)

bot = MyBot()

# ---------- LLM Backends ----------
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def call_openai(model, system_prompt, user_input):
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ OpenAI error: {e}")
        return "OpenAI backend failed."

async def call_anthropic(model, system_prompt, user_input):
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }
            async with session.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload) as resp:
                data = await resp.json()
                return data["content"][0]["text"]
    except Exception as e:
        print(f"⚠️ Anthropic error: {e}")
        return "Anthropic backend failed."

async def call_mistral(model, system_prompt, user_input):
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": f"{system_prompt}\n\nUser: {user_input}",
            }
            async with session.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload) as resp:
                data = await resp.json()
                return data[0]["generated_text"]
    except Exception as e:
        print(f"⚠️ Mistral error: {e}")
        return "Mistral backend failed."

LLM_BACKENDS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "mistral": call_mistral
}

async def call_llm(backend, model, system_prompt, user_input):
    if backend not in LLM_BACKENDS:
        print(f"⚠️ Unknown backend '{backend}', defaulting to openai.")
        backend = "openai"
    return await LLM_BACKENDS[backend](model, system_prompt, user_input)

# ---------- LRS Logging ----------
async def log_to_lrs(
    actor_name,
    verb,
    message,
    channel_name=None,
    guild_name=None,
    channel_id=None,
    guild_id=None,
    message_id=None,
    parent_message_id=None,
    parent_message_text=None,
    setup_info=None
):
    object_id = f"http://example.com/discord/message/{message_id}" if message_id else "http://example.com/discord/message"

    grouping_activities = [
        {
            "id": f"http://example.com/discord/guild/{guild_id or 'unknown'}",
            "definition": {"name": {"en-US": guild_name or "Unknown Guild"}}
        },
        {
            "id": f"http://example.com/discord/channel/{channel_id or 'unknown'}",
            "definition": {"name": {"en-US": channel_name or "Unknown Channel"}}
        }
    ]

    parent_activities = []
    if parent_message_id:
        parent_activities.append({
            "id": f"http://example.com/discord/message/{parent_message_id}",
            "definition": {
                "name": {"en-US": parent_message_text or "Original Message"},
                "description": {"en-US": "Original user message this response refers to"}
            }
        })

    context = {
        "contextActivities": {
            "grouping": grouping_activities
        }
    }
    if parent_activities:
        context["contextActivities"]["parent"] = parent_activities
    if setup_info:
        context["extensions"] = {
            "https://sonsoleslp.github.io/xapi/extensions/discord-setup": setup_info
        }

    statement = {
        "actor": {
            "name": actor_name,
            "mbox": f"mailto:{actor_name}@example.com"
        },
        "verb": {
            "id": f"http://adlnet.gov/expapi/verbs/{verb}",
            "display": {"en-US": verb}
        },
        "object": {
            "id": object_id,
            "definition": {
                "name": {"en-US": message},
                "description": {"en-US": "A Discord message interaction"}
            }
        },
        "context": context
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            LRS_ENDPOINT,
            auth=aiohttp.BasicAuth(LRS_USERNAME, LRS_PASSWORD),
            headers={"Content-Type": "application/json", "X-Experience-API-Version": "1.0.3"},
            json=statement
        ) as response:
            if response.status not in (200, 204):
                text = await response.text()
                print(f"⚠️ LRS logging failed ({response.status}): {text}")

# ---------- Configuration ----------
def load_configs():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_configs(configs):
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=4)

channel_configs = load_configs()

# ---------- Slash Commands ----------
@bot.tree.command(name="setup")
@app_commands.describe(
    backend="LLM backend to use: openai, anthropic, mistral",
    model="Model name, e.g., gpt-4, claude-3, mistralai/Mistral-7B-Instruct-v0.1",
    system_prompt="System prompt to guide the assistant",
    mode="ask (manual prompt) or auto (auto-respond to questions)"
)
@app_commands.checks.has_permissions(administrator=True)
async def setup(interaction: discord.Interaction, backend: str, model: str, system_prompt: str, mode: str = "ask"):
    channel_id = str(interaction.channel_id)
    if mode not in ["ask", "auto"]:
        await interaction.response.send_message("Invalid mode. Use 'ask' or 'auto'.", ephemeral=True)
        return
    channel_configs[channel_id] = {
        "backend": backend,
        "mode": mode,
        "model": model,
        "system_prompt": system_prompt
    }
    save_configs(channel_configs)
    await interaction.response.send_message(f"Channel configured with backend '{backend}', mode '{mode}', model '{model}'.")

@bot.tree.command(name="ask")
@app_commands.describe(question="Your question for the assistant")
async def ask(interaction: discord.Interaction, question: str):
    channel_id = str(interaction.channel_id)
    config = channel_configs.get(channel_id)
    if not config:
        await interaction.response.send_message("This channel is not configured. Use /setup first.", ephemeral=True)
        return

    await log_to_lrs(
        actor_name=interaction.user.name,
        verb="asked",
        message=question,
        channel_name=interaction.channel.name,
        guild_name=interaction.guild.name if interaction.guild else "DM",
        channel_id=interaction.channel.id,
        guild_id=interaction.guild.id if interaction.guild else None,
        message_id=interaction.id
    )

    response = await call_llm(config.get("backend", "openai"), config["model"], config["system_prompt"], question)
    if (not response) or len(response) == 0:
        return
    await interaction.response.send_message(response)

    await log_to_lrs(
        actor_name="bot",
        verb="responded",
        message=response,
        channel_name=interaction.channel.name,
        guild_name=interaction.guild.name if interaction.guild else "DM",
        channel_id=interaction.channel.id,
        guild_id=interaction.guild.id if interaction.guild else None,
        parent_message_id=interaction.id,
        parent_message_text=question,
        setup_info={
            "backend": config.get("backend"),
            "model": config.get("model"),
            "system_prompt": config.get("system_prompt")
        }
    )

# ---------- Message Listener ----------
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    channel_id = str(message.channel.id)
    config = channel_configs.get(channel_id)

    if config and config["mode"] != "auto":
        await log_to_lrs(
            actor_name=message.author.name,
            verb="sent",
            message=message.content,
            channel_name=message.channel.name,
            guild_name=message.guild.name if message.guild else "DM",
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            message_id=message.id
        )

    await bot.process_commands(message)

    if config and config["mode"] == "auto":
        await log_to_lrs(
            actor_name=message.author.name,
            verb="asked",
            message=message.content,
            channel_name=message.channel.name,
            guild_name=message.guild.name if message.guild else "DM",
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            message_id=message.id
        )

        response = await call_llm(config.get("backend", "openai"), config["model"], config["system_prompt"], message.content)
        if (not response) or len(response) == 0:
            return
        reply = await message.channel.send(response)

        await log_to_lrs(
            actor_name="bot",
            verb="responded",
            message=response,
            channel_name=message.channel.name,
            guild_name=message.guild.name if message.guild else "DM",
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            message_id=reply.id,
            parent_message_id=message.id,
            parent_message_text=message.content,
            setup_info={
                "backend": config.get("backend"),
                "model": config.get("model"),
                "system_prompt": config.get("system_prompt")
            }
        )

# ---------- Bot Ready ----------
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"✅ Bot connected as {bot.user}")

# ---------- Run ----------
bot.run(DISCORD_TOKEN)
