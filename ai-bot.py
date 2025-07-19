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

# ---------- Conversation Management ----------
# Store conversation history per channel
channel_conversations = {}
# Store OpenAI clients per channel
channel_openai_clients = {}

def get_conversation_history(channel_id):
    """Get or create conversation history for a channel"""
    if channel_id not in channel_conversations:
        channel_conversations[channel_id] = []
    return channel_conversations[channel_id]

def initialize_conversation(channel_id, system_prompt):
    """Initialize conversation with system prompt"""
    channel_conversations[channel_id] = [{"role": "system", "content": system_prompt}]

def add_to_conversation(channel_id, role, content):
    """Add a message to the conversation history"""
    conversation = get_conversation_history(channel_id)
    conversation.append({"role": role, "content": content})
    
    # Limit conversation history to prevent token overflow
    max_messages = 100  # Adjust based on your needs
    if len(conversation) > max_messages:
        # Keep system message and remove oldest user/assistant messages
        system_msg = conversation[0] if conversation[0]["role"] == "system" else None
        other_messages = [msg for msg in conversation if msg["role"] != "system"]
        
        # Keep the most recent messages
        recent_messages = other_messages[-(max_messages - (1 if system_msg else 0)):]
        
        if system_msg:
            channel_conversations[channel_id] = [system_msg] + recent_messages
        else:
            channel_conversations[channel_id] = recent_messages

def clear_conversation(channel_id):
    """Clear conversation history for a channel"""
    if channel_id in channel_conversations:
        del channel_conversations[channel_id]

def get_openai_client(channel_id):
    """Get or create OpenAI client for a channel"""
    if channel_id not in channel_openai_clients:
        channel_openai_clients[channel_id] = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return channel_openai_clients[channel_id]

# ---------- LLM Backends ----------
async def call_openai(model, user_input, channel_id):
    """OpenAI backend with conversation history"""
    try:
        client = get_openai_client(channel_id)
        
        # Add user message to conversation
        add_to_conversation(channel_id, "user", user_input)
        
        # Get full conversation history
        messages = get_conversation_history(channel_id)
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to conversation
        add_to_conversation(channel_id, "assistant", assistant_response)
        
        return assistant_response
        
    except Exception as e:
        print(f"⚠️ OpenAI error: {e}")
        return "OpenAI backend failed."

async def call_anthropic(model, user_input, channel_id):
    """Anthropic backend with conversation history"""
    try:
        conversation = get_conversation_history(channel_id)
        
        # Extract system prompt and messages
        system_prompt = ""
        messages = []
        
        for msg in conversation:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "max_tokens": 1000,
                "system": system_prompt,
                "messages": messages
            }
            
            async with session.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload) as resp:
                data = await resp.json()
                
                if resp.status != 200:
                    print(f"⚠️ Anthropic API error: {data}")
                    return "Anthropic backend failed."
                
                response_text = data["content"][0]["text"]
                
                # Update conversation history
                add_to_conversation(channel_id, "user", user_input)
                add_to_conversation(channel_id, "assistant", response_text)
                
                return response_text
                
    except Exception as e:
        print(f"⚠️ Anthropic error: {e}")
        return "Anthropic backend failed."

async def call_mistral(model, user_input, channel_id):
    """Mistral backend with conversation history via HuggingFace"""
    try:
        conversation = get_conversation_history(channel_id)
        
        # Build conversation prompt for Mistral
        conversation_text = ""
        for msg in conversation:
            if msg["role"] == "system":
                conversation_text += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                conversation_text += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                conversation_text += f"Assistant: {msg['content']}\n\n"
        
        # Add current user message
        conversation_text += f"User: {user_input}\n\nAssistant:"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": conversation_text,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "do_sample": True,
                    "stop": ["User:", "System:"]
                }
            }
            
            async with session.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload) as resp:
                data = await resp.json()
                
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get("generated_text", "")
                    
                    # Extract only the new response (after "Assistant:")
                    if "Assistant:" in generated_text:
                        response_text = generated_text.split("Assistant:")[-1].strip()
                    else:
                        response_text = generated_text.replace(conversation_text, "").strip()
                    
                    # Update conversation history
                    add_to_conversation(channel_id, "user", user_input)
                    add_to_conversation(channel_id, "assistant", response_text)
                    
                    return response_text
                else:
                    print(f"⚠️ Unexpected Mistral response: {data}")
                    return "Mistral backend failed."
                    
    except Exception as e:
        print(f"⚠️ Mistral error: {e}")
        return "Mistral backend failed."

# For local/custom models via HuggingFace
async def call_custom_hf_model(model, user_input, channel_id):
    """Generic HuggingFace model backend with conversation history"""
    try:
        conversation = get_conversation_history(channel_id)
        
        # Build conversation for chat models
        if "chat" in model.lower() or "instruct" in model.lower():
            conversation_text = ""
            for msg in conversation:
                if msg["role"] == "system":
                    conversation_text += f"<|system|>\n{msg['content']}\n"
                elif msg["role"] == "user":
                    conversation_text += f"<|user|>\n{msg['content']}\n"
                elif msg["role"] == "assistant":
                    conversation_text += f"<|assistant|>\n{msg['content']}\n"
            
            conversation_text += f"<|user|>\n{user_input}\n<|assistant|>\n"
        else:
            # Fallback format
            conversation_text = ""
            for msg in conversation:
                role = msg["role"].capitalize()
                conversation_text += f"{role}: {msg['content']}\n\n"
            conversation_text += f"User: {user_input}\n\nAssistant:"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": conversation_text,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            async with session.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload) as resp:
                data = await resp.json()
                
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get("generated_text", "")
                    
                    # Extract only the new response
                    response_text = generated_text.replace(conversation_text, "").strip()
                    
                    # Clean up common artifacts
                    if response_text.startswith("Assistant:"):
                        response_text = response_text[10:].strip()
                    
                    # Update conversation history
                    add_to_conversation(channel_id, "user", user_input)
                    add_to_conversation(channel_id, "assistant", response_text)
                    
                    return response_text
                else:
                    print(f"⚠️ Unexpected HF model response: {data}")
                    return "HuggingFace model backend failed."
                    
    except Exception as e:
        print(f"⚠️ HuggingFace model error: {e}")
        return "HuggingFace model backend failed."

LLM_BACKENDS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "mistral": call_mistral,
    "huggingface": call_custom_hf_model
}

async def call_llm(backend, model, user_input, channel_id):
    """Call LLM with conversation history"""
    if backend not in LLM_BACKENDS:
        print(f"⚠️ Unknown backend '{backend}', defaulting to openai.")
        backend = "openai"
    
    # All backends now use conversation history
    return await LLM_BACKENDS[backend](model, user_input, channel_id)

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
    backend="LLM backend to use: openai, anthropic, mistral, huggingface",
    model="Model name, e.g., gpt-4, claude-3-sonnet, mistralai/Mistral-7B-Instruct-v0.1",
    system_prompt="System prompt to guide the assistant",
    mode="ask (manual prompt) or auto (auto-respond to questions)"
)
@app_commands.checks.has_permissions(administrator=True)
async def setup(interaction: discord.Interaction, backend: str, model: str, system_prompt: str, mode: str = "ask"):
    channel_id = str(interaction.channel_id)
    if mode not in ["ask", "auto"]:
        await interaction.response.send_message("Invalid mode. Use 'ask' or 'auto'.", ephemeral=True)
        return
    
    # Save configuration
    channel_configs[channel_id] = {
        "backend": backend,
        "mode": mode,
        "model": model,
        "system_prompt": system_prompt
    }
    save_configs(channel_configs)
    
    # Initialize conversation with system prompt
    initialize_conversation(channel_id, system_prompt)
    
    await interaction.response.send_message(
        f"✅ Channel configured:\n"
        f"**Backend:** {backend}\n"
        f"**Model:** {model}\n"
        f"**Mode:** {mode}\n"
        f"**System prompt set and conversation initialized.**"
    )

@bot.tree.command(name="ask")
@app_commands.describe(question="Your question for the assistant")
async def ask(interaction: discord.Interaction, question: str):
    channel_id = str(interaction.channel_id)
    config = channel_configs.get(channel_id)
    if not config:
        await interaction.response.send_message("This channel is not configured. Use /setup first.", ephemeral=True)
        return

    # Check if conversation is initialized
    conversation = get_conversation_history(channel_id)
    if not conversation:
        initialize_conversation(channel_id, config["system_prompt"])

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

    # Call LLM with just the user input (system prompt already in conversation history)
    response = await call_llm(
        config["backend"], 
        config["model"], 
        question,
        channel_id
    )
    
    if (not response) or len(response) == 0:
        await interaction.response.send_message("Sorry, I couldn't generate a response.", ephemeral=True)
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

@bot.tree.command(name="clear_conversation")
@app_commands.describe()
async def clear_conversation_cmd(interaction: discord.Interaction):
    """Clear the conversation history for this channel"""
    channel_id = str(interaction.channel_id)
    config = channel_configs.get(channel_id)
    
    clear_conversation(channel_id)
    
    # Reinitialize with system prompt if configured
    if config:
        initialize_conversation(channel_id, config["system_prompt"])
        await interaction.response.send_message("Conversation history cleared and reinitialized with system prompt.", ephemeral=True)
    else:
        await interaction.response.send_message("Conversation history cleared.", ephemeral=True)

@bot.tree.command(name="conversation_info")
@app_commands.describe()
async def conversation_info(interaction: discord.Interaction):
    """Show information about the current conversation"""
    channel_id = str(interaction.channel_id)
    conversation = get_conversation_history(channel_id)
    
    if not conversation:
        await interaction.response.send_message("No conversation history in this channel.", ephemeral=True)
        return
    
    msg_count = len(conversation)
    user_msgs = len([m for m in conversation if m["role"] == "user"])
    assistant_msgs = len([m for m in conversation if m["role"] == "assistant"])
    system_msgs = len([m for m in conversation if m["role"] == "system"])
    
    info = f"**Conversation Stats:**\n"
    info += f"Total messages: {msg_count}\n"
    info += f"User messages: {user_msgs}\n"
    info += f"Assistant messages: {assistant_msgs}\n"
    info += f"System messages: {system_msgs}\n\n"
    
    if system_msgs > 0:
        system_msg = next(m for m in conversation if m["role"] == "system")
        info += f"**Current System Prompt:**\n```{system_msg['content'][:500]}{'...' if len(system_msg['content']) > 500 else ''}```"
    
    await interaction.response.send_message(info, ephemeral=True)

@bot.tree.command(name="update_system_prompt")
@app_commands.describe(new_prompt="New system prompt for the assistant")
@app_commands.checks.has_permissions(administrator=True)
async def update_system_prompt(interaction: discord.Interaction, new_prompt: str):
    """Update the system prompt without changing other settings"""
    channel_id = str(interaction.channel_id)
    config = channel_configs.get(channel_id)
    
    if not config:
        await interaction.response.send_message("This channel is not configured. Use /setup first.", ephemeral=True)
        return
    
    # Update config
    config["system_prompt"] = new_prompt
    channel_configs[channel_id] = config
    save_configs(channel_configs)
    
    # Reinitialize conversation with new system prompt
    clear_conversation(channel_id)
    initialize_conversation(channel_id, new_prompt)
    
    await interaction.response.send_message(f"✅ System prompt updated and conversation reinitialized.", ephemeral=True)

# ---------- Message Listener ----------
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    channel_id = str(message.channel.id)
    config = channel_configs.get(channel_id)

    if not config or config["mode"] != "auto":
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
        # Check if conversation is initialized
        conversation = get_conversation_history(channel_id)
        if not conversation:
            initialize_conversation(channel_id, config["system_prompt"])

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

        response = await call_llm(
            config["backend"], 
            config["model"], 
            message.content,
            channel_id
        )
        
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