# <img src="logo.png" style="width:40px; margin-right:10px"> Discord LLM Bot 



![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)
[![License](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)



A **Discord** bot that integrates multiple **LLM** backends (OpenAI, Anthropic, Mistral) with **xAPI** Learning Record Store (LRS) logging for educational analytics.

## Features

- **Multiple LLM Backends**: Support for OpenAI GPT models, Anthropic Claude, and Mistral via Hugging Face
- **Flexible Modes**: Manual (`/ask`) or automatic response modes
- **xAPI Logging**: All interactions are logged in a Learning Record Store using xAPI format
- **Per-Channel Configuration**: Different settings for each Discord channel (different backends, models, system prompts, and modes)

## Prerequisites

### API Keys Required
- **Discord Bot Token**: From [Discord Developer Portal](https://discord.com/developers/applications)
- **LLM API Key**: One of the following
    - **OpenAI API Key**: From [OpenAI Platform](https://platform.openai.com/api-keys) (optional)
    - **Anthropic API Key**: From [Anthropic Console](https://console.anthropic.com/) (optional)
    - **Hugging Face Token**: From [HF Settings](https://huggingface.co/settings/tokens) (optional)
- **LRS Credentials**: Learning Record Store endpoint and credentials (optional)

### Discord Bot Setup
1. Create a new application at [Discord Developer Portal](https://discord.com/developers/applications)
2. Go to "Bot" section and create a bot
3. Copy the bot token
4. Enable **Message Content Intent** under "Privileged Gateway Intents"
5. Generate invite link with permissions integer: `2147552256`

## Local Installation

### 1. Clone and Setup
```bash
git clone https://github.com/sonsoleslp/discord-ai-bot
cd discord-ai-bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Rename the `example.env` file to  `.env` in the project root and configure your API 


### 3. Run the Bot
```bash
python ai-bot.py
```

## Docker Installation

### 1. Build the Docker Image
```bash
docker build -t discord-ai-bot .
```

### 2. Run the bot
Rename the `example.env` file to  `.env` in the project root and configure your API. Then run with the environment file:
```bash
# Create data directory for persistent storage
mkdir -p data

# Run with environment file
docker run -d \
  --name discord-bot \
  --restart unless-stopped \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  discord-ai-bot
```

### 3. Docker Management Commands
```bash
# View logs
docker logs -f discord-bot

# Stop the bot
docker stop discord-bot

# Remove container
docker rm discord-bot

# Restart the bot
docker restart discord-bot
```

## Usage

### 1. Bot Setup
Invite the bot to your Discord server using the generated invite link with permission integer `2147552256` (read messages/view channels, send messages, read message history).

### 2. Configure a Channel
Use the `/setup` command in any channel:
```
/setup backend:openai model:gpt-4 system_prompt:"You are a team moderator. Only intervene if someone says something factually wrong or rude. Otherwise do not respond." mode:auto
```

**Parameters:**
- `backend`: `openai`, `anthropic`, or `mistral`
- `model`: Model name (e.g., `gpt-4`, `claude-3-sonnet-20240229`, `mistralai/Mistral-7B-Instruct-v0.1`)
- `system_prompt`: Instructions for the AI assistant
- `mode`: `ask` (manual with `/ask`) or `auto` (responds to all messages)

### 3. Interaction Modes

#### Manual Mode (`ask`)
```
/ask question:"What is machine learning?"
```

#### Automatic Mode (`auto`)
Simply type any message in the configured channel and the bot will respond automatically according to the system prompt entered.

## Model Examples

### OpenAI Models
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic Models
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Mistral Models (via Hugging Face)
- `mistralai/Mistral-7B-Instruct-v0.1`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

## xAPI Learning Analytics

The bot automatically logs all interactions to a Learning Record Store (LRS) in xAPI format, tracking:
- User messages (`sent` verb)
- Questions asked (`asked` verb)  
- Bot responses (`responded` verb)
- Channel and server context
- LLM configuration metadata

## Configuration Files

The bot creates `channel_configs.json` to store per-channel settings:
```json
{
    "123456789": {
        "backend": "openai",
        "mode": "auto",
        "model": "gpt-4",
        "system_prompt": "You are a helpful assistant"
    }
}
```

## Troubleshooting

### Bot Not Responding
1. Verify **Message Content Intent** is enabled in Discord Developer Portal
2. Check bot permissions in server/channel settings
3. Ensure environment variables are set correctly
4. Check console output for error messages

### LRS Logging Issues
- Verify LRS endpoint URL and credentials
- Check LRS supports xAPI version 1.0.3
- Monitor console for LRS error messages

### API Errors
- Verify API keys are valid and have sufficient credits/usage limits
- Check model names are correct for each backend
- Monitor rate limiting

## Development

### Adding New Backends
1. Create async function following the pattern of existing backends
2. Add to `LLM_BACKENDS` dictionary
3. Update setup command description

### Custom xAPI Extensions
Modify the `log_to_lrs` function to add custom xAPI extensions for your specific learning analytics needs.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
See the [LICENSE](./LICENSE) file for details.


## Support

For issues and questions:
- Check the troubleshooting section
- Review Discord bot permissions
- Verify API key validity
- Check console logs for detailed error messages

## Author

Developed by [Sonsoles López-Pernas](https://github.com/sonsoleslp)  
For questions, feel free to reach out or open an issue.