# 🤖 Lua Analysis Discord Bot

A Discord bot for analyzing and deobfuscating Lua scripts. Perfect for security researchers and game developers.

## 🚀 Features

- 🔍 **String Deobfuscation** - Decodes Base64, hex, and custom obfuscated strings
- 🕵️ **Pattern Scanning** - Detects suspicious patterns and calculates risk scores
- ⚙️ **Safe Execution** - Optional sandboxed execution with timeout protection
- 📊 **Detailed Reports** - Comprehensive analysis with risk assessment
- 💾 **Result Caching** - Stores results for 1 hour for easy sharing

## 📋 Commands

| Command | Description |
|---------|-------------|
| `!analyze` | Upload a Lua file for analysis |
| `!analyze --execute` | Include safe execution (⚠️ caution) |
| `!details <id>` | Get full analysis report |
| `!stats` | Show bot statistics |
| `!clean <hours>` | Clean old cache (admin only) |
| `!lua_help` | Show this help |

## 🛠️ Deployment on Railway

### 1. One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/yourusername/lua-discord-bot)

### 2. Manual Setup

```bash
# Clone repository
git clone https://github.com/yourusername/lua-discord-bot.git
cd lua-discord-bot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DISCORD_BOT_TOKEN="your_token_here"

# Run locally
python main.py
