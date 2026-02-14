# main.py - Complete Discord bot for Lua analysis
import discord
from discord.ext import commands
import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
import io
import logging
import signal
import sys
import gc
import tracemalloc
from typing import Optional, Dict, Any
import aiohttp
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
PREFIX = os.getenv('BOT_PREFIX', '!')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '1048576'))  # 1MB default
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '50'))
CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '3600'))  # 1 hour
MAX_EXECUTION_TIME = int(os.getenv('MAX_EXECUTION_TIME', '5'))  # 5 seconds
ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Set log level
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))

# Import your modules
try:
    from deobfuscator_core import Deobfuscator
    from pattern_scanner import PatternScanner
    from execution_engine import ExecutionEngine
    logger.info("✅ Successfully imported analysis modules")
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    logger.error("Make sure all required files are present:")
    logger.error("  - deobfuscator_core.py")
    logger.error("  - pattern_scanner.py")
    logger.error("  - execution_engine.py")
    sys.exit(1)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents, help_command=None)

# Analysis cache
analysis_cache: Dict[str, Dict[str, Any]] = {}
cache_timestamps: Dict[str, datetime] = {}

# Metrics
bot_stats = {
    'total_analyses': 0,
    'total_files_processed': 0,
    'total_errors': 0,
    'start_time': datetime.now(),
    'cache_hits': 0,
    'cache_misses': 0
}

class RateLimiter:
    """Simple rate limiter for commands"""
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def can_call(self) -> bool:
        now = datetime.now()
        self.calls = [call for call in self.calls if now - call < timedelta(seconds=self.period)]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

# Rate limiters per user
rate_limiters: Dict[int, RateLimiter] = {}

def get_rate_limiter(user_id: int) -> RateLimiter:
    """Get or create rate limiter for user"""
    if user_id not in rate_limiters:
        rate_limiters[user_id] = RateLimiter(max_calls=5, period=60)  # 5 calls per minute
    return rate_limiters[user_id]

class LuaAnalyzer:
    """Main analysis class"""
    def __init__(self):
        self.deobfuscator = Deobfuscator()
        self.pattern_scanner = PatternScanner()
        self.execution_engine = ExecutionEngine(max_time=MAX_EXECUTION_TIME)
        self.lock = asyncio.Lock()
    
    async def analyze_file(self, file_content: str, filename: str, modes: list) -> Dict[str, Any]:
        """Analyze a Lua file"""
        async with self.lock:  # Prevent concurrent access
            temp_path = None
            try:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w', 
                    suffix='.lua', 
                    delete=False, 
                    encoding='utf-8'
                ) as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                results = {
                    'filename': filename,
                    'timestamp': datetime.now().isoformat(),
                    'modes_used': modes,
                    'analysis': {},
                    'file_size': len(file_content)
                }
                
                # Run analyses in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                if "strings" in modes:
                    results['analysis']['strings'] = await loop.run_in_executor(
                        None, self.deobfuscator.analyze_script, temp_path
                    )
                
                if "patterns" in modes:
                    self.pattern_scanner.load_default_patterns()
                    results['analysis']['patterns'] = await loop.run_in_executor(
                        None, self.pattern_scanner.analyze_target_file, temp_path
                    )
                
                if "execute" in modes and "--execute" in modes:
                    results['analysis']['execution'] = await loop.run_in_executor(
                        None, self.execution_engine.process_script_file, temp_path
                    )
                
                bot_stats['total_analyses'] += 1
                bot_stats['total_files_processed'] += 1
                
                return results
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                bot_stats['total_errors'] += 1
                raise
            
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Failed to delete temp file: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.execution_engine.execution_log.clear()
        gc.collect()

# Initialize analyzer
analyzer = LuaAnalyzer()

@bot.event
async def on_ready():
    """Bot ready event"""
    logger.info(f'✅ {bot.user} has connected to Discord!')
    logger.info(f'📊 Bot is in {len(bot.guilds)} guilds')
    logger.info(f'👥 Total users: {len(set(bot.get_all_members()))}')
    
    # Set custom status
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name=f"{PREFIX}lua_help | Lua Analysis"
        )
    )
    
    # Log bot info
    logger.info(f"🤖 Bot ID: {bot.user.id}")
    logger.info(f"📝 Prefix: {PREFIX}")
    logger.info(f"⚙️ Max file size: {MAX_FILE_SIZE/1024/1024:.1f}MB")

@bot.event
async def on_command_error(ctx, error):
    """Global error handler"""
    if isinstance(error, commands.CommandNotFound):
        return
    
    logger.error(f"Command error in {ctx.command}: {error}")
    bot_stats['total_errors'] += 1
    
    error_messages = {
        commands.MissingPermissions: "You don't have permission to use this command.",
        commands.BotMissingPermissions: "I don't have permission to do that.",
        commands.NotOwner: "This command is owner only.",
        commands.CommandOnCooldown: f"Command on cooldown. Try again in {error.retry_after:.1f}s",
        commands.MaxConcurrencyReached: "Too many people using this command. Try again later.",
    }
    
    message = error_messages.get(type(error), f"An error occurred: {str(error)}")
    await ctx.send(f"❌ {message}")

@bot.command(name='analyze')
async def analyze_lua(ctx, *, args: str = ""):
    """Analyze a Lua script"""
    # Rate limiting
    limiter = get_rate_limiter(ctx.author.id)
    if not limiter.can_call():
        await ctx.send("⏰ Rate limit exceeded. Please wait a minute before trying again.")
        return
    
    if not ctx.message.attachments:
        await ctx.send("❌ Please attach a Lua file to analyze!\nUsage: `!analyze [--execute]` with file attachment")
        return
    
    attachment = ctx.message.attachments[0]
    
    # Check file size
    if attachment.size > MAX_FILE_SIZE:
        size_mb = attachment.size / 1024 / 1024
        max_mb = MAX_FILE_SIZE / 1024 / 1024
        await ctx.send(f"❌ File too large! ({size_mb:.1f}MB > {max_mb:.1f}MB)")
        return
    
    # Check file type
    if not attachment.filename.endswith(('.lua', '.luac', '.txt')):
        await ctx.send("⚠️ Warning: File doesn't have .lua extension. Analysis may not work properly.")
    
    # Parse modes
    modes = ["strings", "patterns"]
    args_lower = args.lower()
    
    if '--execute' in args_lower:
        modes.append("execute")
        warning_msg = await ctx.send("⚠️ **WARNING**: Execution mode enabled! This will run the Lua code in a sandbox.")
        await asyncio.sleep(2)
        await warning_msg.delete()
    
    # Send initial message
    analyzing_msg = await ctx.send(f"🔄 Analyzing {attachment.filename}... This may take a moment.")
    
    try:
        # Download file
        file_content = (await attachment.read()).decode('utf-8', errors='ignore')
        
        # Analyze
        results = await analyzer.analyze_file(file_content, attachment.filename, modes)
        
        # Cache results
        analysis_id = f"{ctx.author.id}_{int(datetime.now().timestamp())}"
        analysis_cache[analysis_id] = results
        cache_timestamps[analysis_id] = datetime.now()
        
        # Clean old cache
        await clean_cache()
        
        # Create summary embed
        embed = discord.Embed(
            title="✅ Analysis Complete",
            color=discord.Color.green(),
            timestamp=datetime.now()
        )
        
        embed.add_field(name="📄 File", value=attachment.filename, inline=True)
        embed.add_field(name="📏 Size", value=f"{attachment.size/1024:.1f}KB", inline=True)
        embed.add_field(name="🔧 Modes", value=', '.join(modes), inline=True)
        
        # Add pattern analysis summary
        if 'patterns' in results['analysis']:
            pattern_data = results['analysis']['patterns']
            if 'risk_assessment' in pattern_data:
                risk = pattern_data['risk_assessment']
                score = pattern_data.get('total_score_value', 0)
                
                # Color code risk level
                risk_color = {
                    "High": "🔴",
                    "Medium": "🟡",
                    "Low": "🟢",
                    "Minimal": "⚪"
                }.get(risk, "⚪")
                
                embed.add_field(
                    name="⚠️ Risk Level", 
                    value=f"{risk_color} {risk} (Score: {score})", 
                    inline=True
                )
        
        # Add string analysis summary
        if 'strings' in results['analysis']:
            string_data = results['analysis']['strings']
            string_count = len(string_data.get('decrypted_strings', []))
            tables = string_data.get('data_tables_found', 0)
            embed.add_field(
                name="📝 Strings", 
                value=f"Decrypted: {string_count}\nTables: {tables}", 
                inline=True
            )
        
        # Add execution summary if available
        if 'execution' in results['analysis']:
            exec_data = results['analysis']['execution']
            if 'execution_details' in exec_data:
                details = exec_data['execution_details']
                success = "✅ Success" if details.get('successful') else "❌ Failed"
                duration = f"{details.get('duration', 0):.3f}s"
                embed.add_field(
                    name="⚙️ Execution", 
                    value=f"Status: {success}\nTime: {duration}", 
                    inline=True
                )
        
        # Add footer with ID
        embed.set_footer(text=f"ID: {analysis_id[:8]}... | Use !details <id> for full report")
        
        await analyzing_msg.delete()
        await ctx.send(embed=embed)
        
        # Show quick preview of interesting findings
        if 'patterns' in results['analysis']:
            pattern_data = results['analysis']['patterns']
            if pattern_data.get('total_score_value', 0) > 50:
                await ctx.send("🔴 **High Risk**: Script shows strong obfuscation patterns!")
        
    except Exception as e:
        logger.error(f"Error in analyze command: {e}", exc_info=True)
        await analyzing_msg.delete()
        await ctx.send(f"❌ Error analyzing file: {str(e)[:200]}")

@bot.command(name='details')
async def get_details(ctx, analysis_id: str):
    """Get detailed analysis results"""
    # Find matching cache entry
    full_id = None
    for cached_id in analysis_cache:
        if cached_id.startswith(analysis_id) or cached_id.endswith(analysis_id):
            full_id = cached_id
            bot_stats['cache_hits'] += 1
            break
    
    if not full_id:
        bot_stats['cache_misses'] += 1
        await ctx.send("❌ Analysis ID not found! It may have expired (cache lasts 1 hour).")
        return
    
    results = analysis_cache[full_id]
    
    # Create detailed report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(f"🔍 LUA SCRIPT ANALYSIS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"File: {results['filename']}")
    report_lines.append(f"Time: {results['timestamp']}")
    report_lines.append(f"Modes: {', '.join(results['modes_used'])}")
    report_lines.append(f"Size: {results.get('file_size', 0)} bytes")
    report_lines.append("=" * 60)
    
    # Strings section
    if 'strings' in results['analysis']:
        report_lines.append("\n📝 STRING ANALYSIS")
        report_lines.append("-" * 40)
        string_data = results['analysis']['strings']
        report_lines.append(f"Data Tables Found: {string_data.get('data_tables_found', 0)}")
        report_lines.append(f"Cipher Mapping: {string_data.get('cipher_mapping_size', 0)} entries")
        report_lines.append(f"Encryption Functions: {string_data.get('encryption_functions', 0)}")
        
        decrypted = string_data.get('decrypted_strings', [])
        if decrypted:
            report_lines.append(f"\nDecrypted Strings ({len(decrypted)} total):")
            for i, s in enumerate(decrypted[:15]):  # Show first 15
                if s and len(s.strip()) > 0:
                    preview = s[:80] + "..." if len(s) > 80 else s
                    report_lines.append(f"  [{i}] {preview}")
            if len(decrypted) > 15:
                report_lines.append(f"  ... and {len(decrypted) - 15} more")
    
    # Patterns section
    if 'patterns' in results['analysis']:
        report_lines.append("\n🔍 PATTERN ANALYSIS")
        report_lines.append("-" * 40)
        pattern_data = results['analysis']['patterns']
        
        if 'detection_data' in pattern_data:
            report_lines.append("Detected Patterns:")
            for pattern, data in pattern_data['detection_data'].items():
                report_lines.append(f"  • {pattern}: {data['match_count']} matches (score: {data['total_score']})")
        
        report_lines.append(f"\nTotal Score: {pattern_data.get('total_score_value', 0)}")
        report_lines.append(f"Risk Level: {pattern_data.get('risk_assessment', 'Unknown')}")
    
    # Execution section
    if 'execution' in results['analysis']:
        report_lines.append("\n⚙️ EXECUTION ANALYSIS")
        report_lines.append("-" * 40)
        exec_data = results['analysis']['execution']
        
        if 'execution_details' in exec_data:
            details = exec_data['execution_details']
            report_lines.append(f"Success: {details.get('successful', False)}")
            report_lines.append(f"Duration: {details.get('duration', 0):.3f}s")
            report_lines.append(f"Exit Code: {details.get('exit_code', -1)}")
            
            if details.get('output_text'):
                output = details['output_text'][:300]
                report_lines.append(f"\nOutput Preview:\n{output}")
                if len(details['output_text']) > 300:
                    report_lines.append("... (truncated)")
            
            if details.get('error_text'):
                error = details['error_text'][:200]
                report_lines.append(f"\nErrors:\n{error}")
    
    # Join report
    report = '\n'.join(report_lines)
    
    # Send report
    if len(report) <= 1900:
        await ctx.send(f"```\n{report}\n```")
    else:
        # Send as file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(report)
            temp_path = f.name
        
        try:
            await ctx.send(
                content=f"📊 Full analysis report for {results['filename']}:",
                file=discord.File(temp_path, filename=f"analysis_{analysis_id}.txt")
            )
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

@bot.command(name='stats')
async def show_stats(ctx):
    """Show bot statistics"""
    uptime = datetime.now() - bot_stats['start_time']
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    embed = discord.Embed(
        title="📊 Bot Statistics",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    embed.add_field(name="⏰ Uptime", value=f"{hours}h {minutes}m {seconds}s", inline=True)
    embed.add_field(name="📈 Analyses", value=bot_stats['total_analyses'], inline=True)
    embed.add_field(name="📁 Files", value=bot_stats['total_files_processed'], inline=True)
    embed.add_field(name="❌ Errors", value=bot_stats['total_errors'], inline=True)
    embed.add_field(name="💾 Cache Size", value=f"{len(analysis_cache)}/{MAX_CACHE_SIZE}", inline=True)
    embed.add_field(name="🎯 Cache Hits", value=f"{bot_stats['cache_hits']}", inline=True)
    
    embed.set_footer(text=f"Bot Version: 1.0.0 | Python: {sys.version.split()[0]}")
    
    await ctx.send(embed=embed)

@bot.command(name='clean')
@commands.has_permissions(administrator=True)
async def manual_clean(ctx, hours: float = 1.0):
    """Manually clean old cache entries"""
    current_time = datetime.now()
    cutoff = current_time - timedelta(hours=hours)
    
    old_items = [
        aid for aid, timestamp in cache_timestamps.items() 
        if timestamp < cutoff
    ]
    
    for aid in old_items:
        analysis_cache.pop(aid, None)
        cache_timestamps.pop(aid, None)
    
    # Force garbage collection
    gc.collect()
    
    await ctx.send(f"✅ Removed {len(old_items)} old analysis results")

@bot.command(name='lua_help')
async def lua_help(ctx):
    """Show help information"""
    embed = discord.Embed(
        title="🤖 Lua Analysis Bot Commands",
        description="Analyze Lua scripts for obfuscation and malicious patterns",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name=f"{PREFIX}analyze",
        value="📤 Upload a .lua file with this command to analyze it\nExample: `!analyze` (with file attachment)",
        inline=False
    )
    
    embed.add_field(
        name=f"{PREFIX}analyze --execute",
        value="⚙️ Include safe execution in sandbox (⚠️ use with caution)",
        inline=False
    )
    
    embed.add_field(
        name=f"{PREFIX}details <id>",
        value="📋 Get detailed analysis using the ID from !analyze\nExample: `!details a1b2c3`",
        inline=False
    )
    
    embed.add_field(
        name=f"{PREFIX}stats",
        value="📊 Show bot statistics",
        inline=False
    )
    
    embed.add_field(
        name=f"{PREFIX}clean <hours>",
        value="🧹 Clean old cache (admin only)\nExample: `!clean 2`",
        inline=False
    )
    
    embed.add_field(
        name=f"{PREFIX}lua_help",
        value="❓ Show this help message",
        inline=False
    )
    
    embed.set_footer(
        text=f"Max file size: {MAX_FILE_SIZE/1024/1024:.0f}MB | Results cached for {CACHE_TIMEOUT/3600:.0f}h"
    )
    
    await ctx.send(embed=embed)

async def clean_cache():
    """Remove old cache entries"""
    current_time = datetime.now()
    expired = []
    
    for cache_id, timestamp in cache_timestamps.items():
        if current_time - timestamp > timedelta(seconds=CACHE_TIMEOUT):
            expired.append(cache_id)
    
    for cache_id in expired:
        analysis_cache.pop(cache_id, None)
        cache_timestamps.pop(cache_id, None)
    
    # Limit cache size
    if len(analysis_cache) > MAX_CACHE_SIZE:
        sorted_ids = sorted(cache_timestamps.items(), key=lambda x: x[1])
        to_remove = len(analysis_cache) - MAX_CACHE_SIZE
        for i in range(to_remove):
            cache_id = sorted_ids[i][0]
            analysis_cache.pop(cache_id, None)
            cache_timestamps.pop(cache_id, None)
    
    # Periodic cleanup of analyzer resources
    analyzer.cleanup()

async def periodic_cache_cleanup():
    """Periodically clean cache"""
    while not bot.is_closed():
        await asyncio.sleep(300)  # Run every 5 minutes
        await clean_cache()
        logger.debug(f"Cache cleanup completed. Current size: {len(analysis_cache)}")

def signal_handler(sig, frame):
    """Handle shutdown gracefully"""
    logger.info("🛑 Shutting down bot...")
    analyzer.cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Validate token
    if not TOKEN:
        logger.error("❌ No Discord bot token found!")
        logger.error("Please set DISCORD_BOT_TOKEN environment variable")
        logger.error("Example: export DISCORD_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    
    logger.info("🚀 Starting Discord bot...")
    logger.info(f"📝 Command prefix: {PREFIX}")
    logger.info(f"📦 Max cache size: {MAX_CACHE_SIZE}")
    logger.info(f"⏱️  Cache timeout: {CACHE_TIMEOUT}s")
    
    # Start background tasks
    asyncio.create_task(periodic_cache_cleanup())
    
    try:
        bot.run(TOKEN, log_handler=None)  # Let our logging handle it
    except discord.LoginFailure:
        logger.error("❌ Failed to login. Check your bot token!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")
        sys.exit(1)
