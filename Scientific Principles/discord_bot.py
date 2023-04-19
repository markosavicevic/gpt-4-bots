import os
import discord
from discord.ext import commands
from retrieval_agent import conversational_agent

DISCORD_BOT_TOKEN="MTA5NzY0NjQ5MTA0MjA2MjM0Ng.G_VfS3.kiAOXk21ZdxX2BApFGP9md1KpsJWoaE6dwpUso" # Set your Discord bot token as an environment variable

intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True  # Add this line

bot = commands.Bot(command_prefix="", intents=intents)

def format_response(response_text):
    formatted_text = response_text.replace("```json", "```json\n").replace("```", "```\n")
    return formatted_text

@bot.command(name="")
async def ask(ctx, *, question):
    response = conversational_agent(question)
    formatted_response = format_response(response['output'])
    await ctx.send(formatted_response)

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
