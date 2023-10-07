import discord
from discord.ext import commands
from transformers import pipeline

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

pipe = pipeline("image-classification", model="model")

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author.bot:
        return  # Ignore messages from bots

    # Check if the message has attachments
    if message.attachments:
        image_url = message.attachments[0].url
        result = analyze_image(image_url)

        if result[0]['label'] == 'no_safe':
            await message.delete()
            await message.channel.send(f"{message.author.mention}, detected adult content. The image has been deleted.")
            return

    await bot.process_commands(message)

def analyze_image(image_url):
    return pipe(image_url)

# bot.run("YOUR_BOT_TOKEN")

bot.run("API_KEY")
