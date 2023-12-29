import uvloop
uvloop.install()

import os
import json
import asyncio
import requests
import aiohttp

import pyrogram as pyro

bot = pyro.Client(
  "bot",
  in_memory=True,
  api_id=os.getenv("TG_API_ID"), api_hash=os.getenv("TG_API_HASH"),
  bot_token=os.getenv("TG_BOT_TOKEN"),
  workers=4
)

_persons = requests.get("http://engine:6767/persons")
persons = json.loads(_persons.text)

@bot.on_message(pyro.filters.command(list(persons.keys())))
async def bot_command(client, message):
  print(message.text)
  user_input = " ".join(message.command[1:])
  personality = message.command[0]

  if user_input == "":
    return

  async def exe():
    async def record_video_note():
      while True:
        await bot.send_chat_action(message.chat.id, pyro.enums.ChatAction.RECORD_VIDEO_NOTE)
        await asyncio.sleep(1)

    action_task = asyncio.create_task(record_video_note())

    try:
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as h:
        async with h.post("http://engine:6767/generate", json={"person": personality, "text": user_input}) as result:
          res = json.loads(await result.text())
          resp = res["result"]
          print(f"Result: {resp}")
          try:
            if os.path.exists(resp):
              await message.reply_video_note(resp)
            else:
              await message.reply(resp)
          except:
            await message.reply(resp)
    finally:
      action_task.cancel()
      try:
        await action_task
      except asyncio.CancelledError:
        pass
      await bot.send_chat_action(message.chat.id, pyro.enums.ChatAction.CANCEL)

  asyncio.ensure_future(exe())

async def main():
  async with bot:
    await bot.set_bot_commands([pyro.types.BotCommand(p, persons[p]) for p in persons])
    print("started")
    await pyro.idle()

bot.run(main())
