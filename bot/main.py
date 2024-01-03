import asyncio
import os

import aiohttp
import pyrogram as pyro

bot = pyro.Client(
	"bot",
	in_memory=True,
	api_id=os.getenv("TG_API_ID"), api_hash=os.getenv("TG_API_HASH"),
	bot_token=os.getenv("TG_BOT_TOKEN"),
	workers=4
)


async def send_record_video_note(chat_id):
	while True:
		await bot.send_chat_action(chat_id, pyro.enums.ChatAction.RECORD_VIDEO_NOTE)
		await asyncio.sleep(1)


async def handle_command(_, message):
	print(message.text)
	user_input = " ".join(message.command[1:])

	if user_input == "":
		return

	asyncio.ensure_future(command(message, user_input))


async def command(message, user_input):
	personality = message.command[0]

	action_task = asyncio.create_task(send_record_video_note(message.chat.id))

	try:
		async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as h:
			async with h.post("http://engine:6767/generate", json={"person": personality, "text": user_input}) as result:
				res = await result.json()
				resp = res["result"]
				print(f"Result: {resp}")
				if os.path.exists(resp):
					await message.reply_video_note(resp)
				else:
					await message.reply(resp)
	finally:
		action_task.cancel()
		try:
			await action_task
		except asyncio.CancelledError:
			pass
		await bot.send_chat_action(message.chat.id, pyro.enums.ChatAction.CANCEL)


async def main():
	async with aiohttp.ClientSession() as h:
		async with h.get("http://engine:6767/persons") as result:
			persons = await result.json()

	@bot.on_message(pyro.filters.command(list(persons.keys())))
	async def _command(client, message):
		await handle_command(client, message)

	async with bot:
		await bot.set_bot_commands([pyro.types.BotCommand(p, persons[p]) for p in persons])
		print("started")
		await pyro.idle()


bot.run(main())
