import asyncio
from typing import Optional

import chainlit as cl
from dotenv import load_dotenv

from realtime_client import AzureRealtimeClient

# -----------------------------------------------------------------------------
#  Load .env early so realtime_client picks up AZURE_* variables
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
#  Helper – relay transcripts coming from Azure back into Chainlit chat bubbles
# -----------------------------------------------------------------------------
async def _relay_transcripts(client: AzureRealtimeClient):
    async for text in client.transcripts():
        await cl.Message(content=text).send()

# -----------------------------------------------------------------------------
#  Utility – central place to start / stop the WebRTC client
# -----------------------------------------------------------------------------
async def _toggle_session(cmd: str):
    """cmd must be either "start" or "stop"."""

    if cmd == "start":
        if cl.user_session.get("rtc") is not None:
            await cl.Message("The microphone is already live.").send()
            return

        client = AzureRealtimeClient()
        await client.connect()
        cl.user_session.set("rtc", client)
        asyncio.create_task(_relay_transcripts(client))

        await cl.Message(
            "🟢 Microphone is live. Speak now – the assistant will reply through your speakers.",
            actions=[
                cl.Action(
                    name="stop_talking",
                    label="🛑 Stop talking",
                    payload={"cmd": "stop"},
                )
            ],
        ).send()

    else:  # stop ----------
        client: Optional[AzureRealtimeClient] = cl.user_session.get("rtc")
        if client is not None:
            await client.close()
            cl.user_session.set("rtc", None)

        await cl.Message(
            "🔴 Microphone closed.",
            actions=[
                cl.Action(
                    name="start_talking",
                    label="🎤 Start talking",
                    payload={"cmd": "start"},
                )
            ],
        ).send()

# -----------------------------------------------------------------------------
#  Chat lifecycle
# -----------------------------------------------------------------------------

@cl.on_chat_start
async def chat_start():
    await cl.Message(
        "🎙️ Welcome! Click **Start talking** to open your microphone and begin a realtime voice chat with GPT‑4o.",
        actions=[
            cl.Action(
                name="start_talking",
                label="🎤 Start talking",
                payload={"cmd": "start"},
            )
        ],
    ).send()

# -----------------------------------------------------------------------------
#  Button callbacks – one for each action name (Chainlit ≥1.0 requires explicit names)
# -----------------------------------------------------------------------------

@cl.action_callback("start_talking")
async def cb_start(action: cl.Action):
    await _toggle_session("start")


@cl.action_callback("stop_talking")
async def cb_stop(action: cl.Action):
    await _toggle_session("stop")
