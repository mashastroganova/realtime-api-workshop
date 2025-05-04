"""A *minimal* Azure OpenAI Realtime client written in pure Python.

It captures microphone audio (16‑bit PCM @ 24 kHz), sends it to Azure over Web‑RTC,
plays the assistant’s audio reply through the system speakers, and exposes an
async generator that yields the assistant’s partial / final transcripts.

Dependencies
============
    pip install aiortc sounddevice numpy httpx python-dotenv av

Environment variables (.env)
===========================
    AZURE_OPENAI_ENDPOINT   = https://<resource-name>.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT = gpt-4o-mini-realtime-preview   # deployment *name*
    AZURE_OPENAI_API_KEY    = <your‑key>                     # or leave blank for Entra ID
    AZURE_OPENAI_REGION     = <azure‑region>                 # e.g. "swedencentral"

The code intentionally stays very close to the network – no extra abstractions –
so you can see every moving part.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator, Optional

import httpx
import numpy as np
import sounddevice as sd
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av.audio.frame import AudioFrame
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────────────
#  Required environment variables
# ───────────────────────────────────────────────
AZ_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZ_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZ_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
REGION = os.getenv("AZURE_OPENAI_REGION")

if not AZ_ENDPOINT or not AZ_DEPLOY or not REGION:
    raise RuntimeError(
        "Missing one of AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_REGION."
    )

# ───────────────────────────────────────────────
#  Microphone → aiortc track helper
# ───────────────────────────────────────────────
class MicTrack(MediaStreamTrack):
    """Capture mono 24 kHz PCM from the system microphone and expose as aiortc track."""

    kind = "audio"

    def __init__(self, rate: int = 24_000):
        super().__init__()
        self.rate = rate
        self.channels = 1
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=8)
        self._pts = 0  # presentation‑time‑stamp in samples

        sd.default.samplerate = rate
        sd.default.channels = 1

        # Start synchronous callback → asyncio queue pump
        self._stream = sd.InputStream(
            dtype="int16",
            callback=self._callback,
            blocksize=480,  # 20 ms @ 24 kHz mono
        )
        self._stream.start()

    # sounddevice callback (runs in a separate thread)
    def _callback(self, indata, _frames, _time, _status):  # noqa: D401, N802
        try:
            self._queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            pass  # Drop frame if the consumer lags – better stutter than OOM

    # aiortc pulls frames through this coroutine
    async def recv(self):
        pcm: np.ndarray = await self._queue.get()
        frame = AudioFrame(format="s16", layout="mono", samples=len(pcm))
        frame.planes[0].update(pcm.tobytes())
        frame.sample_rate = self.rate
        frame.pts = self._pts
        self._pts += len(pcm)
        return frame

    async def stop(self):  # type: ignore[override]
        await super().stop()
        self._stream.stop()
        self._stream.close()


# ───────────────────────────────────────────────
#  Azure Realtime client (single‑session)
# ───────────────────────────────────────────────
class AzureRealtimeClient:
    """Creates a realtime session, sends mic audio, plays assistant’s audio, yields transcripts."""

    def __init__(self) -> None:
        self.pc = RTCPeerConnection()
        self._transcripts: asyncio.Queue[str] = asyncio.Queue()
        self._mic_track: Optional[MicTrack] = None
        self._speaker_stream: Optional[sd.OutputStream] = None
        self.session_id: Optional[str] = None
        self._bearer: Optional[str] = None

    # ──────────────────────────────── internal helpers ────────────────────────────────
    async def _create_session(self) -> None:
        url = f"{AZ_ENDPOINT}/openai/realtimeapi/sessions?api-version=2025-04-01-preview"
        body = {"model": AZ_DEPLOY, "voice": "alloy"}
        async with httpx.AsyncClient() as cli:
            res = await cli.post(url, json=body, headers={"api-key": AZ_KEY} if AZ_KEY else None)
        res.raise_for_status()
        js = res.json()
        self.session_id = js["id"]
        # client_secret.value is a short‑lived (~60 s) token we use as a Bearer for the WebRTC offer
        self._bearer = js["client_secret"]["value"]

    # ──────────────────────────────── public API ────────────────────────────────
    async def connect(self) -> None:
        """Create session, open Web‑RTC connection, start mic capture + speaker playback."""

        await self._create_session()

        # Up‑stream audio
        self._mic_track = MicTrack()
        self.pc.addTrack(self._mic_track)

        # Down‑stream audio handler – runs once per incoming track
        @self.pc.on("track")
        async def _on_track(track):  # noqa: D401
            if track.kind != "audio":
                return

            speaker: Optional[sd.OutputStream] = None

            async def _play():  # pull PCM from aiortc track → sounddevice
                nonlocal speaker
                try:
                    while True:
                        frame = await track.recv()
                        if speaker is None:
                            speaker = sd.OutputStream(
                                samplerate=frame.sample_rate,
                                channels=len(frame.layout.channels),
                                dtype="float32",
                                blocksize=480,
                            )
                            speaker.start()
                        pcm = frame.to_ndarray()
                        if pcm.dtype == np.int16:
                            pcm = pcm.astype("float32") / 32768.0
                        pcm = pcm.reshape(-1, speaker.channels)
                        speaker.write(pcm)
                except MediaStreamError:
                    pass  # remote closed the track
                finally:
                    if speaker is not None:
                        speaker.stop()
                        speaker.close()

            asyncio.create_task(_play())

        # Data‑channel for transcripts – Azure creates it server‑side
        @self.pc.on("datachannel")
        def _on_datachannel(channel):  # noqa: D401
            @channel.on("message")
            def _on_message(data):  # noqa: D401, N802
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "text":
                        asyncio.create_task(self._transcripts.put(msg["text"]["value"]))
                except Exception:
                    pass

        # SDP offer / answer
        await self.pc.setLocalDescription(await self.pc.createOffer())

        webrtc_url = (
            f"https://{REGION}.realtimeapi-preview.ai.azure.com/"
            f"v1/realtimertc?model={AZ_DEPLOY}"
        )
        async with httpx.AsyncClient() as cli:
            ans = await cli.post(
                webrtc_url,
                content=self.pc.localDescription.sdp,
                headers={
                    "Authorization": f"Bearer {self._bearer}",
                    "Content-Type": "application/sdp",
                },
            )
        ans.raise_for_status()
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=ans.text, type="answer")
        )

    # The user‑facing async generator of transcript deltas / finals
    async def transcripts(self) -> AsyncGenerator[str, None]:  # noqa: D401
        while True:
            yield await self._transcripts.get()

    async def close(self) -> None:  # noqa: D401
        """Gracefully tear down the session and release audio devices."""
        if self._speaker_stream is not None:
            self._speaker_stream.stop()
            self._speaker_stream.close()
        if self._mic_track is not None:
            await self._mic_track.stop()
        await self.pc.close()
