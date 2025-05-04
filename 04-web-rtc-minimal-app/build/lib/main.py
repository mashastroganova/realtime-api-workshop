import asyncio
import fractions
import json
import os
from typing import Optional

import httpx
import numpy as np
import sounddevice as sd
from av.audio.frame import AudioFrame
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

"""A *minimal* client that streams microphone audio to the Azure OpenAI
GPT‑4o Realtime endpoint over WebRTC and plays the assistant’s reply back
over the speakers.

Changes vs. previous draft
-------------------------
* **Region handling** – the WebRTC URL is of the form
  `https://<region>.realtimeapi-preview.ai.azure.com/...` *not* the resource
  name.  The region is now taken from `AZURE_OPENAI_REGION` in `.env`.  If it
  isn’t defined we abort with a clear message instead of generating an
  invalid hostname that fails DNS (the “nodename nor servname provided”
  error the user saw).
* No other functional differences – audio and transcript logic remain the
  same.
"""

# -------------------------------------------------------------------------
#  Load environment variables (.env in project root)
# -------------------------------------------------------------------------
load_dotenv()

AZ_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]            # https://<resource>.openai.azure.com
AZ_DEPLOY   = os.environ["AZURE_OPENAI_DEPLOYMENT"]          # gpt-4o-mini-realtime-preview
AZ_KEY      = os.environ["AZURE_OPENAI_API_KEY"]

# WebRTC URL needs the *Azure region*, not the resource name ----------------
REGION: Optional[str] = os.getenv("AZURE_OPENAI_REGION")      # e.g. "swedencentral" or "eastus2"
if not REGION:
    raise RuntimeError(
        "Environment variable AZURE_OPENAI_REGION is required (e.g. 'swedencentral').\n"
        "The WebRTC endpoint is https://<region>.realtimeapi-preview.ai.azure.com/."
    )

# -------------------------------------------------------------------------
#  Helper class – microphone as an aiortc track
# -------------------------------------------------------------------------

class MicTrack(MediaStreamTrack):
    """Capture mono PCM from the system microphone and expose as 24‑kHz audio."""

    kind = "audio"

    def __init__(self, rate: int = 24_000):
        super().__init__()
        self.rate = rate
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=8)

        sd.default.samplerate = rate
        sd.default.channels = 1
        self._stream = sd.InputStream(
            dtype="int16",
            callback=self._callback,
            blocksize=480,  # 20 ms frame @ 24 kHz mono
        )
        self._stream.start()

    # sounddevice callback (runs in a separate thread)
    def _callback(self, indata, _frames, _time, _status):  # noqa: D401, N802
        try:
            self._queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            pass  # drop if consumer is too slow

    async def recv(self):  # noqa: D401 – aiortc pulls frames here
        pcm: np.ndarray = await self._queue.get()
        frame = AudioFrame(format="s16", layout="mono", samples=len(pcm))
        frame.planes[0].update(pcm.tobytes())
        frame.sample_rate = self.rate
        frame.time_base = fractions.Fraction(1, self.rate)
        return frame


# -------------------------------------------------------------------------
#  Main realtime‑API helper
# -------------------------------------------------------------------------

class AzureRealtimeClient:
    """Creates a session, performs the WebRTC handshake, and yields transcripts."""

    def __init__(self) -> None:
        self.pc = RTCPeerConnection()
        self._transcripts: asyncio.Queue[str] = asyncio.Queue()
        self._speaker: Optional[sd.OutputStream] = None

    # ------------------------- internal helpers --------------------------
    async def _create_session(self) -> None:
        url = f"{AZ_ENDPOINT}/openai/realtimeapi/sessions?api-version=2025-04-01-preview"
        body = {"model": AZ_DEPLOY, "voice": "alloy"}
        async with httpx.AsyncClient() as cli:
            res = await cli.post(url, json=body, headers={"api-key": AZ_KEY})
        res.raise_for_status()
        js = res.json()
        self._session_id: str = js["id"]
        self._bearer: str = js["client_secret"]["value"]  # 60‑second token

    # ---------------------------------------------------------------------
    async def connect(self):  # noqa: D401
        """Full connect sequence. Call once per chat session."""

        await self._create_session()

        # upstream microphone
        self.pc.addTrack(MicTrack())

        # downstream – play assistant audio via sounddevice
        @self.pc.on("track")
        async def _on_track(track):  # noqa: D401, N802
            if track.kind != "audio":
                return

            if self._speaker is None:
                self._speaker = sd.OutputStream(
                    samplerate=24_000, channels=1, dtype="int16", blocksize=480
                )
                self._speaker.start()

            async def _pump():  # pull RTP → PCM → speaker
                while True:
                    frame = await track.recv()
                    pcm = frame.to_ndarray().flatten().astype("int16")
                    self._speaker.write(pcm)

            asyncio.create_task(_pump())

        # DataChannel for transcripts
        dc = self.pc.createDataChannel("oai-ctrl")

        @dc.on("message")
        def _dc_msg(data):  # noqa: D401, N802
            try:
                msg = json.loads(data)
                if msg.get("type") == "text":
                    asyncio.create_task(self._transcripts.put(msg["text"]["value"]))
            except Exception:
                pass

        # SDP offer/answer handshake --------------------
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

    # ---------------------------------------------------------------------
    async def transcripts(self):  # noqa: D401 – async generator
        while True:
            yield await self._transcripts.get()

    # ---------------------------------------------------------------------
    async def close(self):  # noqa: D401
        if self._speaker:
            self._speaker.stop()
            self._speaker.close()
        await self.pc.close()
