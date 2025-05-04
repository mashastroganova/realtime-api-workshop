import asyncio
import fractions
import json
import os

import httpx
import numpy as np
import sounddevice as sd
from av.audio.frame import AudioFrame
from dotenv import load_dotenv
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

# ───────────────────────────────────────────────
#  Load environment variables (.env in project root)
# ───────────────────────────────────────────────
load_dotenv()

AZ_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZ_DEPLOY = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZ_KEY = os.environ["AZURE_OPENAI_API_KEY"]
REGION = AZ_ENDPOINT.split("https://")[1].split(".")[0]  # crude but works for <region>.openai.azure.com


# ───────────────────────────────────────────────
#  Helper class – microphone as an aiortc track
# ───────────────────────────────────────────────
class MicTrack(MediaStreamTrack):
    """Capture mono PCM from the system microphone and expose as 24‑kHz Opus-ready track."""

    kind = "audio"

    def __init__(self, rate: int = 24_000):
        super().__init__()
        self.rate = rate
        self.channels = 1
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=8)

        # sounddevice stream callback will push PCM blocks into the queue
        sd.default.samplerate = rate
        sd.default.channels = 1
        self._stream = sd.InputStream(
            dtype="int16",
            callback=self._callback,
            blocksize=480,  # 20 ms @ 24 kHz mono
        )
        self._stream.start()

    # sounddevice callback (runs in separate thread)
    def _callback(self, indata, _frames, _time, _status):  # noqa: D401, N803
        try:
            self._queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            pass  # drop if the consumer is too slow – better lag than RAM blow‑up

    # aiortc pulls frames via this coroutine
    async def recv(self):  # noqa: D401
        pcm: np.ndarray = await self._queue.get()
        frame = AudioFrame(format="s16", layout="mono", samples=len(pcm))
        frame.planes[0].update(pcm.tobytes())
        frame.sample_rate = self.rate
        frame.time_base = fractions.Fraction(1, self.rate)
        return frame


# ───────────────────────────────────────────────
#  Azure OpenAI Realtime – very small client
# ───────────────────────────────────────────────
class AzureRealtimeClient:
    """Handles session creation + WebRTC handshake; yields text transcripts."""

    def __init__(self) -> None:
        self.pc = RTCPeerConnection()
        self._transcripts: asyncio.Queue[str] = asyncio.Queue()
        self._speaker_stream: sd.OutputStream | None = None

    # ───────────────────────────────────
    async def _create_session(self) -> None:
        url = (
            f"{AZ_ENDPOINT}/openai/realtimeapi/sessions?api-version=2025-04-01-preview"
        )
        body = {"model": AZ_DEPLOY, "voice": "alloy"}
        async with httpx.AsyncClient() as cli:
            res = await cli.post(url, json=body, headers={"api-key": AZ_KEY})
        res.raise_for_status()
        js = res.json()
        self.session_id: str = js["id"]
        self._bearer: str = js["client_secret"]["value"]  # 60‑s token

    # ───────────────────────────────────
    async def connect(self):  # noqa: D401
        """Create session, build WebRTC peer connection, start mic + speaker."""

        await self._create_session()

        # add microphone track (upstream)
        self.pc.addTrack(MicTrack())

        # downstream – open speaker when first audio packet arrives
        @self.pc.on("track")
        async def _on_track(track):  # noqa: D401, N802
            if track.kind != "audio":
                return

            if self._speaker_stream is None:
                self._speaker_stream = sd.OutputStream(
                    samplerate=24_000, channels=1, dtype="int16", blocksize=480
                )
                self._speaker_stream.start()

            async def _play():  # pull frames → speaker
                while True:
                    frame = await track.recv()
                    pcm = frame.to_ndarray().flatten().astype("int16")
                    self._speaker_stream.write(pcm)

            asyncio.create_task(_play())

        # DataChannel for JSON events (transcripts, etc.)
        dc = self.pc.createDataChannel("oai-ctrl")

        @dc.on("message")
        def _on_message(data):  # noqa: D401, N802
            try:
                msg = json.loads(data)
                if msg.get("type") == "text":
                    text_val = msg["text"]["value"]
                    asyncio.create_task(self._transcripts.put(text_val))
            except Exception:
                pass

        # SDP offer
        await self.pc.setLocalDescription(await self.pc.createOffer())

        webrtc_url = (
            f"https://{REGION}.realtimeapi-preview.ai.azure.com/v1/realtimertc?model={AZ_DEPLOY}"
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

    # ───────────────────────────────────
    async def transcripts(self):  # noqa: D401
        """Async generator yielding assistant transcript strings."""
        while True:
            yield await self._transcripts.get()

    # ───────────────────────────────────
    async def close(self):  # noqa: D401
        if self._speaker_stream:
            self._speaker_stream.stop()
            self._speaker_stream.close()
        await self.pc.close()
