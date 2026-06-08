"""
bridge_loader.py

Episode-level loader for BridgeData v2 in LeRobot v2.0 layout.

Data source
-----------
`IPEC-COMMUNITY/bridge_orig_lerobot` — the canonical LeRobot-v2 packaging of
BridgeData v2. Pixels are stored as one MP4 per episode per camera view; this
loader pulls them *lazily*, one episode at a time, via `hf_hub_download`
(content-addressed cache → no re-download across the K-sweep and seeds).

Only the single view `observation.images.image_0` is read — this matches the
preferred image key of the previous loader. The other three camera views are
deliberately NOT fanned in; doing so would change what the study compresses.

Public contract (unchanged from the previous version)
------------------------------------------------------
    from src.data.bridge_loader import BridgeDataLoader

    loader = BridgeDataLoader(root="~/.cache/hf_bridge")
    tasks  = loader.list_tasks(min_demos=20)   # sorted task strings
    eps    = loader.list_episodes(tasks[0])    # ascending episode indices
    n      = loader.num_episodes_for(tasks[0])
    ep     = loader.load_episode(eps[0])
    # ep["images"]    → np.ndarray (T, 256, 256, 3) uint8, RGB
    # ep["task_name"] → str
    # ep["n_frames"]  → int

Implementation notes
--------------------
* Index (`list_tasks` / `list_episodes` / `num_episodes_for`) is built from
  `meta/episodes.jsonl` + `meta/tasks.jsonl` — a few MB of metadata, no pixels.
* `load_episode` fetches exactly one MP4 (`observation.images.image_0`) and
  decodes it to RGB with PyAV (`to_ndarray(format="rgb24")`), so the contract's
  RGB channel order holds without a separate BGR→RGB conversion.
* Episode → file mapping is deterministic: it follows the dataset's own
  `video_path` template and `chunks_size` from `meta/info.json`.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_DATASET_REPO = "IPEC-COMMUNITY/bridge_orig_lerobot"
_VIDEO_KEY = "observation.images.image_0"  # single view — do NOT fan out to all 4
_DECODE_CACHE_SIZE = 16  # in-memory decoded-episode LRU (file cache is separate/unbounded)


class BridgeDataLoader:
    """Episode-level accessor for BridgeData v2 (LeRobot-v2 / IPEC mirror).

    Metadata (episode/task index) is fetched once on construction; each
    episode's frames are fetched and decoded on demand in `load_episode`.
    Both the metadata files and the per-episode MP4s land in the HuggingFace
    content-addressed cache, so repeated access across the K-sweep and the
    3 random seeds re-reads from disk rather than re-downloading.

    Args:
        root:         Local HuggingFace cache directory. Defaults to the
                      HuggingFace default (~/.cache/huggingface).
        dataset_name: HuggingFace repo ID
                      (default: "IPEC-COMMUNITY/bridge_orig_lerobot").
    """

    def __init__(
        self,
        root: Optional[str] = None,
        dataset_name: str = _DATASET_REPO,
    ) -> None:
        self._repo = dataset_name
        self._root = Path(root).expanduser() if root else None
        self._cache_dir = str(self._root) if self._root is not None else None

        # ---- fetch metadata only (no pixels) --------------------------------
        info = self._read_json(self._fetch_meta("meta/info.json"))
        self._video_path_tmpl: str = info["video_path"]
        self._chunks_size: int = int(info.get("chunks_size", 1000))

        episodes = self._read_jsonl(self._fetch_meta("meta/episodes.jsonl"))
        tasks = self._read_jsonl(self._fetch_meta("meta/tasks.jsonl"))

        # Canonical task strings (tasks.jsonl) — used to validate grouping keys.
        self._valid_tasks = {row["task"] for row in tasks if "task" in row}

        # task string → [episode_index, ...] in ascending episode order, and
        # episode_index → (task string, length). episodes.jsonl already carries
        # the resolved task string in `tasks`, so no task_index lookup is needed.
        self._task_to_episodes: Dict[str, List[int]] = {}
        self._episode_meta: Dict[int, Dict] = {}

        for row in sorted(episodes, key=lambda r: int(r["episode_index"])):
            ep_idx = int(row["episode_index"])
            ep_tasks = row.get("tasks") or []
            task = str(ep_tasks[0]) if ep_tasks else ""
            length = int(row.get("length", 0))

            self._episode_meta[ep_idx] = {"task": task, "length": length}
            if task:
                self._task_to_episodes.setdefault(task, []).append(ep_idx)

        # Bounded in-memory decode cache (file cache lives in the HF cache dir).
        self._decode_cache: "OrderedDict[int, dict]" = OrderedDict()

    # ------------------------------------------------------------------
    # Public API (byte-identical signatures to the previous version)
    # ------------------------------------------------------------------

    def list_tasks(self, min_demos: int = 20) -> List[str]:
        """Task strings that have at least *min_demos* episodes."""
        return sorted(
            t for t, eps in self._task_to_episodes.items() if len(eps) >= min_demos
        )

    def list_episodes(self, task: str) -> List[int]:
        """Episode indices that belong to *task* (ascending)."""
        return list(self._task_to_episodes.get(task, []))

    def num_episodes_for(self, task: str) -> int:
        return len(self._task_to_episodes.get(task, []))

    def load_episode(self, episode_index: int) -> dict:
        """Load all frames for *episode_index*.

        Returns:
            dict with:
              "images"    — np.ndarray (T, H, W, 3) uint8, RGB
              "task_name" — str
              "n_frames"  — int
        """
        cached = self._decode_cache.get(episode_index)
        if cached is not None:
            self._decode_cache.move_to_end(episode_index)
            return cached

        meta = self._episode_meta[episode_index]
        task_name = meta["task"]

        mp4_path = self._fetch_episode_video(episode_index)
        images = self._decode_rgb(mp4_path)

        result = {
            "images":    images,
            "task_name": task_name,
            "n_frames":  int(images.shape[0]),
        }

        self._decode_cache[episode_index] = result
        if len(self._decode_cache) > _DECODE_CACHE_SIZE:
            self._decode_cache.popitem(last=False)
        return result

    # ------------------------------------------------------------------
    # Fetch helpers (lazy hf_hub_download → content-addressed cache)
    # ------------------------------------------------------------------

    def _fetch_meta(self, filename: str) -> str:
        return self._hf_download(filename)

    def _fetch_episode_video(self, episode_index: int) -> str:
        """Resolve and download the single-view MP4 for *episode_index*."""
        chunk = episode_index // self._chunks_size
        rel = self._video_path_tmpl.format(
            episode_chunk=chunk,
            video_key=_VIDEO_KEY,
            episode_index=episode_index,
        )
        return self._hf_download(rel)

    def _hf_download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        return hf_hub_download(
            repo_id=self._repo,
            filename=filename,
            repo_type="dataset",
            cache_dir=self._cache_dir,
        )

    # ------------------------------------------------------------------
    # Decode (PyAV → RGB, no BGR round-trip)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_rgb(mp4_path: str) -> np.ndarray:
        """Decode an MP4 to (T, H, W, 3) uint8 RGB.

        Uses PyAV's `to_ndarray(format="rgb24")`, which yields RGB directly —
        the loader never produces BGR, so downstream extractors (optical flow,
        attention) and the retrieval embedder receive the channel order they
        assume.
        """
        import av  # noqa: PLC0415

        frames: List[np.ndarray] = []
        with av.open(mp4_path) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            raise RuntimeError(f"No frames decoded from {mp4_path}")

        return np.stack(frames, axis=0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Small parse helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: str) -> List[dict]:
        rows: List[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
