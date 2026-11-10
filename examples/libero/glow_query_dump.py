"""Per-episode record of what a LIBERO client sent a policy server and what came back (Python 3.8).

`QueryDump.add` keeps one query: both camera images and the state as sent, every other request
field (prompt, plan fields) as JSON, and the response's `glow_done` when there is one.
`QueryDump.save` writes the episode to `<dump_dir>/task<task>_ep<episode>.npz` (compressed) with
its plan trace line, and starts the next episode empty. scripts/glow_monitor_replay.py reads
these files back.

Arrays in the file: image, wrist_image [Q, H, W, 3] uint8; state [Q, S] float32; frame [Q] int64
(the client's policy step at each query); done [Q, K] float32 (NaN where the response had none);
request [Q] JSON strings; and the scalars task, episode, prompt, trace.
"""

import json
import pathlib

import numpy as np

IMAGE_KEY = "observation/image"
WRIST_KEY = "observation/wrist_image"
STATE_KEY = "observation/state"
ARRAY_KEYS = (IMAGE_KEY, WRIST_KEY, STATE_KEY)


def _plain(value):
    """JSON form of a request field (numpy arrays and scalars as lists and numbers)."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class QueryDump:
    def __init__(self, dump_dir):
        self.dir = pathlib.Path(dump_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._reset()

    def _reset(self):
        self.images, self.wrists, self.states, self.frames, self.dones, self.requests = [], [], [], [], [], []

    def __len__(self):
        return len(self.frames)

    def add(self, frame, element, response):
        """One query: `element` as sent to the server, `response` as it came back."""
        self.images.append(np.asarray(element[IMAGE_KEY], dtype=np.uint8))
        self.wrists.append(np.asarray(element[WRIST_KEY], dtype=np.uint8))
        self.states.append(np.asarray(element[STATE_KEY], dtype=np.float32))
        self.frames.append(int(frame))
        rest = {k: _plain(v) for k, v in element.items() if k not in ARRAY_KEYS}
        self.requests.append(json.dumps(rest, sort_keys=True))
        done = response.get("glow_done")
        self.dones.append(None if done is None else np.asarray(done, dtype=np.float32).reshape(-1))

    def save(self, task, episode, prompt, trace=""):
        """Writes the episode's queries (if any) and clears them; returns the path or None."""
        if not self.frames:
            self._reset()
            return None
        width = max((len(d) for d in self.dones if d is not None), default=0)
        done = np.full((len(self.dones), width), np.nan, dtype=np.float32)
        for i, d in enumerate(self.dones):
            if d is not None:
                done[i, : len(d)] = d
        path = self.dir / f"task{int(task)}_ep{int(episode)}.npz"
        np.savez_compressed(
            str(path),
            image=np.stack(self.images),
            wrist_image=np.stack(self.wrists),
            state=np.stack(self.states),
            frame=np.asarray(self.frames, dtype=np.int64),
            done=done,
            request=np.asarray(self.requests),
            task=np.int64(task),
            episode=np.int64(episode),
            prompt=np.asarray(str(prompt)),
            trace=np.asarray(str(trace)),
        )
        self._reset()
        return path


def load(path):
    """A saved episode as a dict (request decoded to dicts, scalars as Python values)."""
    with np.load(str(path), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["request"] = [json.loads(s) for s in out["request"].tolist()]
    for k in ("task", "episode"):
        out[k] = int(out[k])
    for k in ("prompt", "trace"):
        out[k] = str(out[k])
    return out
