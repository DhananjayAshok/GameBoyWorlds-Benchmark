"""
Client for the environment server. Standard library only, so it runs anywhere Python 3 does.

Configure it with env.json next to this file (or $ENVCLIENT_CONFIG):

    {"url": "http://127.0.0.1:8710", "token_file": "..."}

or with $ENV_SERVER_URL and $ENV_SERVER_TOKEN, which win over the file. During an evaluation $ENVCTL_SESSION names a
session already opened for you and $ENVCTL_DEADLINE is the unix time your wall-clock limit ends.

    from envclient import Client
    c = Client.from_config()
    s = c.open(task_id="gb.pokemon_red.004")
    out = s.step("A")        # out["obs"]["frame"] is PNG bytes; out["done"], out["success"], out["valid"]
    s.close()
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request


def seconds_left():
    """Seconds until your wall-clock limit ($ENVCTL_DEADLINE), or None when there is none."""
    deadline = os.environ.get("ENVCTL_DEADLINE")
    return None if not deadline else max(0.0, float(deadline) - time.time())


class EnvError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status


def decode(value):
    if isinstance(value, dict):
        if "__png__" in value:
            return base64.b64decode(value["__png__"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def to_array(png):
    """PNG bytes to a numpy array, if numpy and Pillow are installed."""
    import io
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(png)))


class Server:
    def __init__(self, url, token, timeout=900.0):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def request(self, method, path, body=None, raw=False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": "Bearer " + self.token,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", "")
            except Exception:
                message = str(e)
            raise EnvError(e.code, message) from None
        return out if raw else decode(out)


class Session:
    def __init__(self, server, opened):
        self.server = server
        self.id = opened["session"]
        self.opened = opened
        self.goal = opened.get("goal", "")
        self.max_steps = opened.get("max_steps")

    def _post(self, what, body=None, raw=False):
        return self.server.request("POST", "/sessions/%s/%s" % (self.id, what), body or {}, raw=raw)

    def step(self, action, raw=False):
        return self._post("step", {"action": action}, raw=raw)

    def reset(self, raw=False):
        return self._post("reset", raw=raw)

    def observe(self, raw=False):
        return self._post("observe", raw=raw)

    def save(self):
        return self._post("save")["handle"]

    def load(self, handle, raw=False):
        return self._post("load", {"handle": handle}, raw=raw)

    def describe(self):
        return self.server.request("GET", "/sessions/" + self.id)

    def close(self):
        self.server.request("DELETE", "/sessions/" + self.id)


class Client:
    def __init__(self, url, token):
        self.server = Server(url, token)

    @classmethod
    def from_config(cls, path=None):
        here = os.path.dirname(os.path.abspath(__file__))
        path = path or os.environ.get("ENVCLIENT_CONFIG") or os.path.join(here, "env.json")
        url = os.environ.get("ENV_SERVER_URL")
        token = os.environ.get("ENV_SERVER_TOKEN")
        if os.path.exists(path):
            with open(path) as f:
                config = json.load(f)
            url = url or config.get("url")
            if token is None and config.get("token_file"):
                with open(config["token_file"]) as f:
                    token = f.read().strip()
            token = token or config.get("token")
        if not url or not token:
            raise RuntimeError("no server url or token (looked at %s, $ENV_SERVER_URL, $ENV_SERVER_TOKEN)" % path)
        return cls(url=url, token=token)

    def tasks(self):
        return self.server.request("GET", "/tasks")["tasks"]

    def scenes(self):
        return self.server.request("GET", "/scenes")["scenes"]

    def open(self, task_id=None, game=None, init_state=None, state=False, raw=False):
        body = {"task_id": task_id} if task_id else {"game": game, "init_state": init_state}
        body["state"] = state
        return Session(self.server, self.server.request("POST", "/sessions", body, raw=raw))

    def attach(self, session_id):
        """A session someone else opened: the one the evaluation harness opened for you."""
        described = self.server.request("GET", "/sessions/" + session_id)
        return Session(self.server, dict(described, session=session_id))
