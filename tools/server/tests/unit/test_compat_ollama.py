#!/usr/bin/env python3
import json

import pytest
import requests

from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2"
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = 512


def ollama_stream_request(path: str, data: dict) -> list[dict]:
    """POST a request and parse the response as newline-delimited JSON"""
    url = server.make_url(path)
    response = requests.post(url, json=data, stream=True)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        chunks.append(json.loads(line.decode("utf-8")))
    return chunks


# /api/chat

def test_ollama_chat_basic():
    server.start()

    res = server.make_request("POST", "/api/chat", data={
        "model": "tinyllama-2",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Say hello"},
        ],
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    body = res.body
    assert body["model"] == "tinyllama-2"
    assert isinstance(body["created_at"], str) and body["created_at"].endswith("Z")
    assert body["done"] is True
    assert "done_reason" in body
    assert body["message"]["role"] == "assistant"
    assert isinstance(body["message"]["content"], str)
    assert len(body["message"]["content"]) > 0
    assert body["eval_count"] > 0
    for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_duration"):
        assert key in body, f"Missing '{key}'"


def test_ollama_chat_stream():
    server.start()

    # Ollama streams unless "stream" is explicitly false
    chunks = ollama_stream_request("/api/chat", {
        "model": "tinyllama-2",
        "messages": [
            {"role": "user", "content": "Say hello"},
        ],
    })

    assert len(chunks) > 1, "Expected more than one streamed chunk"
    assert all(c["done"] is False for c in chunks[:-1])
    assert chunks[-1]["done"] is True

    content = "".join(c["message"]["content"] for c in chunks if c["message"]["content"])
    assert len(content) > 0, "Streamed content should not be empty"
    assert all(c["message"]["role"] == "assistant" for c in chunks)


def test_ollama_chat_with_system():
    server.start()

    res = server.make_request("POST", "/api/chat", data={
        "model": "tinyllama-2",
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
    })

    assert res.status_code == 200
    assert res.body["done"] is True
    assert len(res.body["message"]["content"]) > 0


# /api/generate

def test_ollama_generate_basic():
    server.start()

    res = server.make_request("POST", "/api/generate", data={
        "model": "tinyllama-2",
        "stream": False,
        "prompt": "Say hello",
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    body = res.body
    assert body["model"] == "tinyllama-2"
    assert body["done"] is True
    assert body["done_reason"] in ("stop", "length")
    assert isinstance(body["response"], str)
    assert len(body["response"]) > 0
    assert body["eval_count"] > 0


def test_ollama_generate_stream():
    server.start()

    chunks = ollama_stream_request("/api/generate", {
        "model": "tinyllama-2",
        "prompt": "Say hello",
    })

    assert len(chunks) > 1
    assert chunks[-1]["done"] is True
    response = "".join(c["response"] for c in chunks)
    assert len(response) > 0


def test_ollama_generate_options():
    server.start()

    res = server.make_request("POST", "/api/generate", data={
        "model": "tinyllama-2",
        "stream": False,
        "prompt": "Say hello",
        "options": {
            "num_predict": 4,
        },
    })

    assert res.status_code == 200
    assert res.body["eval_count"] <= 4


# metadata endpoints

def test_ollama_tags():
    server.start()

    res = server.make_request("GET", "/api/tags")
    assert res.status_code == 200
    models = res.body["models"]
    assert isinstance(models, list) and len(models) >= 1
    assert models[0]["name"] == "tinyllama-2"
    assert models[0]["model"] == "tinyllama-2"
    assert "details" in models[0]


def test_ollama_ps():
    server.start()

    res = server.make_request("GET", "/api/ps")
    assert res.status_code == 200
    models = res.body["models"]
    assert isinstance(models, list) and len(models) >= 1
    assert models[0]["name"] == "tinyllama-2"


def test_ollama_version():
    server.start()

    res = server.make_request("GET", "/api/version")
    assert res.status_code == 200
    assert isinstance(res.body["version"], str)


def test_ollama_show():
    server.start()

    res = server.make_request("POST", "/api/show", data={
        "model": "tinyllama-2",
    })
    assert res.status_code == 200
    assert "details" in res.body
    assert "capabilities" in res.body


# model management endpoints are not supported

def test_ollama_model_management_not_supported():
    server.start()

    for path in ("/api/pull", "/api/push", "/api/create", "/api/copy"):
        res = server.make_request("POST", path, data={"model": "tinyllama-2"})
        assert res.status_code == 501, f"Expected 501 for {path}, got {res.status_code}"
        assert "error" in res.body

    res = server.make_request("DELETE", "/api/delete", data={"model": "tinyllama-2"})
    assert res.status_code == 501, f"Expected 501 for /api/delete, got {res.status_code}"
    assert "error" in res.body


# /api/embeddings

def test_ollama_embeddings():
    global server
    server = ServerPreset.bert_bge_small()
    server.model_alias = "bert-bge-small"
    server.start()

    res = server.make_request("POST", "/api/embeddings", data={
        "model": "bert-bge-small",
        "prompt": "Hello world",
    })

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.body}"
    embedding = res.body["embedding"]
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)
