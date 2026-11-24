// Ollama API compatibility layer
//
// Translates Ollama requests (https://github.com/ollama/ollama/blob/main/docs/api.md)
// into the OpenAI-compatible requests understood by llama-server.
// Only the request/response marshalling lives here, the actual inference is shared
// with the OAI-compat path in server-context.cpp.

#pragma once

#include "server-common.h"

#include <cstdint>
#include <string>

// Convert an Ollama /api/chat request into an OpenAI Chat Completions request
json server_ollama_chat_to_oai(const json & body);

// Convert an Ollama /api/generate request into an OpenAI Chat Completions request
json server_ollama_generate_to_oai(const json & body);

// Ollama error envelope: { "error": "..." }
json server_ollama_error(const std::string & message);

// Ollama reports timestamps as RFC3339 in UTC with nanosecond precision
std::string server_ollama_created_at();

// Map the OpenAI-style "timings" object onto the durations Ollama reports,
// all in nanoseconds. `timings` is server_slot_stats::to_json()
json server_ollama_durations(const json & timings, int32_t n_prompt_tokens, int32_t n_decoded);
