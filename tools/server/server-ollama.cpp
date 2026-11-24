#include "server-ollama.h"
#include "server-common.h"

#include <chrono>
#include <cstdio>
#include <ctime>

// Ollama options are a flat object; most names differ from the llama.cpp ones.
// Unknown options (num_ctx, num_gpu, ...) are server-level and are dropped.
static void server_ollama_apply_options(json & oai, const json & options) {
    if (!options.is_object()) {
        return;
    }

    static const std::pair<const char *, const char *> mapping[] = {
        { "temperature",       "temperature"       },
        { "top_k",             "top_k"             },
        { "top_p",             "top_p"             },
        { "min_p",             "min_p"             },
        { "typical_p",         "typical_p"         },
        { "seed",              "seed"              },
        { "num_predict",       "n_predict"         },
        { "num_keep",          "n_keep"            },
        { "repeat_penalty",    "repeat_penalty"    },
        { "repeat_last_n",     "repeat_last_n"     },
        { "presence_penalty",  "presence_penalty"  },
        { "frequency_penalty", "frequency_penalty" },
        { "mirostat",          "mirostat"          },
        { "mirostat_tau",      "mirostat_tau"      },
        { "mirostat_eta",      "mirostat_eta"      },
    };
    for (const auto & [from, to] : mapping) {
        if (options.contains(from)) {
            oai[to] = options.at(from);
        }
    }

    // "stop" accepts a string or an array of strings, same as OAI
    if (options.contains("stop")) {
        oai["stop"] = options.at("stop");
    }
}

// Ollama "format" is either the string "json" or a JSON schema object
static void server_ollama_apply_format(json & oai, const json & format) {
    if (format.is_string()) {
        if (format.get<std::string>() == "json") {
            oai["response_format"] = json { { "type", "json_object" } };
        }
    } else if (format.is_object()) {
        oai["json_schema"] = format;
    }
}

static void server_ollama_push_images(json & content, const json & images) {
    if (!images.is_array()) {
        return;
    }
    for (const auto & image : images) {
        if (!image.is_string()) {
            continue;
        }
        // images arrive as raw base64 without a media type; the OAI content loader
        // decodes the payload by sniffing its bytes, so no data URL is needed
        content.push_back({
            { "type", "image_url" },
            { "image_url", { { "url", image.get<std::string>() } } },
        });
    }
}

static json server_ollama_convert_messages(const json & messages) {
    json out = json::array();
    if (!messages.is_array()) {
        return out;
    }

    for (const auto & msg : messages) {
        json converted;
        converted["role"] = json_value(msg, "role", std::string());

        const std::string text = json_value(msg, "content", std::string());
        const json images = json_value(msg, "images", json::array());

        if (images.is_array() && !images.empty()) {
            json content = json::array();
            if (!text.empty()) {
                content.push_back({ { "type", "text" }, { "text", text } });
            }
            server_ollama_push_images(content, images);
            converted["content"] = content;
        } else {
            converted["content"] = text;
        }

        // Ollama tool calls carry the arguments as an object, OAI as a JSON string
        if (msg.contains("tool_calls") && msg.at("tool_calls").is_array()) {
            json tool_calls = json::array();
            for (const auto & tool_call : msg.at("tool_calls")) {
                const json fn = json_value(tool_call, "function", json::object());
                json arguments = json_value(fn, "arguments", json::object());
                tool_calls.push_back({
                    { "type", "function" },
                    { "function", {
                        { "name", json_value(fn, "name", std::string()) },
                        { "arguments", arguments.is_string() ? arguments.get<std::string>() : arguments.dump() },
                    } },
                });
            }
            converted["tool_calls"] = tool_calls;
        }

        out.push_back(std::move(converted));
    }

    return out;
}

json server_ollama_chat_to_oai(const json & body) {
    json oai;

    if (body.contains("model")) {
        oai["model"] = body.at("model");
    }
    // Ollama streams by default, unlike the OpenAI API
    oai["stream"] = json_value(body, "stream", true);

    oai["messages"] = server_ollama_convert_messages(json_value(body, "messages", json::array()));

    if (body.contains("tools") && body.at("tools").is_array()) {
        oai["tools"] = body.at("tools");
    }
    if (body.contains("format")) {
        server_ollama_apply_format(oai, body.at("format"));
    }

    server_ollama_apply_options(oai, json_value(body, "options", json::object()));

    return oai;
}

json server_ollama_generate_to_oai(const json & body) {
    json oai;

    if (body.contains("model")) {
        oai["model"] = body.at("model");
    }
    // Ollama streams by default, unlike the OpenAI API
    oai["stream"] = json_value(body, "stream", true);

    json messages = json::array();

    const std::string system = json_value(body, "system", std::string());
    if (!system.empty()) {
        messages.push_back({ { "role", "system" }, { "content", system } });
    }

    const std::string prompt = json_value(body, "prompt", std::string());
    const json images = json_value(body, "images", json::array());

    json user;
    user["role"] = "user";
    if (images.is_array() && !images.empty()) {
        json content = json::array();
        if (!prompt.empty()) {
            content.push_back({ { "type", "text" }, { "text", prompt } });
        }
        server_ollama_push_images(content, images);
        user["content"] = content;
    } else {
        user["content"] = prompt;
    }
    messages.push_back(std::move(user));

    oai["messages"] = std::move(messages);

    if (body.contains("format")) {
        server_ollama_apply_format(oai, body.at("format"));
    }

    server_ollama_apply_options(oai, json_value(body, "options", json::object()));

    return oai;
}

json server_ollama_error(const std::string & message) {
    return json { { "error", message } };
}

std::string server_ollama_created_at() {
    using namespace std::chrono;

    const auto now = system_clock::now();
    const auto seconds_part = time_point_cast<std::chrono::seconds>(now);
    const int64_t nanos = duration_cast<nanoseconds>(now - seconds_part).count();

    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm = {};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif

    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &tm);
    return string_format("%s.%09lldZ", date, (long long) nanos);
}

json server_ollama_durations(const json & timings, int32_t n_prompt_tokens, int32_t n_decoded) {
    const double prompt_ms = json_value(timings, "prompt_ms", 0.0);
    const double eval_ms   = json_value(timings, "predicted_ms", 0.0);

    const int64_t prompt_ns = (int64_t) (prompt_ms * 1e6);
    const int64_t eval_ns   = (int64_t) (eval_ms * 1e6);

    return json {
        { "total_duration",       prompt_ns + eval_ns },
        { "load_duration",        (int64_t) 0         },
        { "prompt_eval_count",    n_prompt_tokens     },
        { "prompt_eval_duration", prompt_ns           },
        { "eval_count",           n_decoded           },
        { "eval_duration",        eval_ns             },
    };
}
