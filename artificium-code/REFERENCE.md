# API and client reference

[README](../README.md) covers setup, operation, architecture, and safety.
This file defines the integration details needed by custom providers and clients.

## Model transports

| Provider flag | Transport | Reasoning mapping in this release |
|---|---|---|
| `llamacpp` | `/v1/chat/completions` | Effort via `reasoning_effort`; on via `chat_template_kwargs.enable_thinking` |
| `ollama` | Native `/api/chat` | `think`; GPT-OSS uses low/medium/high |
| `vllm` | `/v1/chat/completions` | `reasoning_effort`, optional `thinking_token_budget` |
| `openrouter` | `/api/v1/chat/completions` | `reasoning.effort` or `reasoning.max_tokens` |
| `openai` | `/v1/responses` | `reasoning.effort` |
| `gemini` | `models/{model}:generateContent` | `thinkingLevel` or model-specific `thinkingBudget` |
| `anthropic` | `/v1/messages` | `output_config.effort`, model-specific thinking budget |
| `custom` | Configured chat-completions endpoint | Conventional compatible fields; server-defined support |

Provider-specific constraints are validated before saving. Omitted generation
settings preserve server defaults. CLI help lists controls; the interactive
model editor narrows them by transport/model. Custom endpoints expose optional
reasoning and sampling extensions without assuming that every server honors them.

`doctor` shows the adapter, exact URL, safe header names, and request parameters.
`check` (also `doctor --live`) uses the complete prompt pack, pinned Self,
meta-memory and saved context with a diagnostic instruction. It checks input
capacity, performs inference with the configured generation controls, and tests
image input where applicable. It reports measured usage and effective image
support. It does not start an agent, import mutable tools, consume notifications,
or retain the diagnostic response. Setup and model reconfiguration use this
same check before saving. Artificium supplies its textual tool protocol inside ordinary messages;
no native tool-call IDs or provider conversation chains are required.

Local default addresses: llama.cpp `http://127.0.0.1:8080/v1`, Ollama
`http://127.0.0.1:11434`, vLLM `http://127.0.0.1:8000/v1`.
Keys use `ARTIFICIUM_API_KEY` or `LLAMA_API_KEY`, `OLLAMA_API_KEY`,
`VLLM_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, or
`ANTHROPIC_API_KEY` as appropriate. An explicitly saved key takes precedence so
runtime uses the key that passed setup. Otherwise, the generic environment
variable precedes the provider variable and saved fallback. New saved keys are
bound to provider and address; legacy provider bindings remain readable. `key`
checks a replacement before saving it. A timed-out inference is not blindly
replayed; explicit transient HTTP failures receive bounded retries.

### Context, llama.cpp, and routing

llama.cpp discovery uses `/v1/models` plus `/props?model=ID`. Serving capacity
comes from `default_generation_settings.n_ctx`, not GGUF training capacity;
`modalities.vision` reports image support. Its adapter maps repetition penalty
to `repeat_penalty` and passes `reasoning_effort`. Meaningful effort levels
require a supporting chat template. Exact reasoning budgets and OpenAI reasoning
mode are rejected by this adapter. A vision model also needs the appropriate
server/projector configuration. See the upstream
[llama-server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

llama.cpp's optional `/apply-template` and `/tokenize` endpoints provide the
prompt token count before inference. If unavailable, the check labels its count
as an estimate. It reserves output space and rejects insufficient capacity;
it never truncates the mind or increases the server's allocation itself.

Ollama receives the configured window as `options.num_ctx`, normally 32,768 or
a lower advertised model maximum. An already loaded 4k slot does not cap this
new allocation. vLLM reports `max_model_len`. OpenRouter, Gemini and Anthropic
metadata supply limits/capabilities when exposed; OpenAI model discovery often
does not include capacity. Interactive setup asks once for missing capacity;
headless setup labels its 32,768 fallback as a working budget, not a verified
server limit. An explicit `--context-window` remains available. Automatically
detected capacities refresh on reconnection; deliberate manual limits persist.

`--reasoning auto` clears reasoning overrides; `off` maps to disabled reasoning.
`on` is available only for APIs with a toggle. Effort and exact token budgets are
alternative choices. A server accepting an effort parameter does not prove its
model/template uses it; controls remain dependent on that model. The UI uses
advertised capabilities and template controls where available. See the native
[Ollama thinking contract](https://docs.ollama.com/capabilities/thinking),
[Gemini model metadata](https://ai.google.dev/api/models), and
[Anthropic model metadata](https://platform.claude.com/docs/en/api/models/list).

To select one OpenRouter inference provider:

```bash
python3 artificium.py configure model --openrouter-provider cerebras
python3 artificium.py configure model --openrouter-provider automatic
```

The first sends `provider.only=["cerebras"]`; the second clears that constraint.
Explicit normalized controls use strict parameter routing. A fixed route can
fail if that provider cannot serve the requested model/settings.

For an extension verified against your server, edit `model.request_options` in
`artificium-code/config.json`, for example `chat_template_kwargs`. It cannot
replace protected model/prompt fields or conflict with a normalized setting.
Restart after editing. Do not embed credentials in request bodies or headers.

## Advanced custom JSON APIs

Copy and edit [the example contract](examples/custom-json-contract.example.json)
for APIs that are not OpenAI-compatible:

```bash
python3 artificium.py setup --provider custom --model MODEL --custom-contract my-engine.json --context-window 64000 --no-launch
```

A contract contains a complete `url`, optional static `headers`, `auth`, a JSON
`body`, and `response` paths. `auth: false` means unauthenticated. Otherwise a
supplied key defaults to `Authorization: Bearer`; `auth.header`, `auth.prefix`,
and `auth.required` customize authentication without embedding the key.

Exact `$artificium.NAME` strings preserve the substituted JSON type. Available
values are `model`, `messages`, `prompt`, `system`, `last_user`,
`context_window_tokens`, `reasoning_effort`, `reasoning_budget_tokens`,
`reasoning_mode`, `temperature`, `max_output_tokens`, `top_p`, `top_k`, `min_p`,
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, and
`stop_sequences`. Unset optional values are omitted from objects/lists.

Response fields use JSON Pointer paths, such as `/result/text`. `content` is
required; `reasoning`, `usage`, and `finish_reason` are optional. A list of paths
selects the first existing value. The contract supports one non-streaming JSON
POST and an object-shaped JSON response; it does not provide multipart,
websocket, streaming, or executable adapter plugins.

To return to ordinary compatibility:

```bash
python3 artificium.py configure model --custom-contract none --url http://localhost:8000/v1 --context-window 32768
```

## Interaction clients

Add `artificium-code/` to Python's import path, then use the bundled client:

```python
from artificium import ArtificiumClient

client = ArtificiumClient('/absolute/path/to/Artificium')
event, path = client.send(
    'project-room', sender='user_1', recipient='artificium',
    content='Inspect this image.', attachments=['/absolute/path/image.png'],
    interaction_name='Project room', kind='message',
)
for event in client.events('project-room'):
    print(event['sender'], event['content'])
```

`send` copies attachments before publishing the event, then creates a receipt
and compact notification. The message content is stored in the event; the
notification tells the agent where to read it. Loading an attachment is a
separate agent decision. Clients do not start the life-loop process.

| Relative path | Purpose |
|---|---|
| `mind/interactions/ID/interaction.json` | Interaction metadata and participants |
| `mind/interactions/ID/events/EVENT_ID.json` | Immutable inbound or outbound event |
| `mind/interactions/ID/attachments/` | Durable attachment files |
| `logs/runtime/interaction_receipts/` | Delivery, reading, and handling state; use runtime APIs for updates |

An inbound event has this shape:

```json
{
  "id": "event_example",
  "interaction_id": "project-room",
  "interaction_name": "Project room",
  "created_at": "2026-09-07T12:00:00.000000Z",
  "sender": "user_1",
  "recipient": "artificium",
  "direction": "inbound",
  "kind": "message",
  "content": "Hello.",
  "attachments": [],
  "in_reply_to": null
}
```

IDs use 1–128 letters, numbers, dots, dashes, or underscores. Use UTC ISO 8601
timestamps and an event filename matching its ID. An external writer can publish
metadata, attachments, and events without importing Python: write each file
atomically with a temporary sibling plus rename. The runtime reconciles inbound
`mind/interactions/*/events/*.json` files lacking receipts; it does not watch
arbitrary files. Finish attachment copies before publishing their event.

A reply is an event with `direction: "outbound"`; preserve sender, recipient,
interaction ID, and `in_reply_to`. Sort events by timestamp and ID. Notification
delivery is distinct from handling a request: an agent may answer, postpone, or
ignore an event. Internal trace output is not a reply. UIs should sanitize
rendered content, authenticate remote access, and treat attachment text as data.

Sensors, other agents, and screen observers use the same interface with their
own sender IDs and event kinds. Scheduled tasks arrive as ordinary
`scheduled_task` events; the task text specifies what to do and where results
belong. No separate chat-centric identity or privileged client type is required.

Reusable clients belong in `mind/tools/`; projects built for an entity belong
in `mind/space/`. For oversized textual attachments use bounded reads or Infinite
Attention. Audio, video, PDFs, and other binaries stay as durable files until
appropriate conversion apparatus is available.
