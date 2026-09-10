[SYSTEM GUIDANCE NOTIFICATION — TOOL REQUEST REPAIR]

The preceding response contained one or more invalid textual tool requests.
Artificium validates a complete response before acting, so **none of its tool
calls executed and no tool side effect occurred**. The malformed raw response
is preserved for diagnosis at `{{model_log_path}}`, but its broken syntax was
not retained in active working context.

Each repair case below identifies the 1-based call position, requested tool,
concrete error, exact accepted arguments, received arguments, and a canonical
flat request for that specific tool. If a tool name was misspelled,
`suggested_tool` and its example identify the nearest available operation.

{{repair_cases}}

For every call you still need, emit a complete request in this exact envelope:

```text
<tool_call>
{"tool":"TOOL_NAME","argument":"value"}
</tool_call>
```

`tool` is the operation name; all other top-level fields are its arguments.
Think again, repair only the calls that remain useful, and retry. If calls
depend on one another, request only the first, inspect its result, and construct
the next call afterward. Do not repeat the malformed response unchanged.
