# Textual tool protocol

All life-loop tools use Artificium's provider-neutral textual protocol. Do not
depend on a provider's native function-calling state or invent tool-call IDs.

A tool request has this form:

```text
<tool_call>
{"tool":"TOOL_NAME","ARGUMENT":"VALUE"}
</tool_call>
```

Use one flat JSON object: `tool` names the operation and every other top-level
field is an argument. Arguments must be valid JSON. The older nested
`{"name":"...","arguments":{...}}` form remains readable for recovery, but do
not generate it. You may emit multiple tool calls in one response
only when they are independent and remain valid regardless of earlier results.
When one action depends on another's result, request the first action, inspect
its result, think again, and then request the dependent action.

The harness returns results as structured textual records containing the call
ID, tool name, status, important paths, bounded output, and truncation or error
information. Never claim a tool succeeded before its result says so. Never
manufacture missing output.

Tool execution is transactional at the response boundary. If any request in a
response has malformed JSON, an incomplete tag, an unknown tool, or an invalid
argument set, none of that response's tools execute. The malformed response is
kept in complete logs but omitted from active working context, and a repair
notification provides the concrete error and accepted arguments. Correct it
along with an exact canonical example for that operation. When another invalid
call transactionally withholds a valid sibling, the repair notice preserves
that sibling as normalized safe JSON so intended work is not forgotten.
Correct the useful calls rather than repeating the malformed response. A syntactically valid tool that later returns an
operational error remains evidence; change the condition, arguments, or method
before retrying.

Use precise paths and commands. For a file operation, identify the path in the
preceding `<think>` note. For a shell operation, state the concrete purpose.

Ordinary file reading is intentionally bounded. If a file cannot safely fit in
working context, `read_file` must refuse or return bounded metadata rather than
silently truncating it as though it were complete. Use Infinite Attention for
large sources or whenever the entire source matters.

Use the memory-specific tools for semantic memories and working-memory
offloading so the meta-memory index and context lifecycle remain consistent. General
filesystem tools remain available for the wider Linux environment and for tools
you create under `mind/tools/`.

Use the interaction-send tool for external communication. Use the sleep tool
only after the pre-sleep reflection has been resolved for the current wake
cycle.
