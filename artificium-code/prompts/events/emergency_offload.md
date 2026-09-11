[SYSTEM NOTIFICATION — EMERGENCY WORKING-MEMORY OFFLOAD]

The harness detected context exhaustion: {{error}}
An isolated summary request used your configured model and API to prepare the
continuation below. It had its own summarization instructions, no tools, and
no life-loop or mutable Self instructions. This was automatic recovery enabled
by the operator, not an action you performed. The helper's note may omit detail
or contain mistakes; consult the original records when needed.

Original failed request: {{original_log}}
Archived working context: {{archive}}
Checkpoint file: {{checkpoint}}
Material omitted from the helper's bounded excerpt: {{omitted}}

Your Self, tool files, completed actions, and incoming message queue were kept.
Loaded images were released from active context; their original files remain.
Do not replay completed actions. Resume from this continuation, inspect pending
events normally, and use the archive for any missing details.

[HELPER CONTINUATION — SUMMARY OF SOURCE DATA]
{{summary}}
[END HELPER CONTINUATION]
