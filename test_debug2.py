from pathlib import Path
import tempfile
from thirdeye.config import Config
from thirdeye.store import Store
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.interactions import canonical_interactions

def _append(store, sid, event_type, data):
    return store.append_event(
        session_id=sid, platform="cursor", cwd="/repo", t=event_type, data=data
    )

with tempfile.TemporaryDirectory() as tmp_path:
    tmp_path = Path(tmp_path)
    sid, generation = "recovery-session", "gen-recovery"
    store = Store(Config(root=tmp_path))
    turn_seq = _append(
        store,
        sid,
        "user_message",
        {"generation_id": generation, "prompt": "ship it", "flags": {"urgent": True}},
    )
    _append(
        store,
        sid,
        "assistant_thought",
        {"generation_id": generation, "text": "plan", "model": "claude-4"},
    )
    call_seq = _append(
        store,
        sid,
        "tool_call",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "tool_use_id": "shell-1",
            "command": "pytest -q",
        },
    )
    result_seq = _append(
        store,
        sid,
        "tool_result",
        {
            "generation_id": generation,
            "tool_name": "shell",
            "cursor_tool_family": "shell",
            "tool_use_id": "shell-1",
            "output": "passed",
        },
    )
    response_seq = _append(
        store,
        sid,
        "assistant_message",
        {"generation_id": generation, "text": "all green"},
    )
    stop_seq = _append(
        store,
        sid,
        "turn_stop",
        {"generation_id": generation, "input_tokens": 12, "output_tokens": 3},
    )
    events = list(store.reader(sid).iter_events())
    
    # Get canonical interactions
    interactions = canonical_interactions(events, generation_id=generation, through_seq=stop_seq)
    
    # Filter out tools
    filtered = [item for item in interactions if item.kind not in {"tool_call", "tool_result"}]
    
    print(f"Total interactions: {len(interactions)}")
    print(f"Non-tool interactions: {len(filtered)}")
    for item in filtered:
        print(f"  {item.kind}: seq={item.source_seq}, id={item.interaction_id}")
        
    print(f"\nTool and response seqs: call={call_seq}, result={result_seq}, response={response_seq}")
    print(f"Tool seqs in filtered: {[item.source_seq for item in filtered if item.source_seq in [call_seq, result_seq, response_seq]]}")
