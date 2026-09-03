from pathlib import Path
import tempfile
from thirdeye.config import Config
from thirdeye.store import Store
from thirdeye.paths import session_dir
from thirdeye.platforms.cursor.interactions import canonical_interactions

with tempfile.TemporaryDirectory() as tmp_path:
    tmp_path = Path(tmp_path)
    sid, generation = "test", "gen-test"
    store = Store(Config(root=tmp_path))
    
    # Append two assistant_thought events
    seq1 = store.append_event(
        session_id=sid,
        platform="cursor",
        cwd="/repo",
        t="assistant_thought",
        data={"generation_id": generation, "text": "plan", "model": "claude-4"},
    )
    seq2 = store.append_event(
        session_id=sid,
        platform="cursor",
        cwd="/repo",
        t="assistant_thought",
        data={"generation_id": generation, "text": "plan", "model": "gpt-5"},
    )
    
    # Get all events
    events = list(store.reader(sid).iter_events())
    
    # Print timestamps
    for event in events:
        print(f"seq {event['seq']}: ts={event['ts']}")
    
    # Get canonical interactions
    interactions = canonical_interactions(events, generation_id=generation, through_seq=2)
    print(f"\nCanonical interactions: {len(interactions)}")
    for interaction in interactions:
        print(f"  {interaction.interaction_id}: ts={interaction.ts}, duplicate_seqs={interaction.duplicate_seqs}")
