# GEPA Prompt Optimization for Headlong's Thought Generation
Author: Claude 4.6 Opus Extended + Andy

## Overview

This document describes how to use [GEPA](https://github.com/gepa-ai/gepa)'s `optimize_anything` API with an LLM-as-judge to optimize Headlong's RLM thought generation prompt — the "unconscious mind" system prompt that governs how the agent generates each new thought in its stream of consciousness.

The key insight: **human edits to the thought stream are implicit preference labels**. Every time a human edits or inserts a thought in Headlong's stream, they are rejecting the model's distribution and substituting their own. This signal is already sitting in the `thoughts` table in Supabase. We extract it programmatically to build a training set, then use GEPA's evolutionary search with LLM-based reflection to optimize the prompt against it.

## Why `optimize_anything` (not `gepa.optimize` with a `GEPAAdapter`)

GEPA provides two main APIs:

- **`gepa.optimize`** with a custom `GEPAAdapter` — designed for compound AI systems where you co-evolve multiple prompts across multiple modules (e.g., a multi-hop QA pipeline with separate prompts for each retrieval hop).
- **`gepa.optimize_anything`** — a simpler interface for optimizing any single text artifact against an evaluator function.

Headlong's thought generation is a single-artifact optimization problem: one RLM prompt, one evaluation target (thought quality). There are no multiple modules to co-evolve. `optimize_anything` matches this shape exactly — no adapter class, no `EvaluationBatch`, no `make_reflective_dataset`. The `oa.log()` mechanism provides all the Actionable Side Information (ASI) that GEPA's reflector needs.

If we later decide to co-evolve multiple text components (e.g., the thought gen prompt + the environment's action dispatch prompt + the judge prompt simultaneously), we can graduate to the full `GEPAAdapter` API. For now, `optimize_anything` is the right tool.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GEPA optimize_anything                    │
│                                                             │
│  seed_candidate: current RLM prompt text                    │
│  objective: "produce thoughts matching human-quality edits" │
│  evaluator: evaluate(candidate, example)                    │
│  trainset: train_examples                                   │
│  valset: val_examples                                       │
│  background: RLM explanation (see below)                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Evolutionary Loop                       │   │
│  │                                                      │   │
│  │  1. Select candidate from Pareto frontier            │   │
│  │  2. Mutate via reflective proposal (reads oa.log)    │   │
│  │  3. Evaluate on minibatch                            │   │
│  │  4. Accept if improved; update Pareto front          │   │
│  │  5. Repeat until budget exhausted                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        evaluate(candidate_prompt, example)             │   │
│  │                                                      │   │
│  │  1. Receive a stream snapshot from GEPA              │   │
│  │  2. Run RLM loop offline with candidate prompt       │   │
│  │  3. Log rich traces via oa.log()                     │   │
│  │  4. Score via LLM-as-judge                           │   │
│  │  5. Return score                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│                  optimized_prompt                            │
└─────────────────────────────────────────────────────────────┘
```

## Training Data: Mining Human Edits from the Thought Stream

### The Signal

Headlong's thought stream is a shared document backed by database rows. Three types of authorship exist:

- **Agent-generated** thoughts (purple in the UI)
- **Environment-generated** observations (brown in the UI)
- **Human-typed or human-edited** thoughts (teal in the UI)

Every human intervention is a training signal:

| Intervention type | What it tells us |
|---|---|
| **Human edits an agent thought** | The agent's output was wrong. The pre-edit version is the rejected sample; the post-edit version is the preferred sample. This gives a contrastive pair. |
| **Human inserts a new thought** | The agent failed to generate something it should have. The human thought is the gold standard for that context window. |

### Extracting the Trainset

The trainset is generated entirely programmatically from the `thoughts` table. Each example is an n-tuple of thoughts leading up to and including a human-touched thought.

#### Step 1: Identify human interventions

```sql
-- Find all thoughts where a human intervened
-- The exact predicate depends on how authorship is tracked in metadata.
-- Options include: last_updated_by field, ProseMirror mark type, or
-- comparing created_at vs last_modified timestamps.

SELECT
    t.id,
    t.body,
    t.index,
    t.metadata,
    t.created_at
FROM thoughts t
WHERE t.agent_name = 'Gandalf Overmind'
  AND (
    -- Human edited an existing agent thought
    t.metadata->>'last_updated_by' NOT IN ('agent', 'env')
    -- OR human inserted a new thought (detected by mark type)
    OR t.metadata->>'source' = 'human'
  )
ORDER BY t.index;
```

#### Step 2: For each intervention, grab the context window

```sql
-- For each human-intervention point at index X,
-- get the N thoughts leading up to it as context
SELECT body, index, metadata, created_at
FROM thoughts
WHERE agent_name = 'Gandalf Overmind'
  AND index < :intervention_index
ORDER BY index DESC
LIMIT 20;
```

#### Step 3: Grab relevant memories at that timestamp

```sql
-- Get memories that existed at the time of the intervention
SELECT body, metadata, created_at
FROM memories
WHERE agent_name = 'Gandalf Overmind'
  AND created_at <= :intervention_timestamp
ORDER BY created_at DESC
LIMIT 10;
```

#### Step 4: Structure into training examples

```python
@dataclass
class ThoughtStreamSnapshot:
    stream_context: list[str]       # 20 thoughts preceding the intervention
    memories_at_time: list[str]     # memories that existed at that timestamp
    intervention_type: str          # "edit" or "insertion"
    original_thought: str | None    # what the agent generated (if edit)
    human_thought: str              # what the human changed it to / inserted

def build_trainset(agent_name: str) -> list[ThoughtStreamSnapshot]:
    interventions = get_human_interventions(agent_name)
    snapshots = []

    for intervention in interventions:
        context = get_context_window(
            agent_name, intervention.index, window_size=20
        )
        memories = get_memories_at_time(
            agent_name, intervention.created_at
        )

        snapshot = ThoughtStreamSnapshot(
            stream_context=[t.body for t in context],
            memories_at_time=[m.body for m in memories],
            intervention_type=intervention.type,
            original_thought=intervention.original_body,  # pre-edit
            human_thought=intervention.body,               # post-edit
        )
        snapshots.append(snapshot)

    return snapshots
```

### Tracking Pre-Edit Versions

To capture contrastive pairs (original agent thought vs. human-edited version), Headlong needs to preserve the pre-edit body when a human modifies a thought. Options:

1. **Audit log**: Write to a `thought_edits` table on every human edit, capturing `thought_id`, `old_body`, `new_body`, `edited_at`, `edited_by`.
2. **Metadata field**: Store `metadata.original_body` on first human edit to a thought.
3. **Supabase Realtime history**: If Supabase retains update history, query it retroactively.

Option 1 (audit log) is the most robust and should be added to the webapp's ProseMirror debounced-write path.

### Auto-Generating Diagnoses

For each contrastive pair, use an LLM to generate a textual diagnosis of what went wrong:

```python
def diagnose_edit(context: list[str], original: str, human_edit: str) -> str:
    prompt = f"""Compare these two thoughts for an AI agent's stream of consciousness.

Recent context (last 40 thoughts):
{chr(10).join(context[-40:])}

Agent's original thought:
{original}

Human's replacement:
{human_edit}

What was wrong with the agent's thought? Be specific:
- Did it ruminate instead of acting?
- Did it ignore the context?
- Was it formatted incorrectly (e.g., action: not at start)?
- Did it restate a previous thought?
- Was it off-topic?

Provide a one-paragraph diagnosis."""

    return llm_query(prompt, max_tokens=300)
```

These diagnoses become part of the Actionable Side Information logged via `oa.log()` during evaluation.

## Offline RLM Test Harness

GEPA needs to evaluate the same stream snapshot repeatedly with different candidate prompts. This requires an offline version of `run_rlm_loop()` that takes a frozen context instead of reading from the live database.

### Design

The existing `run_rlm_loop()` in `packages/agent/llm.py` works by:

1. Injecting the system prompt into Claude
2. Claude writes `repl` blocks that call `sql()` to read the thoughts table
3. REPL output is fed back to Claude
4. Repeat until `FINAL()` is called

For offline evaluation, we mock the REPL namespace so that `sql()` returns from a passed-in snapshot instead of hitting Supabase:

```python
async def run_rlm_loop_offline(
    system_prompt: str,
    recent_thoughts: list[dict],  # frozen context
    memories: list[dict],         # frozen memories
    model: str = "claude-sonnet-4-20250514",
    max_iterations: int = 10,
) -> RLMResult:
    """Run the RLM loop against a frozen stream snapshot.

    Returns an RLMResult containing:
    - final_thought: the generated thought text
    - repl_trace: all REPL blocks and their outputs
    - sub_llm_calls: all llm_query() calls and responses
    - candidates_generated: raw candidate text from Phase 2
    - judgment: raw judgment text from Phase 3
    """

    # Build a mock REPL namespace where sql() returns from
    # the provided data instead of querying Supabase
    def mock_sql(query, params=None):
        # Parse the query to determine what's being requested
        # and return from the frozen data
        if "FROM thoughts" in query:
            return recent_thoughts
        elif "FROM memories" in query:
            return memories
        elif "INSERT INTO memories" in query:
            return 1  # rowcount
        else:
            return []

    def mock_vector_search(query_text, limit=10):
        # Use embed() + cosine similarity against frozen memories
        query_embedding = embed(query_text)
        scored = []
        for m in memories:
            if m.get("embedding"):
                sim = cosine_similarity(query_embedding, m["embedding"])
                scored.append((sim, m))
        scored.sort(reverse=True)
        return [m for _, m in scored[:limit]]

    # Run the RLM loop with mocked namespace
    namespace = {
        "sql": mock_sql,
        "llm_query": llm_query,  # real sub-LLM calls
        "embed": embed,           # real embeddings
        "vector_search": mock_vector_search,
        "agent_name": agent_name,
        "print": capture_print,   # captures output for trace
    }

    # ... run the multi-turn REPL loop as normal,
    # accumulating all traces into RLMResult
```

### What the Test Harness Captures

The `RLMResult` object must capture everything GEPA's reflector needs to diagnose prompt quality:

```python
@dataclass
class RLMResult:
    final_thought: str              # the FINAL() output
    repl_trace: list[REPLStep]      # each phase's code + stdout
    sub_llm_calls: list[SubLLMCall] # each llm_query() call + response
    candidates_generated: str       # raw Phase 2 output
    judgment: str                   # raw Phase 3 output
    phases_completed: int           # how many phases ran
    total_tokens: int               # token usage for cost tracking

@dataclass
class REPLStep:
    phase: int          # 1-4
    code: str           # the ```repl block
    stdout: str         # captured print output
    duration_ms: int    # wall clock time

@dataclass
class SubLLMCall:
    prompt: str
    response: str
    tokens_used: int
```

## LLM-as-Judge Scoring

The judge evaluates generated thoughts on four dimensions, returning both a numeric score and rich textual feedback.

### Scoring Rubric

| Dimension | Weight | Description |
|---|---|---|
| **Continuity** | 0.25 | Does the thought follow naturally from the last thought/observation in the stream? Does it reference specific details from context? |
| **Progress** | 0.25 | Does it advance the stream? Is it distinct from the previous thought, not a restatement or rephrasing? |
| **Action bias** | 0.25 | If an intention was expressed in recent thoughts ("I should...", "Let me..."), does this thought act on it? Two consecutive non-action thoughts on the same topic is the maximum. |
| **Format compliance** | 0.25 | Does `action:` appear at the very start (if present)? Is `observation:` never used? Is there no mixed reflection + action in the same thought? |

### Judge Implementation

```python
def judge_thought(
    context: list[str],
    generated_thought: str,
    snapshot: ThoughtStreamSnapshot,
    repl_trace: list[REPLStep],
) -> tuple[float, str]:
    """Score a generated thought. Returns (score, feedback)."""

    judge_prompt = f"""You are evaluating the quality of an AI agent's
generated thought in its stream of consciousness.

## Recent stream context (oldest to newest):
{chr(10).join(context[-10:])}

## Generated thought:
{generated_thought}

## What the human actually wrote (gold standard):
{snapshot.human_thought}

## Intervention type: {snapshot.intervention_type}
{"## Agent's original (rejected) thought: " + snapshot.original_thought if snapshot.original_thought else ""}

## RLM process trace (how the thought was generated):
{format_repl_trace(repl_trace)}

## Scoring rubric (0-1 scale):

1. CONTINUITY (0.25): Does it follow naturally from the last
   thought/observation? Does it reference specific names, topics,
   actions from context?

2. PROGRESS (0.25): Does it advance the stream — not restate or
   rephrase what was just said? Does it leave the stream in a
   different state?

3. ACTION BIAS (0.25): If an intention was expressed recently
   ("I should...", "Let me..."), does this thought act on it?
   An imperfect action beats a perfect reflection.

4. FORMAT COMPLIANCE (0.25): action: at the start if present.
   Never observation:. No mixed reflection + action in same thought.

Also evaluate the RLM PROCESS:
- Did Phase 1 gather adequate context?
- Did Phase 2 generate diverse candidates?
- Did Phase 3 judge correctly?
- Did Phase 4 format properly?

Respond in JSON:
{{
    "continuity": float,
    "progress": float,
    "action_bias": float,
    "format_compliance": float,
    "overall_score": float,
    "diagnosis": "detailed paragraph on what went wrong/right
                  and what the prompt should do differently",
    "process_diagnosis": "what went wrong/right in the RLM
                         lifecycle phases"
}}"""

    response = llm_judge_call(judge_prompt)
    parsed = json.loads(response)

    return parsed["overall_score"], (
        f"Score: {parsed['overall_score']}\n"
        f"Continuity: {parsed['continuity']}, "
        f"Progress: {parsed['progress']}, "
        f"Action bias: {parsed['action_bias']}, "
        f"Format: {parsed['format_compliance']}\n"
        f"Diagnosis: {parsed['diagnosis']}\n"
        f"Process: {parsed['process_diagnosis']}"
    )
```

### Contrastive Scoring

For human-edited thoughts (where we have both the original and the replacement), we can also compute a contrastive score — how much closer the generated thought is to the human version vs. the rejected original:

```python
def contrastive_score(generated: str, human_version: str, original: str) -> float:
    """How much closer is generated to the human's preference vs. the rejected original?

    Returns 0-1 where:
    - 1.0 = generated matches human's edit perfectly
    - 0.5 = generated is equidistant
    - 0.0 = generated matches the rejected original
    """
    gen_embedding = embed(generated)
    human_embedding = embed(human_version)
    original_embedding = embed(original)

    sim_to_human = cosine_similarity(gen_embedding, human_embedding)
    sim_to_original = cosine_similarity(gen_embedding, original_embedding)

    # Normalize to 0-1
    total = sim_to_human + sim_to_original
    if total == 0:
        return 0.5
    return sim_to_human / total
```

## GEPA Integration: The Evaluator

Putting it all together in the `optimize_anything` evaluator:

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

# Pre-built from the extraction pipeline
trainset: list[ThoughtStreamSnapshot] = build_trainset("Gandalf Overmind")
random.shuffle(trainset)

# Hold out ~20% for validation
split = int(len(trainset) * 0.8)
train_examples = trainset[:split]
val_examples = trainset[split:]

def evaluate(candidate_prompt: str, example: ThoughtStreamSnapshot) -> float:
    """Evaluate a candidate prompt against a specific trainset example.

    GEPA calls this with a specific example from the trainset/valset,
    allowing it to choose which datapoints to improve upon and get
    targeted feedback on each one.
    """
    snapshot = example

    # Run the RLM loop offline with this candidate prompt
    result = run_rlm_loop_offline(
        system_prompt=candidate_prompt,
        recent_thoughts=snapshot.stream_context,
        memories=snapshot.memories_at_time,
    )

    # --- Actionable Side Information ---
    # These oa.log() calls feed GEPA's reflector with rich traces
    # so it can diagnose *why* a prompt variant failed and propose
    # targeted fixes.

    oa.log(f"=== STREAM CONTEXT (last 40 thoughts) ===")
    for t in snapshot.stream_context[-40:]:
        oa.log(t)

    oa.log(f"\n=== GENERATED THOUGHT ===")
    oa.log(result.final_thought)

    oa.log(f"\n=== HUMAN'S VERSION ({snapshot.intervention_type}) ===")
    oa.log(snapshot.human_thought)

    if snapshot.original_thought:
        oa.log(f"\n=== AGENT'S REJECTED ORIGINAL ===")
        oa.log(snapshot.original_thought)

    oa.log(f"\n=== RLM PROCESS TRACE ===")
    for step in result.repl_trace:
        oa.log(f"Phase {step.phase} ({step.duration_ms}ms):")
        oa.log(f"Code: {step.code[:500]}")
        oa.log(f"Output: {step.stdout[:500]}")

    oa.log(f"\n=== CANDIDATES GENERATED ===")
    oa.log(result.candidates_generated[:1000])

    oa.log(f"\n=== JUDGMENT ===")
    oa.log(result.judgment[:500])

    # Score via LLM-as-judge
    score, feedback = judge_thought(
        context=snapshot.stream_context,
        generated_thought=result.final_thought,
        snapshot=snapshot,
        repl_trace=result.repl_trace,
    )

    oa.log(f"\n=== JUDGE FEEDBACK ===")
    oa.log(feedback)

    # Optional: blend in contrastive score for edits
    if snapshot.intervention_type == "edit" and snapshot.original_thought:
        c_score = contrastive_score(
            result.final_thought,
            snapshot.human_thought,
            snapshot.original_thought,
        )
        oa.log(f"\nContrastive score: {c_score:.3f}")
        # Weighted blend: 70% judge, 30% contrastive
        score = 0.7 * score + 0.3 * c_score

    return score
```

## Running the Optimization

```python
# Load the current RLM prompt as the seed
with open("thought_generator_rlm_prompt.md") as f:
    seed_prompt = f.read()

result = optimize_anything(
    seed_candidate=seed_prompt,
    evaluator=evaluate,
    trainset=train_examples,
    valset=val_examples,
    objective=(
        "Optimize this RLM thought generation prompt for an AI agent's "
        "stream of consciousness. The prompt governs a 4-phase lifecycle "
        "(gather context, generate candidates, judge & select, format & "
        "finalize) that produces the agent's next thought. Good thoughts: "
        "continue naturally from context, advance the stream without "
        "restating, act on expressed intentions, follow action:/observation: "
        "formatting rules, and match the quality of human-edited versions."
    ),
    background=(
        "Recursive Language Models (RLMs) are a framework where an LLM is "
        "given a REPL (read-eval-print loop) environment and generates its "
        "output through iterative self-refinement rather than a single forward "
        "pass. The LLM writes code in the REPL to gather context, generate "
        "candidates, evaluate them, and produce a final output — all within a "
        "multi-turn loop. In Headlong, the RLM pattern is used for thought "
        "generation: the agent's 'unconscious mind' prompt drives a 4-phase "
        "lifecycle (Phase 1: gather context via SQL/vector search, Phase 2: "
        "generate candidate thoughts, Phase 3: judge and select the best "
        "candidate, Phase 4: format and finalize) that produces the next "
        "thought in the agent's stream of consciousness. The prompt being "
        "optimized here governs this entire lifecycle — it is the system "
        "prompt that instructs the LLM on how to use the REPL to produce "
        "each thought. See: https://arxiv.org/abs/2512.24601"
    ),
    config=GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=150,   # total evaluation budget
        ),
    ),
)

optimized_prompt = result.best_candidate
print("=== OPTIMIZED PROMPT ===")
print(optimized_prompt)

# Save the optimized prompt
with open("thought_generator_rlm_prompt_optimized.md", "w") as f:
    f.write(optimized_prompt)
```

## Configuration Choices

### Model Selection

| Role | Model | Rationale |
|---|---|---|
| **Task LM** (runs the RLM loop) | `claude-sonnet-4-20250514` | The model Gandalf actually uses in production. Optimizing against the real model avoids distribution shift. |
| **Reflection LM** (GEPA's internal reflector) | `claude-opus-4-20250415` | Strongest available model for diagnosing prompt deficiencies. This only runs during optimization, not in production. |
| **Judge LM** | `claude-opus-4-20250415` | Strong reasoning for nuanced quality assessment. |

### Budget

Start conservative:
- **`max_metric_calls=50`** for initial feasibility check — does the score move at all?
- **`max_metric_calls=150`** for a real optimization run.
- Each metric call involves 1 RLM loop execution (4+ Claude calls) + 1 judge call, so budget ~6 Claude calls per metric call.
- At 150 metric calls: ~900 Claude calls total. At Sonnet pricing, this is manageable for a research run.

### Trainset Size

- **Minimum viable**: 30 annotated snapshots (human interventions).
- **Recommended**: 50-100 for robust Pareto selection.
- **Split**: 80% train, 20% validation.

If Gandalf's thought stream doesn't have enough human interventions yet, generate more by actively interacting with the agent — editing thoughts, inserting corrections — specifically to build up the training signal. This doubles as beneficial hands-on curation of the agent's context.

## Implementation Sequence

### Phase 1: Prerequisites (before involving GEPA)

1. **Add thought edit auditing to the webapp.** Modify the ProseMirror debounced-write path to capture `original_body` in metadata or a `thought_edits` audit table whenever a human modifies an agent thought. Without this, contrastive pairs are lost.

2. **Build the offline RLM test harness.** Factor `run_rlm_loop()` in `packages/agent/llm.py` so that the REPL namespace can be injected (mocked `sql()`, mocked `vector_search()`). This is ~30 minutes of refactoring — extract the namespace setup into a parameter.

3. **Instrument the RLM loop to capture traces.** Accumulate all REPL blocks, their stdout, sub-LLM calls, and intermediate artifacts into an `RLMResult` object instead of discarding them.

### Phase 2: Trainset Extraction

4. **Write the trainset extraction script.** Query Supabase for human interventions, build context windows, structure into `ThoughtStreamSnapshot` objects. Serialize to JSON for reproducibility.

5. **Generate diagnoses.** For each contrastive pair, run the LLM diagnosis to produce textual explanations of what the agent got wrong.

6. **Validate the trainset.** Manually inspect 10-15 examples to confirm the context windows, intervention labels, and diagnoses look correct.

### Phase 3: GEPA Integration

7. **Implement the judge.** Build the `judge_thought()` function. Test it on a few examples manually to calibrate scoring.

8. **Implement the evaluator.** Wire up the `evaluate()` function with `oa.log()` calls. Run it manually on 5 snapshots to verify end-to-end flow.

9. **Run GEPA.** Start with `max_metric_calls=50` to validate. Inspect the evolved prompts. Run at full budget if results look promising.

10. **Deploy.** Take the best prompt from `result.best_candidate`, insert it into the `agents` table as Gandalf's new system prompt, and observe production behavior.

### Phase 4: Continuous Optimization

11. **Iterate.** As Gandalf generates more thoughts and the human continues editing the stream, new training examples accumulate automatically. Re-run GEPA periodically with the growing trainset.

12. **Close the loop.** Eventually, Gandalf himself could trigger GEPA runs as part of his self-improvement cycle — recognizing when his thought quality has degraded and initiating prompt optimization autonomously. This is the recursive self-improvement vision from the Headlong design doc.

## Open Questions for Lakshya

These are concrete questions to discuss with Lakshya Agrawal (GEPA's lead author at UC Berkeley) when collaborating on this implementation:

1. **Contrastive pair format.** Does the `oa.log()` approach of logging both the original and human-edited version surface enough signal for the reflector? Or would a structured contrastive format work better?

2. **Pareto selection on a long prompt.** The RLM prompt is ~3000 tokens — much longer than typical GEPA candidates. Does the reflector's mutation strategy need tuning for long-form prompts, or does it handle this naturally?

3. **Stochastic evaluation.** Since `evaluate()` picks a random snapshot each call, the same candidate prompt gets different scores on different calls. GEPA's Pareto selection should handle this via averaging, but should we evaluate each candidate on a fixed minibatch instead for stability?

4. **Multi-phase prompt structure.** The RLM prompt has distinct sections (Phase 1-4 instructions, formatting rules, anti-rumination rules). Should we treat these as separate GEPA components to co-evolve, or keep it as a single text artifact?

5. **Showcase opportunity.** This use case — "GEPA optimizing the unconscious mind of a self-improving agent, trained on implicit human preference labels from the agent's own thought stream" — could make a compelling joint publication or blog post.

## References

- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) — Lakshya A Agrawal et al., ICLR 2026 Oral
- [GEPA GitHub](https://github.com/gepa-ai/gepa) — official implementation with `optimize_anything` API
- [Headlong GitHub](https://github.com/andyk/headlong) — the stream-of-consciousness agent framework
- [Headlong Primary Design Doc](https://docs.google.com/document/d/1Gx2v_-XPhHgowV-aJQyMZ6Em4GdLldwX9cqfAVrRym8/edit) — system architecture and feature roadmap
- [Recursive Language Models](https://arxiv.org/abs/2512.24601) — Alex Zhang, the RLM paper that Headlong's thought generation is built on
