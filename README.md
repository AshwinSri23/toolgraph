# toolgraph

*Durable, side-effect-aware DAG execution for MCP tool calls.*

Status: v0.1, October 2026. Author: Ashwin Srirankan.

---

## The problem

A long-running AI agent may make dozens of tool calls over minutes or hours. Today
three things go wrong:

| Problem | What happens |
|---|---|
| **Slow** | Calls run one after another, even when they don't depend on each other. |
| **Fragile** | If the process dies at minute 40, the work is lost or restarted from scratch. |
| **Dangerous** | On a retry the agent can't tell whether the first attempt went through, so it may send the email twice, charge twice or book twice. |

The MCP spec (2025-11-25) defines no retry or duplicate-effect semantics for
`tools/call`, so every client and server handles this differently, or not at all.

## What toolgraph does

toolgraph runs an agent's tool calls as a dependency graph:

- **in parallel** where calls are independent;
- **in order** where one needs another's result, or where two touch the same resource;
- **durably**, so a crashed run resumes from where it stopped;
- under one rule for side effects: **exactly once, or ask**. A real-world action is never
  silently repeated; when its outcome is unknown, a person or a read-back confirms it.

## Quick start

```bash
pip install -e ".[mcp]"          # or ".[dev]" for the test suite
python examples/travel.py --crash
```

```python
from mcp import Client
from toolgraph import Node, PolicyRegistry, Scheduler, SideEffect, Store, ref
from toolgraph.mcp_backend import MCPBackend

reg = PolicyRegistry()
reg.register("travel.search_*", side_effect=SideEffect.READ)
reg.register("travel.book_flight", side_effect=SideEffect.KEYED_WRITE)
reg.register("mail.send_email", side_effect=SideEffect.UNKEYED_WRITE, writes=("inbox:{to}",))

graph = [
    Node("flights", "travel.search_flights", {"dest": "Lisbon"}),
    Node("hotels", "travel.search_hotels", {"dest": "Lisbon"}),
    Node("book", "travel.book_flight", {"flight": ref("flights", "flight")}),
    Node("email", "mail.send_email", {"to": "me@example.com", "body": ref("book", "confirmation")}),
]

# inside an async function:
async with Client(travel_server) as t, Client(mail_server) as m:
    sched = Scheduler(Store("runs.db"), MCPBackend({"travel": t, "mail": m}), reg, run_id="trip-1")
    sched.start(graph)              # no-op if the run already exists (i.e. after a crash)
    report = await sched.run()      # resumes from the database on restart
```

`report.status` is `completed`, `failed`, or `blocked` (waiting on a person). Settle a
blocked run with `sched.answer(node_id, answer)` or
`sched.resolve_verification(node_id, applied=..., result=...)`, then call `run()` again,
or pass `run(wait_for_humans=True)` to wait for those calls inside the same process.

## How it works

```
search flights ─┐
                ├─► fetch details ─► book flight (write) ───┐
search hotels ──┴───────────────────────────────────────────┴─► send email (write)
```

### The graph: what can run when (`toolgraph/graph.py`)

Each tool call is a node. If a call needs another's output it uses
`ref("node_id", "path")` in its arguments, which creates a data edge. Calls with no path
between them run concurrently.

Some calls conflict without passing data ("edit file" then "run tests"). Tools declare
resources they read or write as templates such as `"file:{path}"`, filled in from the
call's arguments. Two calls touching the same resource, where at least one writes, keep
the order they were listed in. These **hazard edges** (read-after-write,
write-after-read, write-after-write) are the same analysis a compiler uses to order
memory operations that may alias. A resource whose argument is a `ref` (unknown until
run time) widens to a wildcard such as `file:*`, the same way a compiler treats a pointer
that may alias anything.

### Policies: how dangerous is each tool (`toolgraph/policy.py`)

| Policy | Example | On an uncertain outcome |
|---|---|---|
| `READ` | search, fetch | Repeat freely |
| `KEYED_WRITE` | payments or booking APIs | Re-send with the **same** key; the server recognises the repeat |
| `IDEMPOTENT_WRITE` | "set status to X" | Repeat |
| `UNKEYED_WRITE` | plain email tool | **Never repeat.** Park in `needs_verification` |

Unregistered tools are treated as the most dangerous kind (unkeyed write, one attempt).
Policies also set retries, backoff, timeouts, job timeouts and concurrency groups
(for example, a server that runs one job at a time). Keys are derived from
`(run_id, node_id)`, persisted with the node, and sent both in the `key_arg` argument
(default `idempotency_key`) and in `_meta["io.toolgraph/idempotency-key"]`.

A failed attempt is classified as **definite** (rejected before anything happened, such as
a JSON-RPC `invalid params` error) or **uncertain** (lost response, timeout, dropped
connection, or a tool error on a write, which might have partly applied). Definite
failures are retried for every policy; uncertain ones follow the table above.

### The loop: running it (`toolgraph/scheduler.py`)

The scheduler repeats one cycle:

1. Start every call whose dependencies succeeded, within concurrency limits.
2. If a server returns a job handle (MCP Tasks), poll it until it finishes.
3. If a tool asks a question (a 2FA code, a choice), pause that node until answered.
4. Decide retry, verify or fail from the tool's policy.
5. When the graph drains, ask a planner for the next wave of calls (dynamic replanning).

Every state change is committed to SQLite (`toolgraph/store.py`) **before** the scheduler
acts on it: "about to send" → send → "server returned job #5". Every change is also
appended to an event log, which serves as the audit trail and is what the tests check.

Node statuses: `pending`, `sending`, `running`, `waiting_input`, `retry_wait`,
`needs_verification`, `succeeded`, `failed`, `skipped` (a dependency failed).

### Recovery: what if it dies halfway

On restart the scheduler finds calls marked "about to send" with no recorded result.
Their fate is unknown, so the policy decides:

- **read** → send again;
- **keyed write** → re-send with the same key, rejoining the original job;
- **unkeyed write** → do not touch it; mark `needs_verification`.

A tool failure that might have partly applied gets the same treatment. Jobs that were
being polled are polled again (never re-sent), and questions, answers and retry timers
survive restarts.

## How it is tested

Correctness is tested the way databases are tested (`toolgraph/sim/`, `tests/`):

- **Simulated world.** Fake MCP and MCP Tasks servers whose jobs progress on a virtual
  clock and record every applied write in a ledger. Faults are scripted per attempt:
  rejections, lost responses, hangs, partial failures, questions, failed and lost jobs.
- **Deterministic simulation.** An asyncio event loop whose selector, instead of blocking,
  advances a virtual clock. Virtual time therefore only moves when every in-flight call is
  waiting, so hour-long runs finish in milliseconds and replay identically.
- **Crash at every commit point.** A clean run is traced, then repeated once per
  `(point, occurrence)` with a crash injected there, then restarted from the database.
- **Property-based tests.** Hypothesis generates random graphs mixing protocols,
  side-effect classes, resources, failures, questions, planner waves and crash plans.
  Every run must satisfy:
  - a non-idempotent write is applied at most once, and exactly once if it succeeded;
  - nothing starts before everything it depends on has finished;
  - every required data and hazard edge exists, recomputed independently;
  - concurrency limits hold, and final statuses are consistent;
  - every dependent call received the value its ref pointed at.
- **Mutation checks.** Disabling safe-replay checks, key reuse or hazard edges each
  makes the suite fail (Hypothesis is asked to find a counterexample, and does).
- **Real SDK.** Plain MCP calls are tested end to end against the official Python SDK
  (`mcp` 2.3), including `InputRequiredResult` round trips and crash recovery.

```bash
pip install -e ".[dev]"
pytest                                   # 121 tests, about 15 seconds
TOOLGRAPH_EXAMPLES=2000 pytest tests/test_properties.py   # a longer search
```

## Layout

```
src/toolgraph/
  graph.py        nodes, refs, data and hazard edges, cycle checks
  policy.py       side-effect classes and per-tool policies
  store.py        SQLite state and event log, crash points
  scheduler.py    the loop, retry/verify decisions, recovery, replanning
  backend.py      the backend interface and outcome types
  mcp_backend.py  MCP SDK adapter (plain calls, input-required, Tasks)
  sim/            virtual-time loop, simulated servers, harness and invariant checker
examples/travel.py
tests/
```

## Landscape

The individual ideas are not new. The value is in combining them for MCP.

| Area | Existing work | What toolgraph adds |
|---|---|---|
| DAG of tool calls | LLMCompiler (ICML 2024) plans calls as a DAG and runs them in parallel | Side-effect ordering, durability, safe recovery |
| Durable execution | Temporal, Inngest, Restate, Azure Durable Functions, LangGraph checkpoints | Recovery decided per tool by idempotency, not at-least-once retries |
| MCP gateways | Portkey, Cloudflare, Kong, Microsoft, AWS, Docker and others: auth, access control, guardrails, logs | Whole-workflow execution semantics |
| Deduplicating agent calls | agent-middleware (gateway deriving retry keys, never re-running unknown outcomes), OnceOnly (idempotency API), research and guides | Long-running jobs, ordering across a workflow, pauses and human verification |
| The spec | Open MCP issue on duplicate effects after lost responses; idempotency-key proposal discussed | A working client-side implementation today |

## Status and caveats

- MCP Tasks are experimental. `tasks/get`, `tasks/result` and `tasks/cancel` follow the SDK
  types, and their request shapes are checked against a fake session; `tasks/update`
  (sending input to a paused task) is untested against a real server.
- Verification is manual: a person or your code calls `resolve_verification`.
- A plain-MCP `request_state` is subject to the server's own key lifetime; servers using
  a process-local key will reject it after *they* restart.
- Developed and tested locally on Python 3.14; CI covers 3.11–3.14.

## Roadmap

- MCP proxy mode.
- Read-back nodes that resolve `needs_verification` automatically.
- Web inbox for questions and verifications; audit export.
- Cancellation API and run-level deadlines.

## References

- LLMCompiler: <https://icml.cc/virtual/2024/poster/32829>
- MCP Tasks: <https://modelcontextprotocol.io/extensions/tasks/overview.md>
- MCP issue on duplicate effects: <https://github.com/modelcontextprotocol/modelcontextprotocol/issues/3394>
- Temporal, durable MCP: <https://temporal.io/blog/durable-mcp-how-to-give-agentic-systems-superpowers>
- agent-middleware retry keys: <https://github.com/agentmiddleware/agent-middleware/pull/233>
- OnceOnly: <https://www.onceonly.tech/blog/how-to-stop-ai-agents-repeating-actions/>
- MCP gateway features (2026): <https://zuplo.com/blog/what-the-best-mcp-gateways-do-in-2026>

## License

MIT
