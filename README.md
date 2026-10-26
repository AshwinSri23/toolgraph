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
from toolgraph.mcp_backend import client_options, connect

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
servers = {
    "travel": lambda: Client("https://travel.example/mcp", **client_options()),
    "mail": lambda: Client("https://mail.example/mcp", **client_options()),
}
async with connect(servers) as backend:      # reconnects after transport failures
    sched = Scheduler(Store("runs.db"), backend, reg, run_id="trip-1")
    sched.start(graph)              # no-op if the run already exists (i.e. after a crash)
    report = await sched.run()      # resumes from the database on restart
```

`client_options()` declares the MCP Tasks extension, so servers can answer long jobs
with a task handle, and declares form elicitation, so questions arrive in band where
they can be saved. `policies_from_annotations(await client.list_tools())` builds
starting policies from the servers' `readOnlyHint` and `idempotentHint` annotations.
Those are hints, so register keyed writes yourself.

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
pytest                                   # 122 tests, about 15 seconds
TOOLGRAPH_EXAMPLES=2000 pytest tests/test_properties.py   # a longer search
```

### Real-world scenarios (`scenarios/`)

The simulation proves the logic. These scenarios check the system as it would really be
deployed: the agent is its own OS process, servers are separate processes reached over
Streamable HTTP or stdio, crashes are real `SIGKILL`s, and lost responses are real
dropped TCP connections. Each scenario audits the *server's* records afterwards (a
payments ledger, an outbox, files on disk), not toolgraph's own view.

| Scenario | What happens | Result |
|---|---|---|
| **Online shop, kill storm** (`kill_storm.py`) | 40 orders; each is a lookup, a keyed charge, a status update and an unkeyed receipt email, against a separate HTTP server. The agent is SIGKILLed at random moments and restarted, again and again. | 4 seeds, 77 kills, 160 orders: every order charged exactly once, every email sent exactly once; 50 unknown emails settled by read-back, none re-sent blind |
| **Flaky network** (`flaky_network.py`, `chaos_proxy.py`) | A TCP proxy lets half the charge and email requests reach the server, then cuts the connection before the response returns. | 3 seeds, 77 lost responses, 60 orders: no double charges or duplicate emails; keyed charges re-sent with the same key, unknown outcomes settled by looking up the charge or email |
| **Research agent** (`agent.py research`) | Real MCP Tasks on the official `server-everything` reference server: polling, a task that pauses `input_required`, the answer sent with `tasks/update`. Killed after each task is created, while it waits for input, and after the answer is sent. | Every kill resumed the existing task (one task per node, never two); the clarification reached the server's final report |
| **Coding agent** (`files_sweep.py`) | Edits real files through the official `server-filesystem`. One edit is not idempotent (`retries = 1` to `retries = 1 + 1`; a replay would read `1 + 1 + 1`). SIGKILLed at **every** commit point of a run. | 24 of 24 points: the file always ends `retries = 1 + 1`, reads see the edit, directory searches wait for the writes |

```bash
cd scenarios && npm install && cd ..     # pinned reference servers
pytest -m realworld                      # 9 tests, about 2 minutes
python scenarios/kill_storm.py --kills 25 --orders 40 --seed 1
```

Running against real servers found three bugs that the simulation could not:

1. **Wrong Tasks protocol.** The first version followed the 2025-11-25 experimental
   Tasks design (client asks for a task, separate `tasks/result`). Current servers use
   the 2026-07-28 Tasks extension instead: the server returns `resultType: "task"`, the
   result arrives inside `tasks/get`, and the client must declare the extension. Over
   HTTP every `tasks/*` request also needs an `Mcp-Name` header. The backend now speaks
   the real protocol.
2. **One lost response killed the whole client.** In the SDK's Streamable HTTP
   transport, a POST whose connection drops raises inside the transport's task group
   and tears down the entire `mcp.Client`, failing every other call in flight.
   `toolgraph.mcp_backend.connect` keeps each client in its own supervisor task, treats
   the in-flight calls' outcomes as unknown, and reconnects on the next call.
3. **A read-back that could never succeed.** The filesystem server returns file text
   wrapped as `{"content": ...}`, so the first read-back checked dictionary keys and
   would always have answered "not applied". That's harmless here only because a
   replayed edit fails to match, but it's exactly the kind of mistake that would cause a
   duplicate in a less forgiving tool.

## Layout

```
src/toolgraph/
  graph.py        nodes, refs, data and hazard edges, cycle checks
  policy.py       side-effect classes and per-tool policies
  store.py        SQLite state and event log, crash points
  scheduler.py    the loop, retry/verify decisions, recovery, replanning
  backend.py      the backend interface and outcome types
  mcp_backend.py  MCP SDK adapter (plain calls, input-required, Tasks, reconnecting connections)
  sim/            virtual-time loop, simulated servers, harness and invariant checker
examples/travel.py
scenarios/        real-world scenarios: shop server, chaos proxy, agent, kill storms
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

- MCP Tasks are an extension, and the extension is still evolving. toolgraph implements
  the 2026-07-28 Tasks extension and is tested against the official reference server
  (`tasks/get`, `tasks/update`, `input_required`); `tasks/cancel` is sent on job timeouts
  but has only been checked against a fake session. The older 2025-11-25 task layer is
  not supported.
- Verification is your code's job: a person, or a read-back like the ones in
  `scenarios/agent.py`, calls `resolve_verification`. A read-back is only safe once the
  server can no longer apply the original request. After a client crash this is
  normally immediate, but a server that queues requests could apply one later.
- With the SDK's HTTP transport, one dropped connection fails every call in flight on
  that client. They're recovered safely, but each one uses up a retry attempt.
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
