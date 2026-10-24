// #4457: control-plane sidecar system cores (rawAnthropicMessage —
// mid-conversation tool deltas / effort markers) are ORDINARY foldable
// messages. The pre-#4457 never-fold carve (protected.ts) leaked them
// unbounded and, combined with the verbatim emission (#3991), produced
// positionally-illegal wire layouts that 400 the session forever (#4292
// family). Folding is reversible (decompress, tools[] re-states current
// toolsets), so nothing is lost — this pins the whole chain: predicate
// flip, apply-side coverage (no exclusion warning), legal emission.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createCore } from "../src/compress.js";
import { createInitialState } from "../src/state.js";
import { assignRefs } from "../src/refs.js";
import { prune } from "../src/prune.js";
import { anthropicToCore, coreToAnthropic } from "../src/wire/anthropic.js";
import { hasUnrecoverableMediaPayload } from "../src/protected.js";
import { defaultCountTokens } from "../src/tokenize.js";
import type { Config, CoreMessage } from "../src/types.js";

function config(): Config {
  return {
    tiers: { enabled: false },
    nudge: {
      maxContextLimitPct: 0.55, minContextLimitPct: 0.45, frequency: 5,
      iterationThreshold: 15, force: "soft", growthRatio: 0.05,
      growthFloor: 6000, growthCap: 50000, minGrowthFloor: 5000,
      minGrowthRatio: 0.45, emergencyThresholdPct: 0.98,
    },
    promotionThreshold: 5,
    truncate: { threshold: 1 },
    merge: { maxSummaryLength: 3000, minOldGenBlocks: 3 },
    compress: { minCompressRange: 1000, maxSummaryLength: 20000, minSummaryLength: 1 },
    protectedTools: [],
    preserveRecentMessages: 0,
    preserveRecentTokens: 0,
    modelContextLimit: 100000,
  };
}

const TOOL_DELTA_SYSTEM = {
  role: "system",
  content: [
    {
      type: "text",
      text: 'Updated system prompt section "tools":\n<tools>…</tools>',
    },
    { type: "tool_addition", name: "search", input_schema: { type: "object" } },
  ],
};

function baseMessages(): CoreMessage[] {
  return anthropicToCore({
    messages: [
      { role: "user", content: [{ type: "text", text: "q1 " + "x".repeat(400) }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "a1 " + "y".repeat(4000) }],
      },
      TOOL_DELTA_SYSTEM,
      {
        role: "assistant",
        content: [{ type: "text", text: "a2 " + "y".repeat(4000) }],
      },
      { role: "user", content: [{ type: "text", text: "q2" }] },
    ],
  } as never).msgs;
}

test("#4457: tool-carrying sidecar system is not protected media", () => {
  const msgs = baseMessages();
  const sys = msgs.find((m) => m.role === "system");
  assert.ok(sys);
  assert.ok((sys as { rawAnthropicMessage?: unknown }).rawAnthropicMessage);
  assert.equal(hasUnrecoverableMediaPayload(sys), false);
});

test("#4457: fold covering the system applies clean, emission legal", () => {
  const msgs = baseMessages();
  const state = createInitialState();
  state.messageRefs = assignRefs(msgs, {
    existing: state.messageRefs,
    nextIndex: 1,
  }).map;
  const core = createCore({ countTokens: defaultCountTokens });
  const applied = core.applyCompression({
    ranges: [
      {
        startRef: "m00001",
        endRef: "m00004",
        summary: "covered early turns incl. the tool-delta system",
      },
    ],
    messages: msgs,
    state,
    config: config(),
  });
  assert.deepEqual(applied.result.errors, []);
  assert.equal(
    applied.result.warnings.find((w) => w.includes("Excluded")),
    undefined,
    applied.result.warnings.join(" | "),
  );
  const view = prune(msgs, applied.state);
  assert.equal(
    view.find(
      (m) => (m as { rawAnthropicMessage?: unknown }).rawAnthropicMessage,
    ),
    undefined,
    "the sidecar system itself must be gone from the folded view",
  );
  // the fold anchor is a no-sidecar core system — kernel-synthesized, and
  // coreToAnthropic re-voices it to user, so it cannot orphan either.
  assert.ok(view.some((m) => m.role === "system" && m.id.startsWith("acp_summary_")));
  assert.ok(view.length < msgs.length);
  const out = coreToAnthropic(view);
  for (let j = 0; j < out.length; j++) {
    if (out[j]!.role === "system") {
      const next = out[j + 1];
      if (!(!next || next.role === "assistant")) {
        assert.fail(`orphaned system at ${j}: ${JSON.stringify(next)}`);
      }
    }
  }
  assert.ok(
    JSON.stringify(out).includes("covered early turns"),
    "the fold summary must survive into the emission",
  );
});
