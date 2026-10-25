/**
 * #4666: duplicate message rows must be cleaned up by the v8 migration on
 * BOTH backends.
 *
 * Pre-#4390 wireIndex-split generations left libraries with several rows per
 * logical message (same session_id + msg_id, different content_hash, identical
 * identity once id/wireIndex are stripped). The sqlite v8 migration step
 * (src/storage/schema.ts) backfilled identity AND collapsed each group to its
 * lowest-seq arrival, handing the dropped rows' CAS refs back. The MySQL
 * upgrade (src/storage/mysql.ts init()) did the backfill but skipped the
 * collapse — affected libraries replay every duplicate forever, and #4390's
 * historian sweep can never remove them (both rows "hit" the incoming
 * identity set).
 *
 * This suite pins, on both lanes:
 *  - the v8 upgrade collapses each (session_id, msg_id, identity) group to
 *    its lowest seq and refunds every dropped row's CAS refs (media + text);
 *  - the MySQL data step is gated on the bili_meta schema_version STAMP, not
 *    column presence — a crash between the ALTERs and the backfill re-runs on
 *    the next boot instead of stranding rows at identity='';
 *  - the decRef floor (refcount > 0) keeps an already-corrupt ledger from
 *    going negative during migration;
 *  - verifyProjections covers RETIRED rows too (a tombstone-only mutation
 *    keeps projections intact — drift there must fail the check).
 *
 * The sqlite lane always runs; the mysql lane needs BILI_MYSQL_TEST_DSN (CI:
 * the mysql-contract job, same service as storage-backend-contract) and runs
 * in an isolated throwaway database so the shared CI database is untouched.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { UnifiedStore, sha256Hex } from "../src/storage/store.ts";
import { openSqliteDatabase } from "../src/storage/driver.ts";
import { STORAGE_DDL_V1_FOR_TESTS } from "../src/storage/schema.ts";
import { identityOfStoredContent } from "../src/storage/identity.ts";
import { planIngest, type LegacySessionLike } from "../src/storage/ingest.ts";
import { openBackend, type StorageBackend } from "../src/storage/backend.ts";

process.env.BILI_PERSIST_ZSTD = "0";
// Same hermeticity pin as the other storage suites: a developer machine with
// storage settings in its real config must not leak into these tests.
const hermetic = fs.mkdtempSync(path.join(os.tmpdir(), "bili-4666-hermetic-"));
process.env.BILI_CONFIG_FILE = path.join(hermetic, "no-config.json");

const MYSQL_DSN = process.env.BILI_MYSQL_TEST_DSN ?? "";

type RawDb = ReturnType<typeof openSqliteDatabase>["db"];

function tmpdir(prefix: string): string {
    return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function rmBestEffort(p: string): void {
    try {
        fs.rmSync(p, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    } catch {
        /* leaked temp dir on a Windows runner is harmless */
    }
}

/** The dirty fixture: four (msg_id, identity) groups over one session.
 *  mA: dup pair (seq 5 kept, seq 9 dropped, media on the dropped row)
 *  mB: triple (seq 2 kept, seq 7 + seq 11 dropped)
 *  mC: unique clean row (survives, identity backfilled)
 *  mD: dup pair whose dropped row points at a content row ALREADY at
 *      refcount 0 (corrupt ledger — the decRef floor must hold it at 0). */
const FIXTURE = (() => {
    const cA1 = JSON.stringify({ id: "mA", role: "user", contentType: "text", text: "alpha", wireIndex: 1 });
    const cA2 = JSON.stringify({ id: "mA", role: "user", contentType: "text", text: "alpha", wireIndex: 9 });
    const cB1 = JSON.stringify({ id: "mB", role: "user", contentType: "text", text: "beta", wireIndex: 0 });
    const cB2 = JSON.stringify({ id: "mB", role: "user", contentType: "text", text: "beta", wireIndex: 3 });
    const cB3 = JSON.stringify({ id: "mB", role: "user", contentType: "text", text: "beta", wireIndex: 4 });
    const cC1 = JSON.stringify({ id: "mC", role: "assistant", contentType: "text", text: "gamma" });
    const cD1 = JSON.stringify({ id: "mD", role: "user", contentType: "text", text: "delta", wireIndex: 1 });
    const cD2 = JSON.stringify({ id: "mD", role: "user", contentType: "text", text: "delta", wireIndex: 2 });
    const MEDIA = "media-bytes-on-the-dropped-row";
    return { cA1, cA2, cB1, cB2, cB3, cC1, cD1, cD2, MEDIA };
})();

function putContentRow(db: RawDb, body: string, refcount: number): string {
    const bytes = Buffer.from(body, "utf8");
    const hash = sha256Hex(bytes);
    db.prepare("INSERT INTO content (hash, byte_len, refcount, algo, bytes) VALUES (?, ?, ?, 'sha256', ?)")
        .run(hash, bytes.byteLength, refcount, body);
    return hash;
}

function seedDirtyFixture(db: RawDb, session: string): Record<string, string> {
    const hashes: Record<string, string> = {};
    for (const key of ["cA1", "cA2", "cB1", "cB2", "cB3", "cC1", "cD1", "MEDIA"] as (keyof typeof FIXTURE)[]) {
        hashes[key] = putContentRow(db, FIXTURE[key], 1);
    }
    // The corrupt-ledger row: correct bytes/hash, refcount already 0.
    hashes.cD2 = putContentRow(db, FIXTURE.cD2, 0);
    const insMsg = db.prepare(
        "INSERT INTO messages (session_id, seq, msg_id, ref, ref_num, role, kind, content_hash) VALUES (?, ?, ?, NULL, NULL, ?, 'message', ?)",
    );
    const rows: [number, string, string, keyof typeof FIXTURE][] = [
        [1, "mC", "assistant", "cC1"],
        [2, "mB", "user", "cB1"],
        [3, "mD", "user", "cD1"],
        [4, "mD", "user", "cD2"],
        [5, "mA", "user", "cA1"],
        [7, "mB", "user", "cB2"],
        [9, "mA", "user", "cA2"],
        [11, "mB", "user", "cB3"],
    ];
    for (const [seq, msgId, role, key] of rows) insMsg.run(session, seq, msgId, role, hashes[key]);
    db.prepare("INSERT INTO message_media (session_id, seq, media_hash) VALUES (?, ?, ?)").run(session, 9, hashes.MEDIA);
    return hashes;
}

/** Builds a faithful pre-v8 library: the current DDL minus the two v8
 *  columns (regex-strip discipline of the schema-v2 proof in
 *  storage-unified.test.ts), dirty fixture rows, user_version = 7. */
function buildLegacyDirtySqliteDir(): string {
    const dir = tmpdir("bili-4666-sqlite-legacy-");
    const { db } = openSqliteDatabase(path.join(dir, "index.db"));
    const ddl = STORAGE_DDL_V1_FOR_TESTS
        .replace(/^\s*identity\s+TEXT NOT NULL DEFAULT '',\n/m, "")
        .replace(/^\s*retired_at\s+INTEGER,\n/m, "");
    db.exec(ddl);
    seedDirtyFixture(db, "s1");
    db.exec("PRAGMA user_version = 7");
    db.close();
    return dir;
}

function refcountOf(db: RawDb, hash: string): number | null {
    const r = db.prepare("SELECT refcount FROM content WHERE hash = ?").get(hash) as { refcount: number } | undefined;
    return r?.refcount ?? null;
}

test("#4666 sqlite: v8 migration collapses wireIndex-split duplicates and refunds their CAS refs", () => {
    const dir = buildLegacyDirtySqliteDir();
    let store: UnifiedStore | undefined;
    let check: RawDb | undefined;
    try {
        store = UnifiedStore.open(path.join(dir, "index.db"));
        check = openSqliteDatabase(path.join(dir, "index.db")).db;

        const ver = check.prepare("PRAGMA user_version").get() as { user_version: number };
        assert.equal(ver.user_version, 8, "stamped to v8");

        // Exactly the lowest-seq arrival of each (msg_id, identity) group.
        const survivors = check
            .prepare("SELECT seq, msg_id, content_hash, identity FROM messages WHERE session_id = 's1' ORDER BY seq")
            .all() as { seq: number; msg_id: string; content_hash: string; identity: string }[];
        assert.deepEqual(
            survivors.map((r) => r.seq),
            [1, 2, 3, 5],
            "groups collapse to their first arrival; everything else is gone",
        );
        const bySeq = new Map(survivors.map((r) => [r.seq, r]));
        assert.equal(bySeq.get(5)!.content_hash, sha256Hex(FIXTURE.cA1), "kept generation is the lowest-seq bytes");
        assert.equal(bySeq.get(5)!.identity, identityOfStoredContent(FIXTURE.cA1), "identity backfilled from stored bytes");
        assert.equal(bySeq.get(2)!.identity, identityOfStoredContent(FIXTURE.cB1));
        assert.equal(bySeq.get(3)!.identity, identityOfStoredContent(FIXTURE.cD1));
        assert.equal(bySeq.get(1)!.identity, identityOfStoredContent(FIXTURE.cC1), "unique row backfilled too");

        // Refunds: every dropped row handed its CAS refs back (text + media).
        const h = (body: string) => sha256Hex(body);
        assert.equal(refcountOf(check, h(FIXTURE.cA2)), 0, "dropped dup refunded");
        assert.equal(refcountOf(check, h(FIXTURE.cB2)), 0);
        assert.equal(refcountOf(check, h(FIXTURE.cB3)), 0);
        assert.equal(refcountOf(check, h(FIXTURE.MEDIA)), 0, "media mapping of the dropped row released");
        assert.equal(refcountOf(check, h(FIXTURE.cA1)), 1, "kept rows untouched");
        assert.equal(refcountOf(check, h(FIXTURE.cB1)), 1);
        assert.equal(refcountOf(check, h(FIXTURE.cD1)), 1);
        assert.equal(refcountOf(check, h(FIXTURE.cC1)), 1);
        // Corrupt-ledger floor: the dropped row pointed at a content row
        // already at 0 — the floor holds it at 0 instead of driving -1.
        assert.equal(refcountOf(check, h(FIXTURE.cD2)), 0, "decRef floor: never negative");

        const mediaLeft = check.prepare("SELECT COUNT(*) AS n FROM message_media WHERE session_id = 's1'").get() as { n: number };
        assert.equal(mediaLeft.n, 0, "no media mappings survive on dropped rows");

        // Replay view: the survivor is byte-exact.
        const msgs = store.getMessages("s1");
        assert.equal(msgs.length, 4, "replay shows one row per logical message");
        const kept = msgs.find((m) => m.contentHash === h(FIXTURE.cA1))!;
        assert.equal(kept.content, FIXTURE.cA1, "kept generation replays verbatim");
    } finally {
        check?.close();
        store?.close();
        rmBestEffort(dir);
        fs.rmSync(hermetic, { recursive: true, force: true });
    }
});

interface CoreMsg { id: string; role: string; contentType: string; text: string }
const mkMsg = (id: string, text: string): CoreMsg => ({ id, role: "user", contentType: "text", text });

function mkSession(id: string, msgs: CoreMsg[]): LegacySessionLike {
    return {
        id,
        createdAt: 1_000,
        lastSeen: 2_000,
        meta: { protocol: "openai", label: `label-${id}` },
        stats: {},
        metadata: {},
        state: { blocks: [], nextBlockId: 1, nextRunId: 1, messageRefs: { byRaw: {}, byRef: {} } },
        blockContents: new Map(),
        lastMessages: msgs as unknown as LegacySessionLike["lastMessages"],
        lastMessagesFolded: false,
        pluginSnapshot: undefined,
        contentStore: undefined,
    } as unknown as LegacySessionLike;
}

/** Both store shapes (sync sqlite / async mysql) behind one helper. */
type Ingester = {
    ingestSession(input: ReturnType<typeof planIngest>, delta?: unknown, opts?: { completeLog?: boolean }): boolean | Promise<boolean>;
};

async function retireSecondMessage(store: Ingester, sid: string): Promise<void> {
    const m1 = mkMsg("vm1", "first ".repeat(20));
    const m2 = mkMsg("vm2", "second ".repeat(20));
    assert.equal(await store.ingestSession(planIngest(mkSession(sid, [m1, m2])), undefined, { completeLog: true }), true, "initial complete ingest applies");
    assert.equal(await store.ingestSession(planIngest(mkSession(sid, [m1])), undefined, { completeLog: true }), true, "shorter authoritative log applies");
}

test("#4666 sqlite: verifyProjections catches projection drift on RETIRED rows", async () => {
    const dir = tmpdir("bili-4666-sqlite-retired-");
    let store: UnifiedStore | undefined;
    try {
        store = UnifiedStore.open(path.join(dir, "index.db"));
        const sid = "sess-retired";
        await retireSecondMessage(store, sid);

        const all = store.getMessages(sid, { includeRetired: true });
        assert.equal(all.length, 2, "rewind left the second message as a tombstone");
        assert.equal(all.filter((m) => m.retiredAt !== null).length, 1, "exactly one retired row");

        // Control: intact projections pass even WITH a retired row present.
        const clean = store.verifyProjections(sid);
        assert.ok(clean.ok, `intact projections verify (${clean.ok ? clean.checks : clean.error})`);

        // Corrupt a projection on the RETIRED row only.
        const db = (store as unknown as { db: { prepare(q: string): { run(...a: unknown[]): unknown } } }).db;
        db.prepare("UPDATE messages SET role = 'system' WHERE session_id = ? AND retired_at IS NOT NULL").run(sid);
        const broken = store.verifyProjections(sid);
        assert.equal(broken.ok, false, "retired-row drift must fail the check");
        assert.match(broken.ok ? "" : broken.error, /\.role/, "failure names the drifted field");
    } finally {
        store?.close();
        rmBestEffort(dir);
        fs.rmSync(hermetic, { recursive: true, force: true });
    }
});

// ---------------------------------------------------------------------------
// MySQL lane: same pins against a real MySQL 8.0 (BILI_MYSQL_TEST_DSN).
// Runs in a throwaway database — the surgery below mutates table shapes.
// ---------------------------------------------------------------------------

test("#4666 mysql: v8 upgrade collapses duplicates, crash-safe stamp gate, retired-row verification",
    { skip: MYSQL_DSN.length === 0 ? "set BILI_MYSQL_TEST_DSN to run" : false },
    async (t) => {
        const driver = await import("mysql2/promise");
        const dbname = `bili_i4666_${Date.now()}`;
        const admin = driver.createPool(MYSQL_DSN);
        await admin.query(`CREATE DATABASE ${dbname} CHARACTER SET utf8mb4`);
        const url = new URL(MYSQL_DSN);
        url.pathname = `/${dbname}`;
        const dsn = url.toString();
        const raw = driver.createPool(dsn);
        t.after(async () => {
            await raw.end();
            await admin.query(`DROP DATABASE IF EXISTS ${dbname}`);
            await admin.end();
        });

        const q = async (sql: string, args: unknown[] = []): Promise<Record<string, unknown>[]> => {
            const [rows] = (await raw.query(sql, args)) as unknown as [Record<string, unknown>[]];
            return rows;
        };
        const stamp = async (): Promise<string> => {
            const rows = await q("SELECT meta_value FROM bili_meta WHERE meta_key = 'schema_version'");
            return String(rows[0]?.meta_value ?? "");
        };
        const refcount = async (hash: string): Promise<number | null> => {
            const rows = await q("SELECT refcount FROM content WHERE hash = ?", [hash]);
            return rows.length > 0 ? Number(rows[0].refcount) : null;
        };
        const h = (body: string): string => sha256Hex(body);

        let be: StorageBackend | undefined;
        try {
            // Fresh install lands at v8.
            be = await openBackend("mysql", dsn);
            await be.init?.();
            assert.equal(await stamp(), "8", "fresh install stamped v8");

            // Downgrade to the legacy v7 shape: drop the two columns + the
            // index, rewind the stamp, plant the dirty fixture.
            await raw.query("ALTER TABLE messages DROP INDEX idx_messages_identity, DROP COLUMN identity, DROP COLUMN retired_at");
            await raw.query("UPDATE bili_meta SET meta_value = '7' WHERE meta_key = 'schema_version'");
            const hashes: Record<string, string> = {};
            const insContent = "INSERT INTO content (hash, byte_len, refcount, algo, bytes) VALUES (?, ?, ?, 'sha256', ?)";
            for (const key of ["cA1", "cA2", "cB1", "cB2", "cB3", "cC1", "cD1", "MEDIA"] as (keyof typeof FIXTURE)[]) {
                hashes[key] = h(FIXTURE[key]);
                await raw.query(insContent, [hashes[key], Buffer.byteLength(FIXTURE[key]), 1, FIXTURE[key]]);
            }
            hashes.cD2 = h(FIXTURE.cD2);
            await raw.query(insContent, [hashes.cD2, Buffer.byteLength(FIXTURE.cD2), 0, FIXTURE.cD2]);
            const insMsg = "INSERT INTO messages (session_id, seq, msg_id, ref, ref_num, role, kind, content_hash) VALUES (?, ?, ?, NULL, NULL, ?, 'message', ?)";
            const rows: [number, string, string, string][] = [
                [1, "mC", "assistant", FIXTURE.cC1],
                [2, "mB", "user", FIXTURE.cB1],
                [3, "mD", "user", FIXTURE.cD1],
                [4, "mD", "user", FIXTURE.cD2],
                [5, "mA", "user", FIXTURE.cA1],
                [7, "mB", "user", FIXTURE.cB2],
                [9, "mA", "user", FIXTURE.cA2],
                [11, "mB", "user", FIXTURE.cB3],
            ];
            for (const [seq, msgId, role, body] of rows) {
                await raw.query(insMsg, ["s1", seq, msgId, role, h(body)]);
            }
            await raw.query("INSERT INTO message_media (session_id, seq, media_hash) VALUES ('s1', 9, ?)", [hashes.MEDIA]);

            // Fresh boot migrates: ALTERs + transactional backfill/dedup.
            await be.close();
            be = await openBackend("mysql", dsn);
            await be.init?.();
            assert.equal(await stamp(), "8", "upgrade stamped v8");

            const survivors = await q("SELECT seq, msg_id, content_hash, identity FROM messages WHERE session_id = 's1' ORDER BY seq");
            assert.deepEqual(
                survivors.map((r) => Number(r.seq)),
                [1, 2, 3, 5],
                "groups collapse to their first arrival (the pre-fix tree kept all eight)",
            );
            const bySeq = new Map(survivors.map((r) => [Number(r.seq), r]));
            assert.equal(String(bySeq.get(5)!.content_hash), hashes.cA1, "kept generation is the lowest-seq bytes");
            assert.equal(String(bySeq.get(5)!.identity), identityOfStoredContent(FIXTURE.cA1), "identity backfilled from stored bytes");
            assert.equal(String(bySeq.get(1)!.identity), identityOfStoredContent(FIXTURE.cC1), "unique row backfilled too");

            assert.equal(await refcount(hashes.cA2), 0, "dropped dup refunded");
            assert.equal(await refcount(hashes.cB2), 0);
            assert.equal(await refcount(hashes.cB3), 0);
            assert.equal(await refcount(hashes.MEDIA), 0, "media mapping of the dropped row released");
            assert.equal(await refcount(hashes.cA1), 1, "kept rows untouched");
            assert.equal(await refcount(hashes.cD1), 1);
            assert.equal(await refcount(hashes.cC1), 1);
            assert.equal(await refcount(hashes.cD2), 0, "decRef floor: corrupt ledger held at 0, not driven negative");
            const mediaLeft = await q("SELECT COUNT(*) AS n FROM message_media WHERE session_id = 's1'");
            assert.equal(Number(mediaLeft[0].n), 0, "no media mappings survive on dropped rows");

            const replay = await be.getMessages("s1");
            assert.equal(replay.length, 4, "replay shows one row per logical message");
            assert.equal(replay.find((m) => m.contentHash === hashes.cA1)!.content, FIXTURE.cA1, "kept generation replays verbatim");

            // Crash-recovery arm: columns present but the stamp still says
            // 7 (crash between the ALTERs and the data step) plus unbackfilled
            // legacy rows — the gate keys on the STAMP, so init re-runs.
            const cE1 = JSON.stringify({ id: "mE", role: "user", contentType: "text", text: "eps", wireIndex: 1 });
            const cE2 = JSON.stringify({ id: "mE", role: "user", contentType: "text", text: "eps", wireIndex: 2 });
            await raw.query("UPDATE bili_meta SET meta_value = '7' WHERE meta_key = 'schema_version'");
            await raw.query(insContent, [h(cE1), Buffer.byteLength(cE1), 1, cE1]);
            await raw.query(insContent, [h(cE2), Buffer.byteLength(cE2), 1, cE2]);
            const insMsgV8 = "INSERT INTO messages (session_id, seq, msg_id, ref, ref_num, role, kind, content_hash, identity) VALUES (?, ?, ?, NULL, NULL, ?, 'message', ?, '')";
            await raw.query(insMsgV8, ["s2", 5, "mE", "user", h(cE1)]);
            await raw.query(insMsgV8, ["s2", 9, "mE", "user", h(cE2)]);

            await be.close();
            be = await openBackend("mysql", dsn);
            await be.init?.();
            assert.equal(await stamp(), "8", "re-run completed and stamped v8");
            const eRows = await q("SELECT seq, identity FROM messages WHERE session_id = 's2' ORDER BY seq");
            assert.deepEqual(eRows.map((r) => Number(r.seq)), [5], "unbackfilled dup pair collapsed on the re-run");
            assert.notEqual(String(eRows[0].identity), "", "identity backfilled on the re-run");

            // Retired-row projection coverage (same pins as the sqlite lane).
            const sid = "sess-mysql-retired";
            await retireSecondMessage(be, sid);
            const retired = await q("SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND retired_at IS NOT NULL", [sid]);
            assert.equal(Number(retired[0].n), 1, "rewind left the second message as a tombstone");
            const clean = await be.verifyProjections(sid);
            assert.ok(clean.ok, `intact projections verify with a retired row present (${clean.ok ? clean.checks : clean.error})`);
            await raw.query("UPDATE messages SET role = 'system' WHERE session_id = ? AND retired_at IS NOT NULL", [sid]);
            const broken = await be.verifyProjections(sid);
            assert.equal(broken.ok, false, "retired-row drift must fail the check");
            assert.match(broken.ok ? "" : broken.error, /\.role/, "failure names the drifted field");
        } finally {
            await be?.close();
        }
    },
);
