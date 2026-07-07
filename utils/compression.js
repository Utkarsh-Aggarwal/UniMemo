// ============================================================
// UniMemo — CSGC Compression Engine
// Context-Salience Greedy Compression
//
// THREE-STAGE PIPELINE (from the paper):
//
//   Stage 1 — Redundancy Collapse
//     (a) Same-role merging: consecutive same-role turns concatenated
//     (b) Sliding-window supersession detection (W=15, θ=0.95):
//         · high sim + matching entities  → Collapse (merge duplicate)
//         · high sim + diverging entities → Supersession event
//           older msg → OBSOLETE, newer msg → CORRECTION
//
//   Stage 2 — Salience Scoring
//     σ = 0.50 × TF-IDF + 0.30 × entity_density + 0.20 × recency
//     Post-score modifiers:
//       OBSOLETE   → σ × 0.5   (aggressively down-weight wrong facts)
//       CORRECTION → σ × 1.3   (boost corrections above average)
//
//   Stage 3 — Greedy Budget Selection
//     Sort by σ descending → fill byte budget → re-sort chronologically
//
// Browser adaptations (no Python / ML runtime):
//   · e5-small-v2 cosine sim  → Jaccard similarity on normalised word sets
//   · spaCy NER               → Regex entity extractor (proper nouns,
//                               acronyms, numbers, camelCase, snake_case)
//
// Reference: Aggarwal, U. "CSGC: Compressing Conversation History Without
//   Losing Corrections." Zenodo, 2026. DOI: 10.5281/zenodo.21227878
// ============================================================

// ── Config ────────────────────────────────────────────────────
export const CSGC_CONFIG = {
  BUDGET_RATIO:      0.50,   // fraction of original bytes to target
  WINDOW_SIZE:       15,     // look-back window for supersession detection
  SIM_THRESHOLD:     0.50,   // Jaccard threshold (dense cosine sim 0.95 ≈ Jaccard 0.50)
  W_TFIDF:           0.50,   // salience weight: TF-IDF
  W_ENTITY:          0.30,   // salience weight: entity density
  W_RECENCY:         0.20,   // salience weight: recency
  OBSOLETE_PENALTY:  0.50,   // score multiplier for obsolete messages
  CORRECTION_BOOST:  1.30,   // score multiplier for correction messages
};

// ─────────────────────────────────────────────────────────────
// STAGE 1A — Same-Role Merging
// ─────────────────────────────────────────────────────────────

/**
 * Merges consecutive messages that share the same role into one.
 * Tracks the original first index for later chronological re-sorting.
 */
function mergeConsecutiveSameRole(messages) {
  if (!messages.length) return [];
  const result = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (result.length > 0 && result[result.length - 1].role === msg.role) {
      // Append to current group
      const last = result[result.length - 1];
      last.content += '\n\n' + msg.content;
    } else {
      result.push({
        role:      msg.role,
        content:   msg.content,
        _origIdx:  i,        // position in the ORIGINAL messages array
        _status:   'normal', // 'normal' | 'obsolete' | 'correction'
      });
    }
  }

  return result;
}

// ─────────────────────────────────────────────────────────────
// Entity Extraction (spaCy proxy)
// ─────────────────────────────────────────────────────────────

/**
 * Extracts named entities from text using deterministic regex patterns.
 * Detects: proper nouns, acronyms, numbers/versions, camelCase,
 * snake_case identifiers, and short quoted strings.
 */
function extractEntities(text) {
  const entities = new Set();

  // Proper nouns: Title-cased words (≥2 chars)
  for (const e of (text.match(/\b[A-Z][a-z]{1,}\b/g) ?? []))
    entities.add(e.toLowerCase());

  // Acronyms / ALL-CAPS tokens
  for (const e of (text.match(/\b[A-Z]{2,}\b/g) ?? []))
    entities.add(e.toLowerCase());

  // Numeric literals, versions, floats (e.g. "3.11", "v2", "128")
  for (const e of (text.match(/\b\d+(?:[.\-]\d+)*\b/g) ?? []))
    entities.add(e);

  // camelCase identifiers (e.g. "useEffect", "asyncio")
  for (const e of (text.match(/\b[a-z]+[A-Z][a-zA-Z]+\b/g) ?? []))
    entities.add(e.toLowerCase());

  // snake_case identifiers (e.g. "user_id", "file_path")
  for (const e of (text.match(/\b[a-z][a-z0-9]*_[a-z0-9_]+\b/g) ?? []))
    entities.add(e.toLowerCase());

  // Quoted short strings — often entity references ("MySQL", 'asyncio')
  for (const e of (text.match(/"([^"]{1,40})"|'([^']{1,40})'/g) ?? []))
    entities.add(e.replace(/['"]/g, '').toLowerCase().trim());

  return entities;
}

/** Deep equality check for two Sets. */
function setsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const v of a) if (!b.has(v)) return false;
  return true;
}

// ─────────────────────────────────────────────────────────────
// Jaccard Similarity (e5-small-v2 proxy)
// ─────────────────────────────────────────────────────────────

/** Returns a normalised word-set for Jaccard computation. */
function wordSet(text) {
  return new Set(
    text.toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 2)
  );
}

/**
 * Jaccard similarity between two texts on their word sets.
 * |A ∩ B| / |A ∪ B|  — ranges [0, 1].
 */
function jaccardSimilarity(text1, text2) {
  const a = wordSet(text1);
  const b = wordSet(text2);
  if (a.size === 0 && b.size === 0) return 1;
  let inter = 0;
  for (const w of a) if (b.has(w)) inter++;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

// ─────────────────────────────────────────────────────────────
// STAGE 1B — Sliding-Window Supersession Detection
// ─────────────────────────────────────────────────────────────

/**
 * Scans every message against its W predecessors.
 *   sim ≥ θ AND entity sets match  → Collapse (remove older duplicate)
 *   sim ≥ θ AND entity sets differ → Supersession event
 *     older = OBSOLETE, newer = CORRECTION
 */
function detectSupersession(messages) {
  const WINDOW = CSGC_CONFIG.WINDOW_SIZE;
  const THRESH = CSGC_CONFIG.SIM_THRESHOLD;
  const toRemove = new Set();

  for (let i = 0; i < messages.length; i++) {
    const start = Math.max(0, i - WINDOW);
    for (let j = start; j < i; j++) {
      if (toRemove.has(j)) continue;

      const sim = jaccardSimilarity(messages[i].content, messages[j].content);
      if (sim < THRESH) continue;

      const ents_i = extractEntities(messages[i].content);
      const ents_j = extractEntities(messages[j].content);

      if (setsEqual(ents_i, ents_j)) {
        // Exact repetition — collapse: absorb j (keep i, the newer copy)
        toRemove.add(j);
      } else {
        // Entity divergence — supersession pivot
        messages[j]._status = 'obsolete';    // older fact is now wrong
        messages[i]._status = 'correction';  // newer fact is the truth
      }
    }
  }

  return messages.filter((_, idx) => !toRemove.has(idx));
}

/** Full Stage 1 pipeline. */
function stage1RedundancyCollapse(messages) {
  const merged    = mergeConsecutiveSameRole(messages);
  const collapsed = detectSupersession(merged);
  return collapsed;
}

// ─────────────────────────────────────────────────────────────
// STAGE 2 — Salience Scoring
// ─────────────────────────────────────────────────────────────

// ── TF-IDF ──────────────────────────────────────────────────

/**
 * Computes a per-message TF-IDF salience score.
 * TF  = word count / total words in message (normalised frequency)
 * IDF = log(N / (1 + df(word)))
 * Score per message = mean TF×IDF across unique words in that message.
 */
function computeTfIdf(messages) {
  const N = messages.length;

  // Tokenize each message
  const tokenized = messages.map(m =>
    m.content
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 2)
  );

  // Document frequency
  const df = new Map();
  tokenized.forEach(words => {
    for (const w of new Set(words))
      df.set(w, (df.get(w) ?? 0) + 1);
  });

  // Per-message TF-IDF score
  return tokenized.map(words => {
    if (words.length === 0) return 0;
    const tf = new Map();
    for (const w of words)
      tf.set(w, (tf.get(w) ?? 0) + 1 / words.length);
    let score = 0;
    for (const [w, tfVal] of tf) {
      const idf = Math.log(N / (1 + (df.get(w) ?? 0)));
      score += tfVal * idf;
    }
    return Math.max(0, score);
  });
}

// ── Entity Density ───────────────────────────────────────────

/** Entity density = entity count / word count. */
function entityDensity(text) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  if (words.length === 0) return 0;
  return extractEntities(text).size / words.length;
}

// ── Min-Max Normalisation ────────────────────────────────────

function minMaxNorm(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  if (max === min) return arr.map(() => 0.5);
  return arr.map(v => (v - min) / (max - min));
}

// ── Full Stage 2 ─────────────────────────────────────────────

/**
 * Assigns composite salience scores and applies supersession modifiers.
 *
 *   σ_i = 0.50 × TF-IDF_norm + 0.30 × entity_density_norm + 0.20 × recency
 *
 *   OBSOLETE   → σ × 0.5
 *   CORRECTION → σ × 1.3
 */
function stage2SalienceScore(messages) {
  const n = messages.length;
  if (n === 0) return [];

  const { W_TFIDF, W_ENTITY, W_RECENCY, OBSOLETE_PENALTY, CORRECTION_BOOST } = CSGC_CONFIG;

  // Raw component scores
  const rawTfIdf   = computeTfIdf(messages);
  const rawEntity  = messages.map(m => entityDensity(m.content));
  const rawRecency = messages.map((_, i) => n === 1 ? 1 : i / (n - 1)); // 0→oldest, 1→newest

  // Normalise TF-IDF and entity density; recency is already [0,1]
  const normTfIdf  = minMaxNorm(rawTfIdf);
  const normEntity = minMaxNorm(rawEntity);

  return messages.map((msg, i) => {
    let sigma = W_TFIDF  * normTfIdf[i]
              + W_ENTITY * normEntity[i]
              + W_RECENCY * rawRecency[i];

    // Apply supersession modifiers
    if (msg._status === 'obsolete')   sigma *= OBSOLETE_PENALTY;
    if (msg._status === 'correction') sigma *= CORRECTION_BOOST;

    return { ...msg, _sigma: Math.min(sigma, 2.0) }; // cap at 2× (correction boost can exceed 1)
  });
}

// ─────────────────────────────────────────────────────────────
// STAGE 3 — Greedy Budget Selection
// ─────────────────────────────────────────────────────────────

/** UTF-8 byte size of one message as it will be stored. */
function msgByteLen(msg) {
  return new TextEncoder().encode(JSON.stringify({ role: msg.role, content: msg.content })).length;
}

/**
 * Greedy knapsack: sort messages by σ descending, fill byte budget,
 * then re-sort selected messages chronologically.
 */
function stage3GreedySelect(messages, budgetBytes) {
  if (!messages.length) return [];

  // Sort by sigma descending (highest-salience first)
  const sorted = [...messages].sort((a, b) => b._sigma - a._sigma);

  const selected = [];
  let usedBytes  = 0;

  for (const msg of sorted) {
    const size = msgByteLen(msg);
    if (usedBytes + size <= budgetBytes) {
      selected.push(msg);
      usedBytes += size;
    }
    // Continue scanning — a smaller message later may still fit
  }

  // Re-sort selected set by original index to restore conversational order
  selected.sort((a, b) => a._origIdx - b._origIdx);

  return selected;
}

// ─────────────────────────────────────────────────────────────
// DEFLATE (Layer 3 byte compression — Chrome 80+, native)
// ─────────────────────────────────────────────────────────────

async function deflate(text) {
  const bytes  = new TextEncoder().encode(text);
  const cs     = new CompressionStream('deflate-raw');
  const writer = cs.writable.getWriter();
  writer.write(bytes); writer.close();
  const chunks = []; const reader = cs.readable.getReader();
  while (true) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); }
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0; for (const c of chunks) { out.set(c, off); off += c.length; }
  return uint8ToB64(out);
}

async function inflate(b64) {
  const bytes  = b64ToUint8(b64);
  const ds     = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  writer.write(bytes); writer.close();
  const chunks = []; const reader = ds.readable.getReader();
  while (true) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); }
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let off = 0; for (const c of chunks) { out.set(c, off); off += c.length; }
  return new TextDecoder().decode(out);
}

function uint8ToB64(bytes) {
  const CHUNK = 0x8000; let bin = '';
  for (let i = 0; i < bytes.length; i += CHUNK)
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  return btoa(bin);
}

function b64ToUint8(b64) {
  const bin = atob(b64); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

/**
 * CSGC Full Compression Pipeline.
 *
 * @param {Array<{role: string, content: string}>} messages - Raw conversation
 * @param {number} [budgetRatio] - Target fraction of original bytes (default 0.50)
 * @returns {{ compressed: string, meta: object }}
 */
export async function csgcCompress(messages, budgetRatio = CSGC_CONFIG.BUDGET_RATIO) {
  if (!messages || messages.length === 0) {
    return {
      compressed: '',
      meta: {
        originalBytes: 0, compressedBytes: 0, overallRatio: 0,
        messagesBefore: 0, messagesAfter: 0,
        obsoleteCount: 0, correctionCount: 0,
        algorithm: 'CSGC-1.0', budgetRatio,
      }
    };
  }

  const originalJson  = JSON.stringify(messages);
  const originalBytes = new TextEncoder().encode(originalJson).length;
  const budgetBytes   = Math.floor(originalBytes * budgetRatio);

  // ── Stage 1: Redundancy Collapse ──────────────────────────
  const collapsed = stage1RedundancyCollapse(messages);

  // ── Stage 2: Salience Scoring ──────────────────────────────
  const scored = stage2SalienceScore(collapsed);

  // Collect supersession stats before selecting
  const obsoleteCount   = scored.filter(m => m._status === 'obsolete').length;
  const correctionCount = scored.filter(m => m._status === 'correction').length;

  // ── Stage 3: Greedy Budget Selection ───────────────────────
  const selected = stage3GreedySelect(scored, budgetBytes);

  // ── DEFLATE ─────────────────────────────────────────────────
  const outputJson      = JSON.stringify(
    selected.map(m => ({ role: m.role, content: m.content }))
  );
  const compressed      = await deflate(outputJson);
  // base64 encodes ~4/3 bytes, so actual bytes ≈ compressed.length × 0.75
  const compressedBytes = Math.ceil(compressed.length * 0.75);

  const meta = {
    originalBytes,
    compressedBytes,
    overallRatio:    Math.max(0, 1 - compressedBytes / originalBytes),
    messagesBefore:  messages.length,
    messagesAfter:   selected.length,
    obsoleteCount,
    correctionCount,
    algorithm:       'CSGC-1.0',
    budgetRatio,
  };

  return { compressed, meta };
}

/**
 * Decompresses a stored CSGC payload back to a message array.
 *
 * @param {string} compressed - Base64-encoded DEFLATE payload
 * @returns {Promise<Array<{role: string, content: string}>>}
 */
export async function csgcDecompress(compressed) {
  const json = await inflate(compressed);
  return JSON.parse(json);
}

/**
 * Quick ratio estimate without running full compression.
 * Used by the popup for lightweight progress display.
 */
export function estimateCompressionRatio(messages) {
  const total = messages.length;
  if (total === 0) return 0;
  // Approximate: CSGC at 50% budget + ~35% DEFLATE gain ≈ ~67% saved
  const budgetSavings = 1 - CSGC_CONFIG.BUDGET_RATIO; // 50%
  const deflateSavings = 0.35;
  return Math.min(0.80, budgetSavings + CSGC_CONFIG.BUDGET_RATIO * deflateSavings);
}
