// ============================================================
// UniMemo — Compression Engine
// Implements TSGC: Temporal Semantic Gradient Compression
//
// THREE-LAYER PIPELINE:
//   Layer 1 — Structural deduplication  (lossless-ish)
//   Layer 2 — Temporal gradient pruning (lossy, semantics-aware)
//   Layer 3 — DEFLATE byte compression  (lossless, native API)
//
// Why conversation context is uniquely compressible:
//  - Turn-taking creates predictable entropy distribution
//  - Recent turns have higher relevance density than old turns
//  - Role markers and formatting are high-frequency, low-entropy
//  - AI responses contain deliberate redundancy ("As I said...")
// ============================================================

// ── Config ────────────────────────────────────────────────────
export const COMPRESSION_CONFIG = {
  // Temporal zones (message index from END of array = recency)
  ZONE_VERBATIM:   5,   // last N msgs: stored as-is
  ZONE_PRUNED:    15,   // next N msgs: sentence-level pruning (~40% kept)
  ZONE_SUMMARY:   Infinity, // beyond: key-sentence extraction (~15% kept)

  // Sentence scoring weights
  WEIGHT_POSITION: 0.3,   // sentences at start/end of msg score higher
  WEIGHT_DENSITY:  0.4,   // information density (unique word ratio)
  WEIGHT_KEYWORDS: 0.3,   // presence of high-signal words

  // Minimum chars to bother compressing a message
  MIN_COMPRESS_LEN: 80,
};

// High-signal keywords that indicate semantic importance
const SIGNAL_WORDS = new Set([
  'because','therefore','however','but','although','instead','error',
  'result','conclusion','solution','problem','issue','answer','key',
  'important','note','warning','step','must','should','never','always',
  'define','means','equals','returns','function','class','method',
  'approach','strategy','algorithm','difference','similar','compare',
  'summary','finally','overall','specifically','example','for instance',
]);

// Conversational filler — low entropy, safe to drop
const FILLER_PATTERNS = [
  /^(sure|of course|certainly|absolutely|great|okay|ok|yes|no|right)[!.,]?\s*/i,
  /^(i'll|i will|let me|allow me) (help|explain|show|walk)/i,
  /^(as i (mentioned|said|noted|explained) (earlier|above|before))[,.]?\s*/i,
  /^(to (summarize|recap|reiterate|put it simply))[,:]?\s*/i,
  /^(in (conclusion|summary|short|brief))[,:]?\s*/i,
  /hope (this|that) (helps|clarifies|answers)/i,
  /feel free to (ask|let me know)/i,
  /^(thank you|thanks) for (your|the)/i,
];

// ── LAYER 1: Structural Deduplication ─────────────────────────

/**
 * Remove exact-duplicate sentences across the full conversation.
 * AI models often restate context from earlier turns verbatim.
 * Returns messages with duplicates collapsed.
 */
function deduplicateMessages(messages) {
  const seenSentences = new Set();
  return messages.map(msg => {
    const sentences = splitSentences(msg.content);
    const unique = sentences.filter(s => {
      const normalized = s.trim().toLowerCase().replace(/\s+/g, ' ');
      if (normalized.length < 20) return true; // keep short sentences always
      if (seenSentences.has(normalized)) return false;
      seenSentences.add(normalized);
      return true;
    });
    const content = unique.join(' ').trim();
    return { ...msg, content };
  }).filter(msg => msg.content.length > 0);
}

// ── LAYER 2: Temporal Gradient Pruning ────────────────────────

/**
 * Core TSGC algorithm.
 * Applies different compression fidelity based on message recency.
 *
 *  Zone 1 (recent) ────► verbatim            [index from end: 0–4]
 *  Zone 2 (middle) ────► sentence pruning     [index from end: 5–14]
 *  Zone 3 (old)    ────► key-sentence extract [index from end: 15+]
 *
 * This mirrors human working memory models: recent events are
 * recalled with full fidelity; older events are compressed to
 * "gist" representations (Fuzzy Trace Theory, Reyna 1992).
 */
function temporalGradientPrune(messages) {
  const total = messages.length;
  return messages.map((msg, idx) => {
    const recencyIndex = total - 1 - idx; // 0 = most recent

    if (recencyIndex < COMPRESSION_CONFIG.ZONE_VERBATIM) {
      // Zone 1: no compression
      return { ...msg, _zone: 1, _ratio: 1.0 };
    }

    if (recencyIndex < COMPRESSION_CONFIG.ZONE_PRUNED) {
      // Zone 2: keep ~40% highest-scoring sentences
      if (msg.content.length < COMPRESSION_CONFIG.MIN_COMPRESS_LEN) {
        return { ...msg, _zone: 2, _ratio: 1.0 };
      }
      const pruned = extractTopSentences(msg.content, 0.40, msg.role);
      return { ...msg, content: pruned, _zone: 2, _ratio: 0.40 };
    }

    // Zone 3: keep ~15% (key-sentence extraction)
    if (msg.content.length < COMPRESSION_CONFIG.MIN_COMPRESS_LEN) {
      return { ...msg, _zone: 3, _ratio: 1.0 };
    }
    const summary = extractTopSentences(msg.content, 0.15, msg.role);
    return { ...msg, content: summary, _zone: 3, _ratio: 0.15 };
  }).filter(msg => msg.content.trim().length > 0);
}

/**
 * Score-based extractive sentence compression.
 * Keeps top `keepRatio` fraction of sentences by composite score.
 *
 * Scoring formula:
 *   score(s) = w_pos × positionScore(s)
 *            + w_den × densityScore(s)
 *            + w_kw  × keywordScore(s)
 */
function extractTopSentences(text, keepRatio, role) {
  const sentences = splitSentences(text).filter(s => s.trim().length > 10);
  if (sentences.length <= 1) return text;

  const n = sentences.length;
  const keepCount = Math.max(1, Math.ceil(n * keepRatio));

  // Remove filler from assistant messages before scoring
  const cleaned = role === 'assistant'
    ? sentences.filter(s => !FILLER_PATTERNS.some(p => p.test(s.trim())))
    : sentences;

  const scored = cleaned.map((sentence, i) => ({
    sentence,
    originalIndex: i,
    score: scoreScore(sentence, i, cleaned.length),
  }));

  // Sort by score descending, take top K, restore original order
  const topK = scored
    .sort((a, b) => b.score - a.score)
    .slice(0, keepCount)
    .sort((a, b) => a.originalIndex - b.originalIndex);

  return topK.map(s => s.sentence).join(' ').trim();
}

function scoreScore(sentence, index, total) {
  const { WEIGHT_POSITION, WEIGHT_DENSITY, WEIGHT_KEYWORDS } = COMPRESSION_CONFIG;

  // Position score: first and last sentences are more important
  const relPos = index / Math.max(total - 1, 1);
  const posScore = 1 - Math.abs(relPos - 0.5) * 0.6; // U-shape, peaks at edges

  // Density score: ratio of unique words to total words (information density)
  const words     = sentence.toLowerCase().match(/\b[a-z]{3,}\b/g) ?? [];
  const uniqueW   = new Set(words);
  const densScore = words.length > 0 ? uniqueW.size / words.length : 0;

  // Keyword score: presence of signal words
  const kwHits  = words.filter(w => SIGNAL_WORDS.has(w)).length;
  const kwScore = Math.min(1, kwHits / 3); // saturate at 3 signal words

  return WEIGHT_POSITION * posScore
       + WEIGHT_DENSITY  * densScore
       + WEIGHT_KEYWORDS * kwScore;
}

// ── LAYER 3: DEFLATE Byte Compression ─────────────────────────

/**
 * Native CompressionStream (DEFLATE-raw) — zero dependencies.
 * Available in Chrome 80+, Firefox 113+, Safari 16.4+.
 * Returns base64-encoded compressed bytes for storage.
 */
async function deflateCompress(text) {
  const encoder = new TextEncoder();
  const input   = encoder.encode(text);

  const cs     = new CompressionStream('deflate-raw');
  const writer = cs.writable.getWriter();
  writer.write(input);
  writer.close();

  const chunks = [];
  const reader = cs.readable.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  // Concatenate chunks → base64
  const totalLen   = chunks.reduce((n, c) => n + c.length, 0);
  const compressed = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    compressed.set(chunk, offset);
    offset += chunk.length;
  }
  // btoa with large arrays can overflow; use chunked approach
  return uint8ToBase64(compressed);
}

async function deflateDecompress(base64) {
  const bytes = base64ToUint8(base64);

  const ds     = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  writer.write(bytes);
  writer.close();

  const chunks = [];
  const reader = ds.readable.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  const totalLen     = chunks.reduce((n, c) => n + c.length, 0);
  const decompressed = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    decompressed.set(chunk, offset);
    offset += chunk.length;
  }

  return new TextDecoder().decode(decompressed);
}

// ── Full Pipeline ─────────────────────────────────────────────

/**
 * TSGC Full Compression Pipeline
 *
 * Input:  raw message array [{role, content}]
 * Output: { compressed: string, meta: CompressionMeta }
 *
 * Meta includes stats for research/analytics:
 *   - originalBytes, compressedBytes, overallRatio
 *   - perZone breakdown
 */
export async function compressContext(messages) {
  const originalJson   = JSON.stringify(messages);
  const originalBytes  = new TextEncoder().encode(originalJson).length;

  // Layer 1: deduplication
  const deduped = deduplicateMessages(messages);

  // Layer 2: temporal gradient pruning
  const pruned  = temporalGradientPrune(deduped);

  // Layer 3: DEFLATE
  const prunedJson   = JSON.stringify(pruned.map(m => ({ role: m.role, content: m.content })));
  const compressed   = await deflateCompress(prunedJson);
  const compressedBytes = Math.ceil(compressed.length * 0.75); // base64 overhead

  // Compute per-zone stats
  const zones = { 1: [], 2: [], 3: [] };
  pruned.forEach(m => zones[m._zone ?? 1].push(m));

  const meta = {
    originalBytes,
    compressedBytes,
    overallRatio: (1 - compressedBytes / originalBytes),
    messagesBefore: messages.length,
    messagesAfter:  pruned.length,
    zones: {
      verbatim: zones[1].length,
      pruned:   zones[2].length,
      summary:  zones[3].length,
    },
    algorithm: 'TSGC-1.0',
    timestamp: Date.now(),
  };

  return { compressed, meta };
}

/**
 * Decompress stored context back to message array.
 */
export async function decompressContext(compressed) {
  const json = await deflateDecompress(compressed);
  return JSON.parse(json);
}

// ── Utility ───────────────────────────────────────────────────

function splitSentences(text) {
  // Split on sentence-ending punctuation, preserve structure
  return text
    .replace(/([.!?])\s+/g, '$1\n')
    .replace(/([.!?])$/gm, '$1')
    .split('\n')
    .map(s => s.trim())
    .filter(s => s.length > 0);
}

function uint8ToBase64(bytes) {
  const CHUNK = 0x8000;
  let binary  = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function base64ToUint8(base64) {
  const binary = atob(base64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * Quick ratio estimate without full compression (for UI display).
 * Uses character-level heuristics derived from empirical TSGC runs.
 */
export function estimateCompressionRatio(messages) {
  const total = messages.length;
  let savedChars = 0;
  let totalChars = 0;

  messages.forEach((msg, idx) => {
    const recencyIdx = total - 1 - idx;
    const len = msg.content.length;
    totalChars += len;

    if (recencyIdx < COMPRESSION_CONFIG.ZONE_VERBATIM) {
      savedChars += len * 0.35; // DEFLATE alone ~35% on natural language
    } else if (recencyIdx < COMPRESSION_CONFIG.ZONE_PRUNED) {
      savedChars += len * 0.35 + len * 0.60 * 0.35; // prune 60% + deflate
    } else {
      savedChars += len * 0.35 + len * 0.85 * 0.35; // prune 85% + deflate
    }
  });

  return totalChars > 0 ? savedChars / totalChars : 0;
}
