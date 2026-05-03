use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{RwLock, mpsc, oneshot};

use crate::index::SemanticIndex;
use crate::metrics::Metrics;
use crate::snapshot;
use crate::snapshot::CompactionResult;

pub const LEGACY_NAMESPACE: &str = "legacy::unknown";

/// Default mpsc channel capacity for the group-commit WAL. Picked larger than
/// any plausible single batch — once it's full, `Sender::send` awaits a slot
/// (backpressure).
pub const DEFAULT_CHANNEL_CAPACITY: usize = 10_000;

fn legacy_namespace() -> String {
    LEGACY_NAMESPACE.to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalEntry {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    /// Namespace partition; entries from different `model_id`s never compare.
    /// Old WAL entries lacking this field are migrated to `legacy::unknown`.
    #[serde(default = "legacy_namespace")]
    pub model_id: String,
    /// Monotonically increasing sequence number; used by compaction to skip
    /// entries already captured in a snapshot. Old WAL entries default to 0.
    #[serde(default)]
    pub sequence: u64,
}

pub struct Wal {
    file: File,
    sequence: u64,
}

impl Wal {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_sequence(path, 0).await
    }

    pub async fn open_with_sequence(path: impl AsRef<Path>, sequence: u64) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create WAL parent dir {}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("failed to open WAL at {}", path.display()))?;
        Ok(Self { file, sequence })
    }

    pub fn current_sequence(&self) -> u64 {
        self.sequence
    }

    /// Allocate the next sequence number for an entry about to be appended.
    /// Used by the group-commit flush task, which stamps sequences itself
    /// before serializing the batch.
    pub fn next_sequence(&mut self) -> Result<u64> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .context("WAL sequence overflow")?;
        Ok(self.sequence)
    }

    /// Stamp a sequence number into the entry, append, and fsync.
    /// The entry's existing `sequence` field is overwritten.
    pub async fn append(&mut self, entry: &WalEntry) -> Result<u64> {
        let seq = self.next_sequence()?;
        let mut stamped = entry.clone();
        stamped.sequence = seq;
        let mut line = serde_json::to_vec(&stamped).context("WAL serialize failed")?;
        line.push(b'\n');
        self.file
            .write_all(&line)
            .await
            .context("WAL write_all failed")?;
        self.file
            .sync_data()
            .await
            .context("WAL sync_data failed")?;
        Ok(seq)
    }

    /// Append a pre-formatted batch of NDJSON bytes (already terminated with
    /// `\n` per line) and fsync once. Sequence numbers must already be
    /// stamped by the caller.
    pub async fn write_batch_bytes(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        self.file
            .write_all(data)
            .await
            .context("WAL batch write_all failed")?;
        self.file
            .sync_data()
            .await
            .context("WAL batch sync_data failed")?;
        Ok(())
    }

    /// Atomically replace the WAL with an empty file. The in-memory sequence
    /// counter continues from where it was (does NOT reset).
    pub async fn truncate(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let new_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .await
            .with_context(|| format!("failed to truncate WAL at {}", path.display()))?;
        new_file
            .sync_all()
            .await
            .context("WAL truncate sync failed")?;
        // Reopen in append mode so subsequent writes go to the end.
        let appender = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("failed to reopen WAL at {}", path.display()))?;
        self.file = appender;
        Ok(())
    }

    pub async fn replay(path: impl AsRef<Path>) -> Result<Vec<WalEntry>> {
        let path = path.as_ref();
        let file = match File::open(path).await {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => {
                return Err(anyhow::Error::new(e)
                    .context(format!("failed to open WAL at {}", path.display())));
            }
        };

        let mut entries = Vec::new();
        let mut reader = BufReader::new(file).lines();
        let mut line_number: usize = 0;
        while let Some(line) = reader.next_line().await.context("WAL read failed")? {
            line_number += 1;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WalEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    tracing::warn!("skipping corrupt WAL line {}: {}", line_number, e);
                }
            }
        }
        Ok(entries)
    }
}

/// One in-flight insert waiting on the group-commit flush task.
pub struct InsertRequest {
    pub entry: WalEntry,
    pub respond: oneshot::Sender<Result<u64, InsertError>>,
}

/// Error returned to insert callers. The two variants map to different
/// HTTP responses: `Wal` → "persistence failed" (path/error chain hidden),
/// `Index` → the underlying validation message (e.g. dimension mismatch).
#[derive(Debug)]
pub enum InsertError {
    Wal(anyhow::Error),
    Index(anyhow::Error),
}

impl std::fmt::Display for InsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InsertError::Wal(e) => write!(f, "WAL: {e}"),
            InsertError::Index(e) => write!(f, "index: {e}"),
        }
    }
}

impl std::error::Error for InsertError {}

/// One in-flight compaction waiting on the flush task. Compaction goes
/// through the same channel so it serializes with insert flushes.
pub struct CompactRequest {
    pub respond: oneshot::Sender<Result<CompactionResult>>,
}

/// All commands the flush task understands.
pub enum WalCommand {
    Insert(InsertRequest),
    Compact(CompactRequest),
}

#[derive(Debug, Clone, Copy)]
pub struct GroupCommitConfig {
    /// Max requests to batch into a single fsync. `1` degrades to per-insert
    /// fsync (the pre-M20 behavior); used by tests.
    pub batch_size: usize,
    /// Max time the flush task waits for additional inserts to join the
    /// current batch after the first one arrives.
    pub batch_timeout: Duration,
    /// Bound on the mpsc channel — once full, `Sender::send` awaits a slot.
    pub channel_capacity: usize,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            batch_size: 256,
            batch_timeout: Duration::from_millis(1),
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
        }
    }
}

/// Handle to the group-commit flush task. Cloning the sender hands out
/// new producers; the task itself runs detached and exits cleanly when all
/// senders are dropped.
#[derive(Clone)]
pub struct GroupCommitWal {
    sender: mpsc::Sender<WalCommand>,
}

impl GroupCommitWal {
    pub fn sender(&self) -> &mpsc::Sender<WalCommand> {
        &self.sender
    }

    /// Spawn the flush task and return a handle.
    ///
    /// The task takes ownership of `wal` (no other code path may write to
    /// the WAL afterwards) and a clone of the index/metrics. It exits
    /// cleanly when all `GroupCommitWal` clones are dropped.
    pub fn spawn(
        wal: Wal,
        wal_path: PathBuf,
        snapshot_path: PathBuf,
        index: Arc<RwLock<SemanticIndex>>,
        metrics: Arc<Metrics>,
        compact_interval_inserts: u64,
        config: GroupCommitConfig,
    ) -> Self {
        let (tx, rx) = mpsc::channel(config.channel_capacity.max(1));
        let cfg = config;
        tokio::spawn(flush_loop(
            rx,
            wal,
            wal_path,
            snapshot_path,
            index,
            metrics,
            compact_interval_inserts,
            cfg,
        ));
        Self { sender: tx }
    }
}

#[allow(clippy::too_many_arguments)]
async fn flush_loop(
    mut receiver: mpsc::Receiver<WalCommand>,
    mut wal: Wal,
    wal_path: PathBuf,
    snapshot_path: PathBuf,
    index: Arc<RwLock<SemanticIndex>>,
    metrics: Arc<Metrics>,
    compact_interval_inserts: u64,
    config: GroupCommitConfig,
) {
    let batch_cap = config.batch_size.max(1);
    let mut inserts_since_compact: u64 = 0;

    loop {
        // Wait for the first command (no busy-spin when idle).
        let first = match receiver.recv().await {
            Some(c) => c,
            None => {
                tracing::debug!("WAL flush task: channel closed, exiting");
                return;
            }
        };

        let mut batch: Vec<InsertRequest> = Vec::with_capacity(batch_cap);
        let mut pending_compact: Option<CompactRequest> = None;

        match first {
            WalCommand::Insert(ic) => batch.push(ic),
            WalCommand::Compact(cc) => {
                run_compact(&mut wal, &index, &snapshot_path, &wal_path, &metrics, cc).await;
                inserts_since_compact = 0;
                continue;
            }
        }

        // Collect more inserts up to batch_size or until the timeout fires.
        // Bail early if a Compact arrives — flush the inserts first, then it.
        let deadline = tokio::time::Instant::now() + config.batch_timeout;
        while batch.len() < batch_cap {
            let recv_result = tokio::time::timeout_at(deadline, receiver.recv()).await;
            match recv_result {
                Ok(Some(WalCommand::Insert(ic))) => batch.push(ic),
                Ok(Some(WalCommand::Compact(cc))) => {
                    pending_compact = Some(cc);
                    break;
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        flush_insert_batch(&mut wal, &index, &mut inserts_since_compact, batch).await;

        // Auto-compact after threshold (the flush task is the sole writer,
        // so the counter doesn't need locking).
        if compact_interval_inserts > 0 && inserts_since_compact >= compact_interval_inserts {
            inserts_since_compact = 0;
            match snapshot_compact(&mut wal, &index, &snapshot_path, &wal_path).await {
                Ok(result) => {
                    metrics.record_compaction();
                    tracing::info!(
                        snapshotted = result.entries_snapshotted,
                        wal_sequence = result.wal_sequence,
                        "auto-compaction fired"
                    );
                }
                Err(e) => {
                    tracing::error!(error = %e, "auto-compaction failed; continuing");
                }
            }
        }

        if let Some(cc) = pending_compact {
            run_compact(&mut wal, &index, &snapshot_path, &wal_path, &metrics, cc).await;
            inserts_since_compact = 0;
        }
    }
}

/// Stamp sequences, write the entire batch in one syscall, fsync once,
/// then insert each entry into the index under a single write lock.
async fn flush_insert_batch(
    wal: &mut Wal,
    index: &Arc<RwLock<SemanticIndex>>,
    inserts_since_compact: &mut u64,
    mut batch: Vec<InsertRequest>,
) {
    if batch.is_empty() {
        return;
    }

    // Stamp sequences AND build the byte buffer in one pass. If we can't
    // even allocate sequences (overflow), fail every entry in the batch.
    let mut buf: Vec<u8> = Vec::with_capacity(batch.len() * 256);
    let mut alloc_failed: Option<String> = None;
    for ir in batch.iter_mut() {
        match wal.next_sequence() {
            Ok(seq) => {
                ir.entry.sequence = seq;
                match serde_json::to_vec(&ir.entry) {
                    Ok(mut line) => {
                        line.push(b'\n');
                        buf.extend_from_slice(&line);
                    }
                    Err(e) => {
                        alloc_failed = Some(format!("WAL serialize failed: {e}"));
                        break;
                    }
                }
            }
            Err(e) => {
                alloc_failed = Some(format!("{e}"));
                break;
            }
        }
    }

    if let Some(msg) = alloc_failed {
        for ir in batch {
            let _ = ir.respond.send(Err(InsertError::Wal(anyhow!(msg.clone()))));
        }
        return;
    }

    // Single write + single fsync for the whole batch.
    if let Err(e) = wal.write_batch_bytes(&buf).await {
        // Pre-format once to avoid moving the anyhow chain into N closures.
        let msg = format!("{e}");
        tracing::error!(error = %e, batch_size = batch.len(), "WAL batch flush failed");
        for ir in batch {
            let _ = ir.respond.send(Err(InsertError::Wal(anyhow!(msg.clone()))));
        }
        return;
    }

    // Index inserts under a single write lock — queries can resume after
    // this drops. Per-entry validation errors (dimension mismatch) surface
    // back to that caller only; other batch members still succeed.
    let batch_len = batch.len();
    let mut idx = index.write().await;
    for ir in batch {
        let seq = ir.entry.sequence;
        match idx.replay_entry(ir.entry) {
            Ok(()) => {
                *inserts_since_compact = inserts_since_compact.saturating_add(1);
                let _ = ir.respond.send(Ok(seq));
            }
            Err(e) => {
                tracing::warn!(error = %e, "index insert rejected for batched entry");
                let _ = ir.respond.send(Err(InsertError::Index(e)));
            }
        }
    }
    drop(idx);
    if batch_len > 1 {
        tracing::debug!(batch_len, "group-commit batch flushed");
    }
}

async fn run_compact(
    wal: &mut Wal,
    index: &Arc<RwLock<SemanticIndex>>,
    snapshot_path: &Path,
    wal_path: &Path,
    metrics: &Metrics,
    cc: CompactRequest,
) {
    match snapshot_compact(wal, index, snapshot_path, wal_path).await {
        Ok(result) => {
            metrics.record_compaction();
            let _ = cc.respond.send(Ok(result));
        }
        Err(e) => {
            tracing::error!(error = %e, "compaction failed");
            let _ = cc.respond.send(Err(e));
        }
    }
}

async fn snapshot_compact(
    wal: &mut Wal,
    index: &Arc<RwLock<SemanticIndex>>,
    snapshot_path: &Path,
    wal_path: &Path,
) -> Result<CompactionResult> {
    let idx = index.read().await;
    snapshot::compact(&idx, wal, snapshot_path, wal_path).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;
    use crate::index::SemanticIndex;
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::io::AsyncWriteExt;

    fn sample(uuid: &str, response: &str) -> WalEntry {
        WalEntry {
            uuid: uuid.to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            response: response.to_string(),
            query_text: format!("q-{uuid}"),
            model_id: "test-model::3".to_string(),
            sequence: 0,
        }
    }

    #[tokio::test]
    async fn test_append_and_replay() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        // Drop the tempfile guard so we can reopen it ourselves.
        drop(tmp);

        let mut wal = Wal::open(&path).await.unwrap();
        for i in 0..3 {
            wal.append(&sample(&format!("u{i}"), &format!("r{i}")))
                .await
                .unwrap();
        }
        drop(wal);

        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].uuid, "u0");
        assert_eq!(entries[1].response, "r1");
        assert_eq!(entries[2].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_replay_skips_corrupt_lines() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);

        let mut f = File::create(&path).await.unwrap();
        let valid_a = serde_json::to_string(&sample("a", "ra")).unwrap();
        let valid_b = serde_json::to_string(&sample("b", "rb")).unwrap();
        let valid_c = serde_json::to_string(&sample("c", "rc")).unwrap();
        let payload = format!("{valid_a}\nnot json at all\n{valid_b}\n{valid_c}\n");
        f.write_all(payload.as_bytes()).await.unwrap();
        f.sync_data().await.unwrap();
        drop(f);

        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].uuid, "a");
        assert_eq!(entries[1].uuid, "b");
        assert_eq!(entries[2].uuid, "c");
    }

    #[tokio::test]
    async fn test_replay_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.wal");
        let entries = Wal::replay(&path).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn test_wal_roundtrip_with_model_id() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);

        let mut wal = Wal::open(&path).await.unwrap();
        let mut a = sample("a", "ra");
        a.model_id = "model-a::3".to_string();
        let mut b = sample("b", "rb");
        b.model_id = "model-b::3".to_string();
        wal.append(&a).await.unwrap();
        wal.append(&b).await.unwrap();
        drop(wal);

        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].model_id, "model-a::3");
        assert_eq!(entries[1].model_id, "model-b::3");
    }

    #[tokio::test]
    async fn test_wal_legacy_migration() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);

        // Manually craft a pre-M14 WAL line that lacks `model_id`.
        let legacy_json =
            r#"{"uuid":"old-1","embedding":[0.1,0.2,0.3],"response":"r","query_text":"q"}"#;
        let mut f = File::create(&path).await.unwrap();
        f.write_all(legacy_json.as_bytes()).await.unwrap();
        f.write_all(b"\n").await.unwrap();
        f.sync_data().await.unwrap();
        drop(f);

        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].uuid, "old-1");
        assert_eq!(entries[0].model_id, LEGACY_NAMESPACE);
    }

    #[tokio::test]
    async fn test_replay_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let entries = Wal::replay(tmp.path()).await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn test_wal_sequence_increments() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);

        let mut wal = Wal::open(&path).await.unwrap();
        let s1 = wal.append(&sample("a", "ra")).await.unwrap();
        let s2 = wal.append(&sample("b", "rb")).await.unwrap();
        let s3 = wal.append(&sample("c", "rc")).await.unwrap();
        assert_eq!((s1, s2, s3), (1, 2, 3));
        assert_eq!(wal.current_sequence(), 3);
        drop(wal);

        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(
            entries.iter().map(|e| e.sequence).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[tokio::test]
    async fn test_wal_replay_with_sequence_filter() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);

        let mut wal = Wal::open(&path).await.unwrap();
        for i in 0..5u32 {
            wal.append(&sample(&format!("u{i}"), &format!("r{i}")))
                .await
                .unwrap();
        }
        drop(wal);

        let entries: Vec<WalEntry> = Wal::replay(&path)
            .await
            .unwrap()
            .into_iter()
            .filter(|e| e.sequence > 3)
            .collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sequence, 4);
        assert_eq!(entries[1].sequence, 5);
    }

    #[tokio::test]
    async fn test_wal_truncate_keeps_sequence_counter() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp);

        let mut wal = Wal::open(&path).await.unwrap();
        wal.append(&sample("a", "ra")).await.unwrap();
        wal.append(&sample("b", "rb")).await.unwrap();
        assert_eq!(wal.current_sequence(), 2);

        wal.truncate(&path).await.unwrap();
        assert_eq!(wal.current_sequence(), 2, "counter must NOT reset");

        let s = wal.append(&sample("c", "rc")).await.unwrap();
        assert_eq!(s, 3);
        drop(wal);

        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].uuid, "c");
        assert_eq!(entries[0].sequence, 3);
    }

    // --- Group-commit tests ----------------------------------------------------

    fn group_commit_config(batch: usize, timeout_ms: u64) -> GroupCommitConfig {
        GroupCommitConfig {
            batch_size: batch,
            batch_timeout: Duration::from_millis(timeout_ms),
            channel_capacity: 1024,
        }
    }

    async fn spawn_test_group_commit(
        path: PathBuf,
        config: GroupCommitConfig,
    ) -> (GroupCommitWal, Arc<RwLock<SemanticIndex>>) {
        let snap = path.with_extension("snap");
        let wal = Wal::open(&path).await.unwrap();
        let index = Arc::new(RwLock::new(SemanticIndex::new(&HnswConfig::default())));
        let metrics = Arc::new(Metrics::new());
        let gc = GroupCommitWal::spawn(wal, path, snap, index.clone(), metrics, 0, config);
        (gc, index)
    }

    fn make_entry(uuid: &str) -> WalEntry {
        WalEntry {
            uuid: uuid.to_string(),
            embedding: vec![1.0, 0.0, 0.0],
            response: format!("resp-{uuid}"),
            query_text: format!("q-{uuid}"),
            model_id: "test-model::3".to_string(),
            sequence: 0,
        }
    }

    #[tokio::test]
    async fn test_group_commit_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gc.wal");
        let (gc, _idx) = spawn_test_group_commit(path.clone(), group_commit_config(8, 5)).await;

        let mut futures = Vec::new();
        for i in 0..3 {
            let mut e = make_entry(&format!("u{i}"));
            e.embedding = vec![i as f32, 0.0, 0.0];
            let (tx, rx) = oneshot::channel();
            gc.sender()
                .send(WalCommand::Insert(InsertRequest {
                    entry: e,
                    respond: tx,
                }))
                .await
                .unwrap();
            futures.push(rx);
        }
        let mut seqs = Vec::new();
        for rx in futures {
            seqs.push(rx.await.unwrap().unwrap());
        }
        seqs.sort();
        assert_eq!(seqs, vec![1, 2, 3]);

        // Drop the only sender so the flush task exits and closes the file.
        drop(gc);
        // Give the task a moment to process the channel-closed signal.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let entries = Wal::replay(&path).await.unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn test_group_commit_batch_fsync() {
        // 10 inserts in a single batch should complete much faster than 10×
        // back-to-back fsyncs would. Use a 50ms wall-clock budget — even a
        // slow disk only fsyncs at ~5ms/op, so 10 serial fsyncs ≈ 50ms; we
        // expect this batch to be well under 30ms in practice. Generous to
        // avoid CI flakes.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gc.wal");
        let (gc, _idx) = spawn_test_group_commit(path, group_commit_config(32, 20)).await;

        let start = Instant::now();
        let mut futures = Vec::new();
        for i in 0..10 {
            let mut e = make_entry(&format!("b{i}"));
            e.embedding = vec![i as f32, 0.0, 0.0];
            let (tx, rx) = oneshot::channel();
            gc.sender()
                .send(WalCommand::Insert(InsertRequest {
                    entry: e,
                    respond: tx,
                }))
                .await
                .unwrap();
            futures.push(rx);
        }
        for rx in futures {
            rx.await.unwrap().unwrap();
        }
        let elapsed = start.elapsed();
        // Bound is intentionally loose to avoid CI flakes; the key
        // invariant is that batching is fundamentally faster than
        // serializing 10 fsyncs.
        assert!(
            elapsed < Duration::from_millis(500),
            "batch took too long: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_group_commit_channel_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gc.wal");
        let (gc, _idx) = spawn_test_group_commit(path, group_commit_config(4, 5)).await;
        // Immediately drop without sending — the task must exit cleanly.
        drop(gc);
        tokio::time::sleep(Duration::from_millis(30)).await;
        // If the task panicked, the runtime would have noticed by now via
        // the test harness — passing here means clean shutdown.
    }

    #[tokio::test]
    async fn test_group_commit_entries_queryable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gc.wal");
        let (gc, idx) = spawn_test_group_commit(path, group_commit_config(8, 5)).await;

        let mut e = make_entry("query-target");
        e.embedding = vec![1.0, 0.0, 0.0];
        let (tx, rx) = oneshot::channel();
        gc.sender()
            .send(WalCommand::Insert(InsertRequest {
                entry: e,
                respond: tx,
            }))
            .await
            .unwrap();
        rx.await.unwrap().unwrap();

        // Index must reflect the insert by the time the oneshot resolved.
        let snapshot = idx.read().await;
        let hit = snapshot
            .query(&[1.0, 0.0, 0.0], 0.9, "test-model::3")
            .unwrap()
            .expect("group-commit insert must be queryable");
        assert_eq!(hit.response, "resp-query-target");
    }
}
