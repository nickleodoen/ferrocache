use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

use crate::index::SemanticIndex;
use crate::wal::Wal;

/// "FERROSNA" — magic bytes identifying a ferrocache snapshot.
pub const MAGIC: u64 = 0x4645_5252_4F53_4E41;
pub const VERSION: u64 = 1;

/// Side-table data for one cached entry, decoupled from `WalEntry` so the
/// on-disk snapshot format can evolve independently of the WAL JSON format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SnapshotEntry {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    pub model_id: String,
    /// Access tracking (M24). Snapshots preserve these across compaction
    /// so a compaction cycle doesn't reset access counts. Pre-M24 snapshots
    /// lack these fields and `#[serde(default)]` zeroes them.
    #[serde(default)]
    pub inserted_at: u64,
    #[serde(default)]
    pub last_accessed_at: u64,
    #[serde(default)]
    pub access_count: u64,
    /// Defensive forward-compat (M25). Snapshots only carry live entries —
    /// `snapshot_entries()` iterates the side-table, which excludes
    /// evicted entries by construction — so this is always `false` in
    /// practice. Included so a future `tombstone-aware` snapshot format
    /// stays bincode-compatible without a version bump.
    #[serde(default)]
    pub tombstone: bool,
}

#[derive(Debug)]
pub struct CompactionResult {
    pub entries_snapshotted: usize,
    pub wal_sequence: u64,
}

/// Derive the snapshot path from a WAL path: `./ferrocache.wal` → `./ferrocache.wal.snap`.
pub fn snapshot_path_for(wal_path: &str) -> PathBuf {
    PathBuf::from(format!("{wal_path}.snap"))
}

fn temp_path_for(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    PathBuf::from(p)
}

/// Write a snapshot atomically: encode to a temp file, fsync, then rename.
/// A crash mid-write leaves any pre-existing snapshot intact.
pub async fn write_snapshot(
    path: &Path,
    entries: &[SnapshotEntry],
    wal_sequence: u64,
) -> Result<()> {
    let tmp = temp_path_for(path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!("failed to create snapshot parent dir {}", parent.display())
        })?;
    }

    let body = bincode::serialize(entries).context("bincode serialize snapshot entries")?;

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp)
        .await
        .with_context(|| format!("failed to create snapshot temp file {}", tmp.display()))?;

    let entry_count = entries.len() as u64;
    file.write_all(&MAGIC.to_le_bytes()).await?;
    file.write_all(&VERSION.to_le_bytes()).await?;
    file.write_all(&wal_sequence.to_le_bytes()).await?;
    file.write_all(&entry_count.to_le_bytes()).await?;
    file.write_all(&body).await?;
    file.sync_all()
        .await
        .context("snapshot temp file sync_all failed")?;
    drop(file);

    tokio::fs::rename(&tmp, path).await.with_context(|| {
        format!(
            "failed to rename snapshot {} -> {}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

/// Read and validate a snapshot. Returns `(entries, wal_sequence)`.
pub async fn read_snapshot(path: &Path) -> Result<(Vec<SnapshotEntry>, u64)> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("failed to read snapshot {}", path.display()))?;
    if bytes.len() < 32 {
        bail!("snapshot file too short ({} bytes)", bytes.len());
    }
    let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    if magic != MAGIC {
        bail!(
            "snapshot magic mismatch: expected {:#x}, got {:#x}",
            MAGIC,
            magic
        );
    }
    let version = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    if version != VERSION {
        bail!("unsupported snapshot version: {version}");
    }
    let wal_sequence = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let entry_count = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

    let body = &bytes[32..];
    let entries: Vec<SnapshotEntry> =
        bincode::deserialize(body).context("bincode deserialize snapshot entries")?;
    if entries.len() as u64 != entry_count {
        return Err(anyhow!(
            "snapshot entry_count mismatch: header={}, decoded={}",
            entry_count,
            entries.len()
        ));
    }
    Ok((entries, wal_sequence))
}

/// Snapshot the in-memory index, then truncate the WAL. Caller is responsible
/// for holding the appropriate locks.
pub async fn compact(
    index: &SemanticIndex,
    wal: &mut Wal,
    snapshot_path: &Path,
    wal_path: &Path,
) -> Result<CompactionResult> {
    let entries = index.snapshot_entries();
    let wal_sequence = wal.current_sequence();
    write_snapshot(snapshot_path, &entries, wal_sequence)
        .await
        .context("write_snapshot failed during compaction")?;
    wal.truncate(wal_path)
        .await
        .context("WAL truncate failed during compaction")?;
    let entries_snapshotted = entries.len();
    tracing::info!(
        snapshotted = entries_snapshotted,
        wal_sequence,
        "compaction complete"
    );
    Ok(CompactionResult {
        entries_snapshotted,
        wal_sequence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;

    fn entry(i: usize) -> SnapshotEntry {
        SnapshotEntry {
            uuid: format!("u-{i}"),
            embedding: vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3],
            response: format!("resp-{i}"),
            query_text: format!("q-{i}"),
            model_id: "m::3".into(),
            inserted_at: 0,
            last_accessed_at: 0,
            access_count: 0,
            tombstone: false,
        }
    }

    #[tokio::test]
    async fn test_write_and_read_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap");
        let entries: Vec<SnapshotEntry> = (0..100).map(entry).collect();

        write_snapshot(&path, &entries, 12_345).await.unwrap();
        let (round_trip, seq) = read_snapshot(&path).await.unwrap();
        assert_eq!(seq, 12_345);
        assert_eq!(round_trip, entries);
    }

    #[tokio::test]
    async fn test_snapshot_atomic_rename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap");
        write_snapshot(&path, &[entry(0)], 1).await.unwrap();

        let tmp = temp_path_for(&path);
        assert!(!tmp.exists(), "temp file must be renamed away");
        assert!(path.exists(), "final file must exist");
    }

    #[tokio::test]
    async fn test_read_corrupt_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap");
        // Header is correct but bincode body is garbage.
        tokio::fs::write(
            &path,
            [
                MAGIC.to_le_bytes().as_slice(),
                VERSION.to_le_bytes().as_slice(),
                0u64.to_le_bytes().as_slice(),
                10u64.to_le_bytes().as_slice(),
                b"this is not bincode at all".as_slice(),
            ]
            .concat(),
        )
        .await
        .unwrap();
        let err = read_snapshot(&path).await.unwrap_err();
        assert!(err.to_string().contains("bincode"));
    }

    #[tokio::test]
    async fn test_read_wrong_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap");
        tokio::fs::write(&path, vec![0u8; 64]).await.unwrap();
        let err = read_snapshot(&path).await.unwrap_err();
        assert!(err.to_string().contains("magic mismatch"));
    }

    #[tokio::test]
    async fn test_read_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist");
        let err = read_snapshot(&path).await.unwrap_err();
        assert!(err.to_string().contains("failed to read snapshot"));
    }

    #[tokio::test]
    async fn test_compact_truncates_wal_and_writes_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snap_path = dir.path().join("test.wal.snap");

        let mut wal = Wal::open(&wal_path).await.unwrap();
        let mut index = SemanticIndex::new(&HnswConfig::default());
        for i in 0..3 {
            let we = crate::wal::WalEntry {
                uuid: format!("u{i}"),
                embedding: vec![i as f32, 0.0, 0.0],
                response: format!("r{i}"),
                query_text: format!("q{i}"),
                model_id: "m::3".into(),
                sequence: 0,
                inserted_at: 0,
                tombstone: false,
            };
            wal.append(&we).await.unwrap();
            index.replay_entry(we).unwrap();
        }
        assert_eq!(wal.current_sequence(), 3);

        let result = compact(&index, &mut wal, &snap_path, &wal_path)
            .await
            .unwrap();
        assert_eq!(result.entries_snapshotted, 3);
        assert_eq!(result.wal_sequence, 3);
        assert!(snap_path.exists());

        // WAL is now empty on disk but the in-memory counter persists.
        let after_wal = Wal::replay(&wal_path).await.unwrap();
        assert!(after_wal.is_empty());
        assert_eq!(wal.current_sequence(), 3);

        // Subsequent appends continue from sequence=4.
        let we = crate::wal::WalEntry {
            uuid: "u4".into(),
            embedding: vec![1.0, 0.0, 0.0],
            response: "r4".into(),
            query_text: "q4".into(),
            model_id: "m::3".into(),
            sequence: 0,
            inserted_at: 0,
            tombstone: false,
        };
        let s = wal.append(&we).await.unwrap();
        assert_eq!(s, 4);
    }
}
