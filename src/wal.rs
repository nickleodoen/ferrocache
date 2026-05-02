use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

pub const LEGACY_NAMESPACE: &str = "legacy::unknown";

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
}

pub struct Wal {
    file: File,
}

impl Wal {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
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
        Ok(Self { file })
    }

    pub async fn append(&mut self, entry: &WalEntry) -> Result<()> {
        let mut line = serde_json::to_vec(entry).context("WAL serialize failed")?;
        line.push(b'\n');
        self.file
            .write_all(&line)
            .await
            .context("WAL write_all failed")?;
        self.file
            .sync_data()
            .await
            .context("WAL sync_data failed")?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    fn sample(uuid: &str, response: &str) -> WalEntry {
        WalEntry {
            uuid: uuid.to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            response: response.to_string(),
            query_text: format!("q-{uuid}"),
            model_id: "test-model::3".to_string(),
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
}
