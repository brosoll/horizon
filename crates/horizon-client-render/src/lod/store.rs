//! Far terrain from the chunks that exist, kept on disk: the source of
//! distant terrain where the game has no generator (a server), or where the
//! player asked for no speculative generation (Far Terrain: Existing Chunks),
//! so what was explored stays visible past the render distance, across
//! sessions.
//!
//! One folder per server (or world) and dimension. A region of 32 x 32
//! chunks is two files: `r.X.Z.lod`, an append-only log of each chunk's 256
//! columns (the latest record wins; compacted when it is mostly stale), and
//! `r.X.Z.coarse`, one merged column per chunk, rewritten as it changes. The
//! coarser levels (2 x 2 chunks up to the whole region) are mipped from those
//! when a chunk is kept, as Voxy's `WorldUpdater` does, so a far cell is a
//! lookup rather than a merge of every chunk under it. A `meta` file names the
//! block and biome tables the columns index; when they change the folder is
//! cleared rather than drawn with the wrong blocks.
use super::{Column, ReducedColumns, Span, Surface};
use crate::{
    fast_hash::FxHashMap,
    streaming::{LodDetail, LodSampler, LodSource},
};
use horizon_world::registry::BlockStateId;
use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

const REGION: i32 = 32;
/// A record without its columns' hash (written by earlier versions).
const RECORD: u8 = 0xc7;
const HEADER: usize = 13;
/// A record with the hash of its encoded columns: an unchanged chunk kept
/// again is not written again.
const RECORD_HASHED: u8 = 0xc8;
const HASHED_HEADER: usize = 21;
const COARSE_MAGIC: &[u8; 6] = b"HLODC1";
/// Regions whose coarse columns stay in memory (about 130 KiB each when full).
const KEPT_REGIONS: usize = 256;
/// Chunks a sampler keeps decoded, with their reductions: a level-4 tile
/// reads 16 x 16 chunks.
const KEPT_CHUNKS: usize = 320;
/// Records written to a region before its coarse file is rewritten.
const FLUSH_EVERY: u32 = 64;
/// Log bytes past which a log mostly of replaced records is compacted.
const COMPACT_AFTER: usize = if cfg!(test) { 16 << 10 } else { 256 << 10 };
/// Log files kept open for appending.
const OPEN_LOGS: usize = 64;
/// A chunk's encoded columns are far below this; anything larger is damage.
const MAX_CHUNK_BYTES: usize = 4 << 20;
const MAX_SPANS: usize = 4096;
/// Mip levels inside a region: cells 32, 64, 128, 256 and 512 blocks wide.
const LEVELS: usize = 5;

type Columns = Arc<[Column; 256]>;

#[derive(Clone, Copy)]
struct Entry {
    offset: u64,
    length: u32,
    /// `hash` of the encoded columns; 0 when the record has none.
    hash: u64,
}

struct Region {
    /// Each chunk's columns merged into one (`Column::merge`), by
    /// `(z & 31) * 32 + (x & 31)`.
    coarse: Vec<Option<Column>>,
    /// `pyramid[l]`: cells `32 << l` blocks wide, `16 >> l` to a side, each
    /// the 2 x 2 cells below it merged (`merge4`).
    pyramid: [Vec<Option<Column>>; LEVELS],
    /// Each chunk's latest record in the log.
    index: FxHashMap<u16, Entry>,
    unflushed: u32,
}

impl Region {
    fn empty() -> Self {
        Self {
            coarse: vec![None; (REGION * REGION) as usize],
            pyramid: std::array::from_fn(|level| vec![None; (16 >> level) * (16 >> level)]),
            index: FxHashMap::default(),
            unflushed: 0,
        }
    }
    fn child(&self, level: usize, x: usize, z: usize) -> &Option<Column> {
        if level == 0 {
            &self.coarse[z * REGION as usize + x]
        } else {
            &self.pyramid[level - 1][z * (16 >> (level - 1)) + x]
        }
    }
    /// Mips the cells above the chunk at `local`, from its level up.
    fn mip(&mut self, local: u16) {
        let (mut x, mut z) = (usize::from(local) % 32, usize::from(local) / 32);
        for level in 0..LEVELS {
            (x, z) = (x / 2, z / 2);
            let cell = merge4([
                self.child(level, x * 2, z * 2),
                self.child(level, x * 2 + 1, z * 2),
                self.child(level, x * 2, z * 2 + 1),
                self.child(level, x * 2 + 1, z * 2 + 1),
            ]);
            self.pyramid[level][z * (16 >> level) + x] = cell;
        }
    }
    fn mip_all(&mut self) {
        for level in 0..LEVELS {
            let side = 16 >> level;
            for z in 0..side {
                for x in 0..side {
                    let cell = merge4([
                        self.child(level, x * 2, z * 2),
                        self.child(level, x * 2 + 1, z * 2),
                        self.child(level, x * 2, z * 2 + 1),
                        self.child(level, x * 2 + 1, z * 2 + 1),
                    ]);
                    self.pyramid[level][z * side + x] = cell;
                }
            }
        }
    }
}

/// Four cells merged into the one covering them: none unless at least half
/// of them were seen. One seen chunk must not stand in for a whole square of
/// unseen ones (an ocean chunk at the edge of the explored area became a tall
/// wall of water); unseen cells under a square that is drawn vote for air.
fn merge4(cells: [&Option<Column>; 4]) -> Option<Column> {
    let seen: Vec<&Column> = cells.into_iter().flatten().collect();
    (seen.len() * 2 >= 4).then(|| Column::merge(seen, 4))
}

struct Inner {
    regions: FxHashMap<(i32, i32), Region>,
    order: VecDeque<(i32, i32)>,
}

/// The seen chunks of one server's (or world's) dimension (`open`).
pub struct LodStore {
    root: PathBuf,
    fingerprint: u64,
    inner: Mutex<Inner>,
    /// Logs open for appending. Writes take only this lock, never `inner`:
    /// samplers reading cells must not wait for the disk.
    logs: Mutex<FxHashMap<(i32, i32), File>>,
    /// The saved region files imported (`ChunkImport`), and their
    /// modification times then.
    imported: Mutex<Option<FxHashMap<(i32, i32), u64>>>,
}

fn region_of(chunk: (i32, i32)) -> ((i32, i32), u16) {
    let region = (chunk.0.div_euclid(REGION), chunk.1.div_euclid(REGION));
    let local = chunk.1.rem_euclid(REGION) * REGION + chunk.0.rem_euclid(REGION);
    (region, local as u16)
}

/// FNV-1a: stable across builds, unlike the standard hasher.
fn hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    hash.max(1)
}

impl LodStore {
    /// The store at `root`, created if needed. `fingerprint` names the block
    /// state and biome tables (`fingerprint`): a store written with others is
    /// emptied. One store per folder: a far terrain still saving what it had
    /// queued (`Lod`'s drop) shares it with the next one, as two would write
    /// over each other's records.
    pub fn open(root: PathBuf, fingerprint: u64) -> io::Result<Arc<Self>> {
        static OPEN: Mutex<Vec<(PathBuf, std::sync::Weak<LodStore>)>> = Mutex::new(Vec::new());
        let mut open = OPEN.lock().unwrap();
        open.retain(|(_, store)| store.strong_count() > 0);
        if let Some(store) = open
            .iter()
            .filter(|(at, _)| *at == root)
            .find_map(|(_, store)| store.upgrade())
            && store.fingerprint == fingerprint
        {
            return Ok(store);
        }
        let store = Arc::new(Self::create(root.clone(), fingerprint)?);
        open.push((root, Arc::downgrade(&store)));
        Ok(store)
    }
    fn create(root: PathBuf, fingerprint: u64) -> io::Result<Self> {
        fs::create_dir_all(&root)?;
        let meta = root.join("meta");
        let expected = format!("horizon-lod 1 {fingerprint:016x}\n");
        if fs::read_to_string(&meta).ok().as_deref() != Some(expected.as_str()) {
            for entry in fs::read_dir(&root)? {
                let path = entry?.path();
                if path
                    .extension()
                    .is_some_and(|e| e == "lod" || e == "coarse")
                    || path.file_name().is_some_and(|name| name == "imported")
                {
                    fs::remove_file(path)?;
                }
            }
            fs::write(&meta, expected)?;
        }
        Ok(Self {
            root,
            fingerprint,
            inner: Mutex::new(Inner {
                regions: FxHashMap::default(),
                order: VecDeque::new(),
            }),
            logs: Mutex::new(FxHashMap::default()),
            imported: Mutex::new(None),
        })
    }
    fn path(&self, region: (i32, i32), extension: &str) -> PathBuf {
        self.root
            .join(format!("r.{}.{}.{extension}", region.0, region.1))
    }
    /// Runs `f` on the region, loading it first if it isn't in memory. The
    /// load (and writing out the region it pushes out) happens outside the
    /// lock.
    fn with_region<R>(&self, key: (i32, i32), f: impl FnOnce(&mut Region) -> R) -> R {
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(region) = inner.regions.get_mut(&key) {
                return f(region);
            }
        }
        let loaded = self.load(key);
        let mut evicted = Vec::new();
        let result = {
            let mut inner = self.inner.lock().unwrap();
            if !inner.regions.contains_key(&key) {
                while inner.regions.len() >= KEPT_REGIONS
                    && let Some(old) = inner.order.pop_front()
                {
                    if let Some(region) = inner.regions.remove(&old)
                        && region.unflushed > 0
                    {
                        evicted.push((old, coarse_raw(&region.coarse)));
                    }
                }
                inner.order.push_back(key);
                inner.regions.insert(key, loaded);
            }
            f(inner.regions.get_mut(&key).expect("inserted above"))
        };
        for (old, raw) in evicted {
            if let Err(e) = write_atomic(&self.path(old, "coarse"), &pack_coarse(&raw)) {
                log::warn!("Horizon: could not save far terrain: {e}");
            }
        }
        result
    }
    /// Keeps a chunk's columns, replacing what was kept for it. False when
    /// they are what was kept already: nothing is written then.
    pub fn put(&self, chunk: (i32, i32), columns: &[Column; 256]) -> io::Result<bool> {
        let (key, local) = region_of(chunk);
        let raw = encode_columns(columns);
        let hash = hash(&raw);
        if self.with_region(key, |region| {
            region
                .index
                .get(&local)
                .is_some_and(|entry| entry.hash == hash)
        }) {
            return Ok(false);
        }
        let data = miniz_oxide::deflate::compress_to_vec(&raw, 6);
        let mut record = Vec::with_capacity(HASHED_HEADER + data.len());
        record.push(RECORD_HASHED);
        record.extend_from_slice(&chunk.0.to_le_bytes());
        record.extend_from_slice(&chunk.1.to_le_bytes());
        record.extend_from_slice(&(data.len() as u32).to_le_bytes());
        record.extend_from_slice(&hash.to_le_bytes());
        record.extend_from_slice(&data);
        let coarse = Column::merge(columns.iter(), 256);
        let offset = {
            let mut logs = self.logs.lock().unwrap();
            if !logs.contains_key(&key) && logs.len() >= OPEN_LOGS {
                logs.clear();
            }
            let file = match logs.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(self.path(key, "lod"))?,
                ),
            };
            let offset = file.seek(SeekFrom::End(0))?;
            if let Err(e) = file.write_all(&record) {
                logs.remove(&key);
                return Err(e);
            }
            offset
        };
        let flush = self.with_region(key, |region| {
            region.index.insert(
                local,
                Entry {
                    offset,
                    length: record.len() as u32,
                    hash,
                },
            );
            region.coarse[usize::from(local)] = Some(coarse);
            region.mip(local);
            region.unflushed += 1;
            (region.unflushed >= FLUSH_EVERY).then(|| {
                region.unflushed = 0;
                coarse_raw(&region.coarse)
            })
        });
        if let Some(raw) = flush {
            write_atomic(&self.path(key, "coarse"), &pack_coarse(&raw))?;
        }
        Ok(true)
    }
    /// A chunk's columns, if it was ever kept.
    pub fn chunk(&self, chunk: (i32, i32)) -> Option<Columns> {
        let (key, local) = region_of(chunk);
        let entry = self.with_region(key, |region| region.index.get(&local).copied())?;
        // Read outside the lock; the record's header says whose it is.
        let mut file = File::open(self.path(key, "lod")).ok()?;
        file.seek(SeekFrom::Start(entry.offset)).ok()?;
        let mut record = vec![0; entry.length as usize];
        file.read_exact(&mut record).ok()?;
        let parsed = parse_record(&record)?;
        if parsed.chunk != chunk || parsed.length != record.len() {
            return None;
        }
        decode_chunk(parsed.data)
    }
    /// Whether a chunk was ever kept, without reading it.
    pub fn contains(&self, chunk: (i32, i32)) -> bool {
        let (key, local) = region_of(chunk);
        self.with_region(key, |region| region.index.contains_key(&local))
    }
    /// One column standing for the `width`-block square holding (x, z)
    /// (`width` over 16, a power of two): the region's mip of the chunks
    /// under it (`Region::mip`), or the four squares half as wide merged,
    /// past a region. None unless at least half of each square was seen.
    pub fn cell(&self, x: i32, z: i32, width: i32) -> Option<Column> {
        let (x, z) = (x.div_euclid(width) * width, z.div_euclid(width) * width);
        if width > REGION * 16 {
            let half = width / 2;
            let cells = [
                self.cell(x, z, half),
                self.cell(x + half, z, half),
                self.cell(x, z + half, half),
                self.cell(x + half, z + half, half),
            ];
            return merge4([&cells[0], &cells[1], &cells[2], &cells[3]]);
        }
        let level = (width.trailing_zeros() as usize)
            .saturating_sub(5)
            .min(LEVELS - 1);
        let side = 16 >> level;
        let key = (x.div_euclid(REGION * 16), z.div_euclid(REGION * 16));
        let cx = (x.rem_euclid(REGION * 16) / width) as usize;
        let cz = (z.rem_euclid(REGION * 16) / width) as usize;
        self.with_region(key, |region| region.pyramid[level][cz * side + cx].clone())
    }
    /// Writes every region's coarse columns that changed.
    pub fn flush(&self) -> io::Result<()> {
        let changed: Vec<_> = {
            let mut inner = self.inner.lock().unwrap();
            inner
                .regions
                .iter_mut()
                .filter(|(_, region)| region.unflushed > 0)
                .map(|(&key, region)| {
                    region.unflushed = 0;
                    (key, coarse_raw(&region.coarse))
                })
                .collect()
        };
        for (key, raw) in changed {
            write_atomic(&self.path(key, "coarse"), &pack_coarse(&raw))?;
        }
        Ok(())
    }
    /// The saved region's modification time when it was last imported.
    pub fn imported(&self, region: (i32, i32)) -> Option<u64> {
        let mut imported = self.imported.lock().unwrap();
        imported
            .get_or_insert_with(|| {
                fs::read_to_string(self.root.join("imported"))
                    .unwrap_or_default()
                    .lines()
                    .filter_map(|line| {
                        let mut words = line.split_whitespace();
                        let x = words.next()?.parse().ok()?;
                        let z = words.next()?.parse().ok()?;
                        Some(((x, z), words.next()?.parse().ok()?))
                    })
                    .collect()
            })
            .get(&region)
            .copied()
    }
    /// Notes a saved region as imported at `modified` (`imported`).
    pub fn set_imported(&self, region: (i32, i32), modified: u64) -> io::Result<()> {
        self.imported(region);
        let text = {
            let mut imported = self.imported.lock().unwrap();
            let imported = imported.as_mut().expect("read above");
            imported.insert(region, modified);
            let mut text = String::new();
            for ((x, z), modified) in imported.iter() {
                text.push_str(&format!("{x} {z} {modified}\n"));
            }
            text
        };
        write_atomic(&self.root.join("imported"), text.as_bytes())
    }
    /// The region from disk. A log that is mostly records later ones
    /// replaced is compacted first, holding the log lock so that no append
    /// is lost.
    fn load(&self, key: (i32, i32)) -> Region {
        let mut region = Region::empty();
        let path = self.path(key, "lod");
        let mut logs = self.logs.lock().unwrap();
        let mut log = fs::read(&path).unwrap_or_default();
        let mut offset = 0;
        let mut latest = FxHashMap::default();
        while let Some(record) = parse_record(&log[offset..]) {
            let (at, local) = region_of(record.chunk);
            if at == key {
                latest.insert(local, (offset, record.length, record.hash));
            }
            offset += record.length;
        }
        let live: usize = latest.values().map(|&(_, length, _)| length).sum();
        if offset < log.len() || (offset > COMPACT_AFTER && offset > live * 2) {
            // A record cut short (the game stopped mid-write), or a log of
            // mostly stale records: rewritten with the latest whole ones.
            logs.remove(&key);
            let mut records: Vec<_> = latest.iter().map(|(&l, &r)| (l, r)).collect();
            records.sort_unstable_by_key(|&(_, (offset, _, _))| offset);
            let mut compacted = Vec::with_capacity(live);
            let mut moved = FxHashMap::default();
            for (local, (at, length, hash)) in records {
                moved.insert(local, (compacted.len(), length, hash));
                compacted.extend_from_slice(&log[at..at + length]);
            }
            if write_atomic(&path, &compacted).is_ok() {
                log = compacted;
                latest = moved;
            } else if offset < log.len()
                && let Ok(file) = OpenOptions::new().write(true).open(&path)
            {
                // Appends go after the last whole record.
                let _ = file.set_len(offset as u64);
                log.truncate(offset);
            }
        }
        drop(logs);
        for (&local, &(offset, length, hash)) in &latest {
            region.index.insert(
                local,
                Entry {
                    offset: offset as u64,
                    length: length as u32,
                    hash,
                },
            );
        }
        match fs::read(self.path(key, "coarse"))
            .ok()
            .and_then(|data| decode_coarse(&data))
        {
            Some(coarse) => region.coarse = coarse,
            None => {
                // Missing or damaged: rebuilt from the log, and written again.
                for (&local, &(offset, _, _)) in &latest {
                    if let Some(record) = parse_record(&log[offset..])
                        && let Some(columns) = decode_chunk(record.data)
                    {
                        region.coarse[usize::from(local)] =
                            Some(Column::merge(columns.iter(), 256));
                        region.unflushed += 1;
                    }
                }
            }
        }
        region.mip_all();
        region
    }
}

impl Drop for LodStore {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            log::warn!("Horizon: could not save far terrain: {e}");
        }
    }
}

/// A store's block state and biome tables, as `LodStore::open` compares
/// them: the column ids mean these blocks and biomes.
pub fn fingerprint<'a>(
    states: impl IntoIterator<Item = &'a str>,
    biomes: impl IntoIterator<Item = (&'a str, u16)>,
) -> u64 {
    // FNV-1a: stable across builds, unlike the standard hasher.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for state in states {
        eat(state.as_bytes());
        eat(&[0]);
    }
    let mut biomes: Vec<_> = biomes.into_iter().collect();
    biomes.sort_unstable_by_key(|&(_, id)| id);
    for (name, id) in biomes {
        eat(name.as_bytes());
        eat(&id.to_le_bytes());
    }
    hash
}

fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, data)?;
    fs::rename(temporary, path)
}

struct Record<'a> {
    chunk: (i32, i32),
    hash: u64,
    data: &'a [u8],
    /// The whole record's length, header included.
    length: usize,
}

fn parse_record(bytes: &[u8]) -> Option<Record<'_>> {
    let header = match *bytes.first()? {
        RECORD => HEADER,
        RECORD_HASHED => HASHED_HEADER,
        _ => return None,
    };
    if bytes.len() < header {
        return None;
    }
    let word = |at: usize| <[u8; 4]>::try_from(&bytes[at..at + 4]).expect("4 bytes");
    let chunk = (i32::from_le_bytes(word(1)), i32::from_le_bytes(word(5)));
    let length = u32::from_le_bytes(word(9)) as usize;
    if length > MAX_CHUNK_BYTES {
        return None;
    }
    let hash = if header == HASHED_HEADER {
        u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[13..21]).expect("8 bytes"))
    } else {
        0
    };
    Some(Record {
        chunk,
        hash,
        data: bytes.get(header..header + length)?,
        length: header + length,
    })
}

fn decode_chunk(data: &[u8]) -> Option<Columns> {
    let raw = miniz_oxide::inflate::decompress_to_vec_with_limit(data, MAX_CHUNK_BYTES).ok()?;
    let mut reader = Reader(&raw);
    let mut columns = Vec::with_capacity(256);
    for _ in 0..256 {
        columns.push(reader.column()?);
    }
    reader.0.is_empty().then_some(())?;
    <Box<[Column; 256]>>::try_from(columns.into_boxed_slice())
        .ok()
        .map(Arc::from)
}

fn encode_columns(columns: &[Column; 256]) -> Vec<u8> {
    let mut out = Vec::new();
    for column in columns {
        write_column(&mut out, column);
    }
    out
}

fn coarse_raw(coarse: &[Option<Column>]) -> Vec<u8> {
    let mut raw = Vec::new();
    for column in coarse {
        match column {
            Some(column) => {
                raw.push(1);
                write_column(&mut raw, column);
            }
            None => raw.push(0),
        }
    }
    raw
}

fn pack_coarse(raw: &[u8]) -> Vec<u8> {
    let mut out = COARSE_MAGIC.to_vec();
    out.extend(miniz_oxide::deflate::compress_to_vec(raw, 6));
    out
}

#[cfg(test)]
fn encode_coarse(coarse: &[Option<Column>]) -> Vec<u8> {
    pack_coarse(&coarse_raw(coarse))
}

fn decode_coarse(data: &[u8]) -> Option<Vec<Option<Column>>> {
    let data = data.strip_prefix(COARSE_MAGIC)?;
    let raw = miniz_oxide::inflate::decompress_to_vec_with_limit(data, MAX_CHUNK_BYTES).ok()?;
    let mut reader = Reader(&raw);
    let mut coarse = Vec::with_capacity((REGION * REGION) as usize);
    for _ in 0..REGION * REGION {
        coarse.push(match reader.byte()? {
            0 => None,
            1 => Some(reader.column()?),
            _ => return None,
        });
    }
    reader.0.is_empty().then_some(coarse)
}

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn signed(out: &mut Vec<u8>, value: i32) {
    varint(out, u64::from(((value << 1) ^ (value >> 31)) as u32));
}

/// A column: flags (1 surface, 2 its water), the surface, then its spans as
/// bottom, height, state, biome and light.
fn write_column(out: &mut Vec<u8>, column: &Column) {
    let water = column.surface.and_then(|s| s.water);
    out.push(u8::from(column.surface.is_some()) | (u8::from(water.is_some()) << 1));
    if let Some(surface) = column.surface {
        signed(out, surface.height);
        varint(out, u64::from(surface.state.0));
        varint(out, u64::from(surface.biome));
        if let Some((height, state)) = water {
            signed(out, height);
            varint(out, u64::from(state.0));
        }
    }
    varint(out, column.spans.len() as u64);
    for span in &column.spans {
        signed(out, i32::from(span.min));
        varint(
            out,
            (i32::from(span.max) - i32::from(span.min)).max(0) as u64,
        );
        varint(out, u64::from(span.state.0));
        varint(out, u64::from(span.biome));
        varint(out, u64::from(span.light));
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&first, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(first)
    }
    fn varint(&mut self) -> Option<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return Some(value);
            }
        }
        None
    }
    fn small<T: TryFrom<u64>>(&mut self) -> Option<T> {
        T::try_from(self.varint()?).ok()
    }
    fn signed(&mut self) -> Option<i32> {
        let value: u32 = self.small()?;
        Some(((value >> 1) as i32) ^ -((value & 1) as i32))
    }
    fn column(&mut self) -> Option<Column> {
        let flags = self.byte()?;
        let surface = if flags & 1 != 0 {
            let height = self.signed()?;
            let state = BlockStateId(self.small()?);
            let biome = self.small()?;
            let water = if flags & 2 != 0 {
                Some((self.signed()?, BlockStateId(self.small()?)))
            } else {
                None
            };
            Some(Surface {
                height,
                state,
                water,
                biome,
            })
        } else {
            None
        };
        let count: usize = self.small()?;
        if count > MAX_SPANS {
            return None;
        }
        let mut spans = Vec::with_capacity(count);
        for _ in 0..count {
            let min = self.signed()?;
            let height: u16 = self.small()?;
            let height = i32::from(height);
            spans.push(Span {
                min: i16::try_from(min).ok()?,
                max: i16::try_from(min + height).ok()?,
                state: BlockStateId(self.small()?),
                biome: self.small()?,
                light: self.small()?,
            });
        }
        Some(Column {
            surface,
            spans,
            unseen: false,
        })
    }
}

/// Chunks that exist on disk outside the store, such as a local world's
/// saved region files: imported once (and again when a region file changes)
/// on a thread of their own, nearest first, as Voxy's `WorldImporter` does.
pub trait ChunkImport: Send + Sync {
    /// The region files there, and their modification times.
    fn regions(&self) -> Vec<((i32, i32), u64)>;
    /// The finished chunks of one region, as far-terrain columns.
    fn region(&self, region: (i32, i32), stop: &AtomicBool) -> Vec<((i32, i32), Columns)>;
}

/// Far terrain from existing chunks only: a server's (the chunks seen
/// there), or a local world's with Far Terrain: Existing Chunks (its saved
/// chunks too, `import`). Nothing where nobody looked.
pub struct StoreSource {
    pub store: Arc<LodStore>,
    /// Where else existing chunks are, for a local world (`ChunkImport`).
    pub import: Option<Arc<dyn ChunkImport>>,
}

impl LodSource for StoreSource {
    fn sampler(self: Arc<Self>) -> Box<dyn LodSampler> {
        let importer = self
            .import
            .clone()
            .and_then(|import| Importer::start(self.store.clone(), import));
        Box::new(StoreSampler {
            store: self.store.clone(),
            cache: FxHashMap::default(),
            order: VecDeque::new(),
            details: Vec::new(),
            importer,
        })
    }
    fn remember(&self, chunk: (i32, i32), columns: &[Column; 256]) {
        if let Err(e) = self.store.put(chunk, columns) {
            log::warn!("Horizon: could not save far terrain: {e}");
        }
    }
    fn keeps(&self) -> bool {
        true
    }
}

/// The import thread (`ChunkImport`): imported chunks that changed come back
/// as details (`LodSampler::drain_details`), so that drawn tiles update.
struct Importer {
    results: mpsc::Receiver<LodDetail>,
    center: Arc<Mutex<[i32; 2]>>,
    stop: Arc<AtomicBool>,
}

impl Importer {
    fn start(store: Arc<LodStore>, import: Arc<dyn ChunkImport>) -> Option<Self> {
        let (tx, results) = mpsc::channel();
        let center = Arc::new(Mutex::new([0i32, 0]));
        let stop = Arc::new(AtomicBool::new(false));
        let (at, stopping) = (center.clone(), stop.clone());
        std::thread::Builder::new()
            .name("Far terrain import".into())
            .spawn(move || {
                let mut pending: Vec<_> = import
                    .regions()
                    .into_iter()
                    .filter(|&(region, modified)| store.imported(region) != Some(modified))
                    .collect();
                while !pending.is_empty() && !stopping.load(Ordering::Relaxed) {
                    let [cx, cz] = *at.lock().unwrap();
                    let (rx, rz) = (cx.div_euclid(512), cz.div_euclid(512));
                    let nearest = (0..pending.len())
                        .min_by_key(|&i| {
                            let ((x, z), _) = pending[i];
                            (i64::from(x - rx).pow(2) + i64::from(z - rz).pow(2), x, z)
                        })
                        .expect("not empty");
                    let (region, modified) = pending.swap_remove(nearest);
                    for (chunk, columns) in import.region(region, &stopping) {
                        if stopping.load(Ordering::Relaxed) {
                            return;
                        }
                        match store.put(chunk, &columns) {
                            Ok(true) => {
                                if tx.send((chunk, columns)).is_err() {
                                    return;
                                }
                            }
                            Ok(false) => {}
                            Err(e) => {
                                log::warn!("Horizon: could not import far terrain: {e}");
                                return;
                            }
                        }
                    }
                    if stopping.load(Ordering::Relaxed) {
                        return;
                    }
                    if let Err(e) = store.set_imported(region, modified) {
                        log::warn!("Horizon: could not import far terrain: {e}");
                    }
                }
                if let Err(e) = store.flush() {
                    log::warn!("Horizon: could not save far terrain: {e}");
                }
            })
            .map_err(|e| log::warn!("Horizon: far terrain import unavailable: {e}"))
            .ok()?;
        Some(Self {
            results,
            center,
            stop,
        })
    }
}

impl Drop for Importer {
    fn drop(&mut self) {
        // Not joined: it stops between chunks.
        self.stop.store(true, Ordering::Relaxed);
    }
}

struct StoreSampler {
    store: Arc<LodStore>,
    /// Decoded chunks and their reductions (`ReducedColumns`), so that a
    /// cell 2 to 16 blocks wide is a lookup.
    cache: FxHashMap<(i32, i32), Option<Arc<ReducedColumns>>>,
    order: VecDeque<(i32, i32)>,
    details: Vec<LodDetail>,
    /// The owner's import thread (`ChunkImport`); forks have none.
    importer: Option<Importer>,
}

impl StoreSampler {
    fn chunk(&mut self, key: (i32, i32)) -> Option<Arc<ReducedColumns>> {
        if let Some(columns) = self.cache.get(&key) {
            return columns.clone();
        }
        let columns = self
            .store
            .chunk(key)
            .map(|columns| Arc::new(ReducedColumns::new(columns)));
        while self.cache.len() >= KEPT_CHUNKS
            && let Some(old) = self.order.pop_front()
        {
            self.cache.remove(&old);
        }
        self.order.push_back(key);
        self.cache.insert(key, columns.clone());
        columns
    }
}

impl LodSampler for StoreSampler {
    fn fork(&self) -> Option<Box<dyn LodSampler>> {
        Some(Box::new(Self {
            store: self.store.clone(),
            cache: FxHashMap::default(),
            order: VecDeque::new(),
            details: Vec::new(),
            importer: None,
        }))
    }
    fn request_detail(&mut self, chunks: &[(i32, i32)], center: [i32; 2]) {
        if let Some(importer) = &self.importer {
            *importer.center.lock().unwrap() = center;
        }
        for &key in chunks {
            // A chunk kept since this sampler read it is read again.
            if self.cache.get(&key).is_some_and(Option::is_none) && self.store.contains(key) {
                self.cache.remove(&key);
            }
            if let Some(columns) = self.chunk(key) {
                self.details.push((key, columns.columns.clone()));
            }
        }
    }
    fn drain_details(&mut self) -> Vec<LodDetail> {
        let mut details = std::mem::take(&mut self.details);
        if let Some(importer) = &self.importer {
            while let Ok((key, columns)) = importer.results.try_recv() {
                self.cache.remove(&key);
                details.push((key, columns));
            }
        }
        details
    }
    fn sample(&mut self, x: i32, z: i32) -> Option<Surface> {
        self.column(x, z).surface
    }
    fn column(&mut self, x: i32, z: i32) -> Column {
        self.reduced_column(x, z, 1)
    }
    fn reduced_column(&mut self, x: i32, z: i32, width: i32) -> Column {
        let unseen = Column {
            unseen: true,
            ..Column::default()
        };
        if width > 16 {
            return self.store.cell(x, z, width).unwrap_or(unseen);
        }
        self.chunk((x.div_euclid(16), z.div_euclid(16)))
            .map(|columns| {
                columns
                    .get(
                        x.rem_euclid(16) as usize,
                        z.rem_euclid(16) as usize,
                        width as usize,
                    )
                    .clone()
            })
            .unwrap_or(unseen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::MIN_Y;

    fn solid(height: i32, state: u32) -> Column {
        Column::from_surface(Some(Surface {
            height,
            state: BlockStateId(state),
            water: None,
            biome: 3,
        }))
    }

    fn temp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("horizon-lod-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn chunk(height: i32, state: u32) -> [Column; 256] {
        std::array::from_fn(|i| {
            let mut column = solid(height + (i % 3) as i32, state);
            column.surface.as_mut().expect("solid").water = Some((height + 9, BlockStateId(7)));
            column.spans.push(Span {
                min: (height + 3) as i16,
                max: (height + 9) as i16,
                state: BlockStateId(7),
                biome: 2,
                light: 0xf0f0,
            });
            column
        })
    }

    #[test]
    fn columns_round_trip_through_the_encoding() {
        let columns = chunk(MIN_Y + 70, 1);
        let data = miniz_oxide::deflate::compress_to_vec(&encode_columns(&columns), 6);
        assert_eq!(*decode_chunk(&data).expect("decodes"), columns);
        let coarse = vec![None, Some(columns[5].clone()), Some(Column::default())];
        let mut padded = coarse.clone();
        padded.resize((REGION * REGION) as usize, None);
        assert_eq!(decode_coarse(&encode_coarse(&padded)), Some(padded));
        assert!(decode_chunk(&data[..data.len() / 2]).is_none());
    }

    #[test]
    fn kept_chunks_survive_reopening_and_the_latest_wins() -> io::Result<()> {
        let dir = temp("reopen");
        {
            let store = LodStore::open(dir.clone(), 1)?;
            store.put((-1, 40), &chunk(60, 1))?;
            store.put((-1, 40), &chunk(64, 2))?;
            store.put((5, 5), &chunk(10, 3))?;
            store.put((0, 40), &chunk(64, 2))?;
            store.put((-2, 40), &chunk(64, 2))?;
        }
        let store = LodStore::open(dir.clone(), 1)?;
        assert_eq!(*store.chunk((-1, 40)).expect("kept"), chunk(64, 2));
        assert_eq!(*store.chunk((5, 5)).expect("kept"), chunk(10, 3));
        assert!(store.chunk((6, 5)).is_none());
        // Coarse columns too, though fewer than FLUSH_EVERY records were written:
        // half of this 2 x 2 chunk square was seen.
        let cell = store.cell(-32, 40 * 16, 32).expect("seen");
        assert_eq!(cell.spans[0].state, BlockStateId(2));
        // A quarter of this one: not enough to draw.
        assert!(store.cell(5 * 16, 5 * 16, 32).is_none());
        assert!(store.cell(1600, 1600, 512).is_none());
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn a_cut_record_is_dropped_and_appends_follow_the_last_whole_one() -> io::Result<()> {
        let dir = temp("cut");
        {
            let store = LodStore::open(dir.clone(), 1)?;
            store.put((0, 0), &chunk(60, 1))?;
            store.put((1, 0), &chunk(60, 2))?;
        }
        let log = dir.join("r.0.0.lod");
        let length = fs::metadata(&log)?.len();
        OpenOptions::new()
            .write(true)
            .open(&log)?
            .set_len(length - 5)?;
        fs::remove_file(dir.join("r.0.0.coarse"))?;
        {
            let store = LodStore::open(dir.clone(), 1)?;
            assert!(store.chunk((1, 0)).is_none());
            store.put((2, 0), &chunk(60, 3))?;
        }
        let store = LodStore::open(dir.clone(), 1)?;
        assert_eq!(*store.chunk((0, 0)).expect("whole"), chunk(60, 1));
        assert_eq!(*store.chunk((2, 0)).expect("appended"), chunk(60, 3));
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn a_folder_has_one_store_while_it_is_in_use() -> io::Result<()> {
        let dir = temp("shared");
        let first = LodStore::open(dir.clone(), 1)?;
        let second = LodStore::open(dir.clone(), 1)?;
        assert!(Arc::ptr_eq(&first, &second));
        // Both write through it: neither record hides the other.
        first.put((0, 0), &chunk(60, 1))?;
        second.put((1, 0), &chunk(60, 2))?;
        drop((first, second));
        let store = LodStore::open(dir.clone(), 1)?;
        assert!(store.chunk((0, 0)).is_some() && store.chunk((1, 0)).is_some());
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }
    #[test]
    fn other_block_tables_empty_the_store() -> io::Result<()> {
        let dir = temp("fingerprint");
        LodStore::open(dir.clone(), 1)?.put((0, 0), &chunk(60, 1))?;
        assert!(LodStore::open(dir.clone(), 1)?.chunk((0, 0)).is_some());
        assert!(LodStore::open(dir.clone(), 2)?.chunk((0, 0)).is_none());
        assert_ne!(
            fingerprint(["a", "b"], [("plains", 0)]),
            fingerprint(["a", "b"], [("plains", 1)])
        );
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn the_sampler_draws_seen_chunks_and_leaves_the_rest_empty() -> io::Result<()> {
        let dir = temp("sampler");
        let source = Arc::new(StoreSource {
            store: LodStore::open(dir.clone(), 1)?,
            import: None,
        });
        // Half of the 4 x 4 chunks under the first 64-block cell, and one
        // chunk under the second.
        for cz in 0..2 {
            for cx in 0..4 {
                source.remember((cx, cz), &chunk(80, 4));
            }
        }
        source.remember((4, 0), &chunk(80, 5));
        source.remember((2, 3), &chunk(80, 4));
        let mut sampler = source.clone().sampler();
        let fine = sampler.grid(32, 48, 1, 16);
        assert_eq!(fine, chunk(80, 4).to_vec());
        assert!(sampler.column(0, 100).unseen);
        assert!(sampler.column(0, 100).spans.is_empty());
        assert!(!sampler.column(40, 50).unseen);
        let far = sampler.grid(0, 0, 64, 16);
        assert_eq!(far[0].spans[0].state, BlockStateId(4));
        assert!(!far[0].unseen);
        assert!(far[1].unseen && far[1].spans.is_empty(), "{:?}", far[1]);
        assert!(far[2].unseen);
        sampler.request_detail(&[(2, 3), (9, 9)], [0, 0]);
        let details = sampler.drain_details();
        assert_eq!(details.len(), 1);
        assert_eq!(details[0].0, (2, 3));
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn an_unchanged_chunk_is_not_written_again() -> io::Result<()> {
        let dir = temp("unchanged");
        let store = LodStore::open(dir.clone(), 1)?;
        assert!(store.put((3, 3), &chunk(60, 1))?);
        let length = fs::metadata(dir.join("r.0.0.lod"))?.len();
        assert!(!store.put((3, 3), &chunk(60, 1))?);
        assert_eq!(fs::metadata(dir.join("r.0.0.lod"))?.len(), length);
        assert!(store.put((3, 3), &chunk(61, 1))?);
        drop(store);
        // The hash is kept in the record: still skipped after reopening.
        let store = LodStore::open(dir.clone(), 1)?;
        assert!(!store.put((3, 3), &chunk(61, 1))?);
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn coarse_cells_are_mipped_up_to_the_region_and_past_it() -> io::Result<()> {
        let dir = temp("pyramid");
        let store = LodStore::open(dir.clone(), 1)?;
        // Every chunk of the region (0, 0), none of the others.
        for cz in 0..REGION {
            for cx in 0..REGION {
                store.put((cx, cz), &chunk(70, 4))?;
            }
        }
        for width in [32, 64, 128, 256, 512] {
            let cell = store.cell(width % 512, 0, width).expect("seen");
            assert_eq!(cell.spans[0].state, BlockStateId(4), "{width}");
        }
        // A quarter of the 1024-block square: not drawn.
        assert!(store.cell(0, 0, 1024).is_none());
        // Edits mip up at once.
        for cz in 0..2 {
            for cx in 0..2 {
                store.put((cx, cz), &chunk(70, 5))?;
            }
        }
        assert_eq!(
            store.cell(0, 0, 32).unwrap().spans[0].state,
            BlockStateId(5)
        );
        drop(store);
        let store = LodStore::open(dir.clone(), 1)?;
        assert_eq!(
            store.cell(0, 0, 32).unwrap().spans[0].state,
            BlockStateId(5)
        );
        assert_eq!(
            store.cell(0, 0, 512).unwrap().spans[0].state,
            BlockStateId(4)
        );
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn a_mostly_stale_log_is_compacted() -> io::Result<()> {
        let dir = temp("compact");
        {
            let store = LodStore::open(dir.clone(), 1)?;
            for height in 0..400 {
                store.put((1, 1), &chunk(height, 1))?;
            }
            store.put((2, 1), &chunk(9, 2))?;
        }
        let before = fs::metadata(dir.join("r.0.0.lod"))?.len();
        assert!(before as usize > COMPACT_AFTER, "{before}");
        let store = LodStore::open(dir.clone(), 1)?;
        assert_eq!(*store.chunk((1, 1)).expect("latest"), chunk(399, 1));
        assert_eq!(*store.chunk((2, 1)).expect("kept"), chunk(9, 2));
        assert!(fs::metadata(dir.join("r.0.0.lod"))?.len() < before / 10);
        store.put((3, 1), &chunk(5, 3))?;
        assert_eq!(*store.chunk((3, 1)).expect("appended"), chunk(5, 3));
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }

    #[test]
    fn existing_chunks_are_imported_once() -> io::Result<()> {
        struct Saved(std::sync::atomic::AtomicUsize);
        impl ChunkImport for Saved {
            fn regions(&self) -> Vec<((i32, i32), u64)> {
                vec![((0, 0), 7), ((-1, 0), 7)]
            }
            fn region(&self, region: (i32, i32), _: &AtomicBool) -> Vec<((i32, i32), Columns)> {
                self.0.fetch_add(1, Ordering::Relaxed);
                vec![((region.0 * 32 + 1, 2), Arc::new(chunk(80, 6)))]
            }
        }
        let dir = temp("import");
        let saved = Arc::new(Saved(Default::default()));
        let source = Arc::new(StoreSource {
            store: LodStore::open(dir.clone(), 1)?,
            import: Some(saved.clone()),
        });
        let mut sampler = source.clone().sampler();
        let mut details = Vec::new();
        for _ in 0..500 {
            details.extend(sampler.drain_details());
            if details.len() == 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let mut keys: Vec<_> = details.iter().map(|(key, _)| *key).collect();
        keys.sort_unstable();
        assert_eq!(keys, [(-31, 2), (1, 2)]);
        assert_eq!(sampler.column(16, 40).spans, chunk(80, 6)[8].spans);
        drop(sampler);
        while source.store.imported((-1, 0)).is_none() || source.store.imported((0, 0)).is_none() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Unchanged region files are not read again.
        let _again = source.clone().sampler();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(saved.0.load(Ordering::Relaxed), 2);
        drop(_again);
        let _ = fs::remove_dir_all(dir);
        Ok(())
    }
}
