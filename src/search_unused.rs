use aho_corasick::AhoCorasick;
use anyhow::Context as _;
use cargo_metadata::CargoOpt;
use grep::{
    matcher::LineTerminator,
    regex::{RegexMatcher, RegexMatcherBuilder},
    searcher::{self, Searcher, SearcherBuilder, Sink},
};
use log::{debug, trace, warn};
use meta::MetadataFields;
use rayon::prelude::*;
use std::{
    cell::OnceCell,
    collections::{BTreeMap, HashMap, HashSet},
    error,
    fs::File,
    io::Read as _,
    num::NonZeroUsize,
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};
use walkdir::WalkDir;

#[cfg(test)]
use crate::TOP_LEVEL;
use crate::UseCargoMetadata;

use self::meta::PackageMetadata;

mod meta {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct PackageMetadata {
        #[serde(rename = "cargo-machete")]
        pub cargo_machete: Option<MetadataFields>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct MetadataFields {
        /// Crates triggering false positives in `cargo-machete`, which should not be reported as
        /// unused.
        #[serde(default)]
        pub ignored: Vec<String>,

        /// Crates whose name is different than the name in the Cargo.toml.
        ///
        /// Some crates have a different lib name (the name in `use` statements) that their package
        /// name (e.g. `rustls-webpki` is imported with `use webpki;`).
        #[serde(default)]
        pub renamed: BTreeMap<Box<str>, Box<str>>,
    }
}

pub(crate) struct PackageAnalysis {
    pub manifest: cargo_toml::Manifest<meta::PackageMetadata>,
    pub package_name: String,
    pub unused: Vec<String>,
    pub ignored_used: Vec<String>,
}

impl PackageAnalysis {
    fn new(package_name: String, manifest: cargo_toml::Manifest<meta::PackageMetadata>) -> Self {
        Self {
            manifest,
            package_name,
            unused: Vec::default(),
            ignored_used: Vec::default(),
        }
    }
}

/// Canonicalizes a path, or returns it unchanged if that fails (e.g. it doesn't exist).
fn canonicalize_or_keep(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|err| {
        warn!("error when canonicalizing {}: {err}", path.display());
        path.to_owned()
    })
}

/// The parts of `cargo metadata`'s output cargo-machete uses.
///
/// Parsing only these, instead of the whole `cargo_metadata::Metadata`, takes a fraction of the
/// time and memory on large workspaces.
mod raw_metadata {
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    pub(super) struct Metadata {
        pub packages: Vec<Package>,
        pub workspace_members: Vec<String>,
        pub resolve: Option<Resolve>,
    }

    #[derive(Deserialize)]
    pub(super) struct Package {
        pub id: String,
        pub name: String,
        pub manifest_path: PathBuf,
        pub dependencies: Vec<Dependency>,
    }

    #[derive(Deserialize)]
    pub(super) struct Dependency {
        pub name: String,
        pub rename: Option<String>,
    }

    #[derive(Deserialize)]
    pub(super) struct Resolve {
        pub nodes: Vec<Node>,
    }

    #[derive(Deserialize)]
    pub(super) struct Node {
        pub id: String,
        pub deps: Vec<NodeDep>,
    }

    #[derive(Deserialize)]
    pub(super) struct NodeDep {
        pub name: String,
        pub pkg: String,
    }
}

/// Maps dependency keys (the names of the keys in a Cargo.toml dependency table) to the crate
/// names they're imported with.
type ResolvedDependencies = BTreeMap<String, String>;

/// What cargo-machete needs from one `cargo metadata` invocation.
struct WorkspaceMetadata {
    /// The resolved dependencies of each workspace member, by canonical manifest path. `None`
    /// if cargo didn't resolve the dependency graph.
    members: HashMap<PathBuf, Option<ResolvedDependencies>>,
}

impl WorkspaceMetadata {
    fn load(manifest_path: &Path) -> Result<Self, cargo_metadata::Error> {
        debug!("running cargo metadata for {}", manifest_path.display());

        let output = cargo_metadata::MetadataCommand::new()
            .features(CargoOpt::AllFeatures)
            .manifest_path(manifest_path)
            //.other_options(["--frozen".to_owned()]) // TODO causes errors in cargo-metadata
            .cargo_command()
            .output()?;

        // Same checks as `cargo_metadata::MetadataCommand::exec`.
        if !output.status.success() {
            return Err(cargo_metadata::Error::CargoMetadata {
                stderr: String::from_utf8(output.stderr)?,
            });
        }
        let json = std::str::from_utf8(&output.stdout)?
            .lines()
            .find(|line| line.starts_with('{'))
            .ok_or(cargo_metadata::Error::NoJson)?;
        let metadata: raw_metadata::Metadata = serde_json::from_str(json)?;

        Ok(Self::from_raw(&metadata))
    }

    fn from_raw(metadata: &raw_metadata::Metadata) -> Self {
        let packages_by_id: HashMap<&str, &raw_metadata::Package> = metadata
            .packages
            .iter()
            .map(|package| (package.id.as_str(), package))
            .collect();

        let nodes_by_id: Option<HashMap<&str, &raw_metadata::Node>> =
            metadata.resolve.as_ref().map(|resolve| {
                resolve
                    .nodes
                    .iter()
                    .map(|node| (node.id.as_str(), node))
                    .collect()
            });

        let members = metadata
            .workspace_members
            .iter()
            .map(|id| {
                let package = *packages_by_id
                    .get(id.as_str())
                    .expect("workspace members should appear under cargo-metadata packages");
                let resolved = nodes_by_id.as_ref().map(|nodes_by_id| {
                    let node = *nodes_by_id
                        .get(id.as_str())
                        .expect("workspace members should be resolved by cargo-metadata");
                    resolve_dependencies(package, node, &packages_by_id)
                });
                (canonicalize_or_keep(&package.manifest_path), resolved)
            })
            .collect();

        Self { members }
    }
}

/// Maps the dependency keys of a package to the crate names they're imported with.
fn resolve_dependencies(
    // This gives us the original dependency table
    // May have more than resolved if some were never enabled
    package: &raw_metadata::Package,
    // This gives us resolved dependencies, in crate form
    node: &raw_metadata::Node,
    packages_by_id: &HashMap<&str, &raw_metadata::Package>,
) -> ResolvedDependencies {
    // For every resolved dependency:
    // look it up in the package list to find the name (the one in registries)
    // look up that name in dependencies of the package;
    // find if it uses a different key through the rename field
    node.deps
        .iter()
        .map(|dep| {
            let crate_name = dep.name.clone();
            let dep_pkg = packages_by_id
                .get(dep.pkg.as_str())
                .expect("resolved dependencies should appear under cargo-metadata packages");

            // The dependency can appear more than once, for example if it is both
            // a dependency and a dev-dependency (often with more features enabled).
            // We'll assume cargo enforces consistency.
            let dep_spec = package
                .dependencies
                .iter()
                .find(|dep_spec| dep_spec.name == dep_pkg.name)
                .expect("resolved dependency should have a matching dependency spec");

            // If the dependency was renamed, through key = { package = … },
            // the original key is in dep_spec.rename.
            let dep_key = dep_spec
                .rename
                .clone()
                .unwrap_or_else(|| dep_spec.name.clone());
            (dep_key, crate_name)
        })
        .collect()
}

type LoadedMetadata = Result<Arc<WorkspaceMetadata>, Arc<cargo_metadata::Error>>;

/// Shares `cargo metadata` results between all the packages of a workspace.
///
/// `cargo metadata` resolves the whole workspace, whichever member's manifest it's run on. Running
/// it once per package makes the analysis of large workspaces quadratic, so its output is cached
/// per manifest path instead.
#[derive(Default)]
pub(crate) struct MetadataCache {
    loaded: Mutex<HashMap<PathBuf, Arc<OnceLock<LoadedMetadata>>>>,
}

impl MetadataCache {
    /// Runs `cargo metadata` on the given manifest, unless it's already been run.
    fn get_or_load(&self, manifest_path: &Path) -> LoadedMetadata {
        let cell = {
            let mut loaded = self.loaded.lock().expect("metadata cache lock poisoned");
            Arc::clone(loaded.entry(manifest_path.to_owned()).or_default())
        };

        // The map lock is released at this point, so different manifests load in parallel, while
        // concurrent requests for the same manifest wait for the first one to complete.
        cell.get_or_init(|| {
            WorkspaceMetadata::load(manifest_path)
                .map(Arc::new)
                .map_err(Arc::new)
        })
        .clone()
    }

    /// Number of manifests `cargo metadata` has been run on.
    #[cfg(test)]
    fn loaded_count(&self) -> usize {
        self.loaded
            .lock()
            .expect("metadata cache lock poisoned")
            .len()
    }

    /// Returns the resolved dependencies of the package at `manifest_path`, or `None` if cargo
    /// didn't resolve them.
    ///
    /// `workspace_manifest_path` is a guess of the manifest of the workspace this package belongs
    /// to. If the guess is wrong (the package is excluded from that workspace, or the workspace
    /// is broken), this falls back to running `cargo metadata` on the package's own manifest.
    ///
    /// Both paths must be canonical.
    fn for_package(
        &self,
        manifest_path: &Path,
        workspace_manifest_path: Option<&Path>,
    ) -> anyhow::Result<Option<ResolvedDependencies>> {
        if let Some(workspace_manifest_path) = workspace_manifest_path {
            match self.get_or_load(workspace_manifest_path) {
                Ok(workspace) => {
                    if let Some(resolved) = workspace.members.get(manifest_path) {
                        return Ok(resolved.clone());
                    }
                    debug!(
                        "{} isn't a member of the workspace {}",
                        manifest_path.display(),
                        workspace_manifest_path.display()
                    );
                }
                Err(err) => {
                    debug!(
                        "error when running cargo metadata for {}: {err}",
                        workspace_manifest_path.display()
                    );
                }
            }
        }

        let workspace = self.get_or_load(manifest_path)?;
        let resolved = workspace.members.get(manifest_path).with_context(|| {
            format!(
                "{} not found in the cargo metadata output",
                manifest_path.display()
            )
        })?;
        Ok(resolved.clone())
    }
}

/// Runs `cargo metadata` on the given manifests on background threads of `scope`, so it runs while
/// the caller does other work. The results are cached in `cache`.
fn spawn_metadata_loads<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    cache: &'env MetadataCache,
    manifest_paths: &'env [PathBuf],
) {
    // Each `cargo metadata` invocation is a process mostly using one core.
    let threads_count = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(manifest_paths.len());

    for thread_index in 0..threads_count {
        scope.spawn(move || {
            for manifest_path in manifest_paths
                .iter()
                .skip(thread_index)
                .step_by(threads_count)
            {
                // The result is cached, and errors are reported when a package needs it.
                let _ = cache.get_or_load(manifest_path);
            }
        });
    }
}

fn make_line_regexp(name: &str) -> String {
    // Syntax documentation: https://docs.rs/regex/latest/regex/#syntax
    //
    // Breaking down this regular expression: given a line,
    // - `use (::)?(?i){name}(?-i)(::|;| as)`: matches `use foo;`, `use foo::bar`, `use foo as bar;`, with
    // an optional "::" in front of the crate's name.
    // - `(?:[^:]|^|\W::)\b(?i){name}(?-i)::`: matches `foo::X`, but not `barfoo::X`. To ensure there's no polluting
    //   prefix we add `(?:[^:]|^|\W::)\b`, meaning that the crate name must be prefixed by either:
    //    * Not a `:` (therefore not a sub module)
    //    * The start of a line
    //    * Not a word character followed by `::` (to allow ::my_crate)
    // - `extern crate (?i){name}(?-i)( |;)`: matches `extern crate foo`, or `extern crate foo as bar`.
    // - `(?i){name}(?-i)` makes the match against the crate's name case insensitive
    format!(
        r#"use (::)?(?i){name}(?-i)(::|;| as)|(?:[^:]|^|\W::)\b(?i){name}(?-i)::|extern crate (?i){name}(?-i)( |;)"#
    )
}

fn make_multiline_regexp(name: &str) -> String {
    // Syntax documentation: https://docs.rs/regex/latest/regex/#syntax
    //
    // Breaking down this Terrible regular expression: tries to match uses of the crate's name in
    // compound `use` statement across multiple lines.
    //
    // It's split into 3 parts:
    //   1. Matches modules before the usage of the crate's name: `\s*(?:(::)?\w+{sub_modules_match}\s*,\s*)*`
    //   2. Matches the crate's name with optional sub-modules: `(::)?{name}{sub_modules_match}\s*`
    //   3. Matches modules after the usage of the crate's name: `(?:\s*,\s*(::)?\w+{sub_modules_match})*\s*,?\s*`
    //
    // In order to avoid false usage detection of `not_my_dep::my_dep` the regexp ensures that the
    // crate's name is at the top level of the use statement. However, it's not possible with
    // regexp to allow any number of matching `{` and `}` before the crate's usage (rust regexp
    // engine doesn't support recursion). Therefore, sub modules are authorized up to 4 levels
    // deep.

    let sub_modules_match = r#"(?:::\w+)*(?:::\*|\s+as\s+\w+|::\{(?:[^{}]*(?:\{(?:[^{}]*(?:\{(?:[^{}]*(?:\{[^{}]*\})?[^{}]*)*\})?[^{}]*)*\})?[^{}]*)*\})?"#;

    format!(
        r#"use \{{\s*(?:(::)?\w+{sub_modules_match}\s*,\s*)*(::)?{name}{sub_modules_match}\s*(?:\s*,\s*(::)?\w+{sub_modules_match})*\s*,?\s*\}};"#
    )
}

/// Returns all the paths to the Rust source files for a crate contained at the given path.
fn collect_paths(dir_path: &Path, analysis: &PackageAnalysis) -> Vec<PathBuf> {
    let manifest = &analysis.manifest;

    let mut root_paths: HashSet<PathBuf> = manifest
        .lib
        .iter()
        .chain(manifest.bin.iter())
        .chain(manifest.bench.iter())
        .chain(manifest.test.iter())
        .chain(manifest.example.iter())
        .filter_map(|p| {
            // Keep only files which names in `.rs`.
            let path_str = p.path.as_ref().filter(|s| s.ends_with(".rs"))?;
            // Remove the file name.
            PathBuf::from(path_str).parent().map(PathBuf::from)
        })
        .collect();

    trace!("found root paths: {root_paths:?}");

    if root_paths.is_empty() {
        // Assume "src/" if cargo_toml didn't find anything.
        root_paths.insert(PathBuf::from("src"));
        trace!("adding src/ since paths was empty");
    }

    // Collect all final paths for the crate first.
    let paths = root_paths
        .iter()
        .flat_map(|root| WalkDir::new(dir_path.join(root)).into_iter())
        .filter_map(|result| {
            result
                .inspect_err(|err| eprintln!("{err}"))
                .ok()
                .and_then(|entry| {
                    (entry.file_type().is_file()
                        && entry.path().extension().is_some_and(|ext| ext == "rs"))
                    .then(|| entry.path().to_owned())
                })
        })
        .collect();

    trace!("found transitive paths: {paths:?}");

    paths
}

/// The regular expressions looking for usages of one crate name.
///
/// Splitting the single line matcher from the multiline matcher makes maintenance of the regular
/// expressions simpler (oh well), and likely faster too since most use statements will be caught
/// by the single line matcher.
///
/// Compiling them costs much more than running them over a package's sources: around 2 ms and
/// 0.5 MB each.
struct CrateMatchers {
    crate_name: String,
    line: RegexMatcher,
    /// Compiled on first use, since the line matcher finds most usages.
    multiline: OnceCell<RegexMatcher>,
}

impl CrateMatchers {
    fn new(crate_name: &str) -> anyhow::Result<Self> {
        assert!(!crate_name.contains('-'));

        let line = RegexMatcher::new_line_matcher(&make_line_regexp(crate_name))?;

        Ok(Self {
            crate_name: crate_name.to_owned(),
            line,
            multiline: OnceCell::new(),
        })
    }

    fn multiline(&self) -> anyhow::Result<&RegexMatcher> {
        if let Some(multiline) = self.multiline.get() {
            return Ok(multiline);
        }
        let multiline = RegexMatcherBuilder::new()
            .multi_line(true)
            .build(&make_multiline_regexp(&self.crate_name))?;
        Ok(self.multiline.get_or_init(|| multiline))
    }
}

/// Searches using [`CrateMatchers`].
///
/// Building a searcher allocates its buffers, so searchers are reused across searches.
struct Searchers {
    line: Searcher,
    multiline: Searcher,
}

impl Searchers {
    fn new() -> Self {
        // Binary detection isn't needed: `SourceFiles` already cuts files at their first NUL byte.
        let line = SearcherBuilder::new()
            .line_terminator(LineTerminator::byte(b'\n'))
            .line_number(false)
            .build();
        let multiline = SearcherBuilder::new()
            .multi_line(true)
            .line_number(false)
            .build();
        Self { line, multiline }
    }

    /// Returns whether the source code in `haystack` uses the crate: first try to use the line
    /// matcher, then the multiline matcher if the line matcher failed.
    fn is_used_in(&mut self, matchers: &CrateMatchers, haystack: &[u8]) -> anyhow::Result<bool> {
        let mut sink = StopAfterFirstMatch::new();

        self.line
            .search_slice(&matchers.line, haystack, &mut sink)
            .map_err(|err| anyhow::anyhow!("when searching with line pattern: {err}"))?;
        if sink.found {
            return Ok(true);
        }

        let multiline_matcher = matchers.multiline()?;

        // Sanity-check: the matcher must allow multi-line searching.
        debug_assert!(self.multiline.multi_line_with_matcher(multiline_matcher));

        self.multiline
            .search_slice(multiline_matcher, haystack, &mut sink)
            .map_err(|err| anyhow::anyhow!("when searching with complex pattern: {err}"))?;
        Ok(sink.found)
    }
}

/// The Rust source files of a package, read once into a single contiguous buffer.
struct SourceFiles {
    paths: Vec<PathBuf>,
    bytes: Vec<u8>,
    /// The range in `bytes` of each file in `paths`.
    ranges: Vec<Range<usize>>,
}

impl SourceFiles {
    /// Reads the given files, skipping (and reporting) those which can't be read.
    fn read(paths: Vec<PathBuf>) -> Self {
        let mut bytes = Vec::new();
        let mut ranges = Vec::with_capacity(paths.len());
        let mut readable_paths = Vec::with_capacity(paths.len());

        for path in paths {
            let start = bytes.len();
            match File::open(&path).and_then(|mut file| file.read_to_end(&mut bytes)) {
                Ok(_) => {
                    // Like grep's binary detection, only search up to the first NUL byte.
                    if let Some(nul_offset) = memchr::memchr(b'\x00', &bytes[start..]) {
                        bytes.truncate(start + nul_offset);
                    }
                    ranges.push(start..bytes.len());
                    readable_paths.push(path);
                }
                Err(err) => {
                    bytes.truncate(start);
                    eprintln!("{}: {}", path.display(), err);
                }
            }
        }

        assert!(ranges.len() == readable_paths.len());
        Self {
            paths: readable_paths,
            bytes,
            ranges,
        }
    }

    #[cfg(test)]
    fn from_contents(contents: &[&str]) -> Self {
        let mut bytes = Vec::new();
        let mut ranges = Vec::with_capacity(contents.len());
        for content in contents {
            let start = bytes.len();
            bytes.extend_from_slice(content.as_bytes());
            ranges.push(start..bytes.len());
        }
        Self {
            paths: (0..contents.len())
                .map(|index| PathBuf::from(format!("file{index}.rs")))
                .collect(),
            bytes,
            ranges,
        }
    }

    fn files_count(&self) -> usize {
        self.ranges.len()
    }

    fn file(&self, file_index: usize) -> &[u8] {
        &self.bytes[self.ranges[file_index].clone()]
    }
}

/// Non-ASCII characters which match ASCII letters in case-insensitive regular expressions: the
/// Kelvin sign matches `k`, and the long s matches `s`.
const ASCII_CASE_FOLDING_CHARS: [&str; 2] = ["\u{212A}", "\u{17F}"];

/// Whether the crate name is matched literally once embedded in a regular expression.
fn is_literal_crate_name(crate_name: &str) -> bool {
    crate_name
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// Returns the sorted `(crate name index, file index)` pairs where the file may use the crate.
///
/// The regular expressions built by `make_line_regexp` and `make_multiline_regexp` only match
/// text containing the crate name, ignoring ASCII case at most. So instead of running both of
/// them on every file for every crate name, a single pass over each file finds all the crate names
/// it contains, and the regular expressions only run on those files.
fn candidate_files(
    crate_names: &[&str],
    sources: &SourceFiles,
) -> anyhow::Result<Vec<(usize, usize)>> {
    let mut candidates = Vec::new();

    // A crate name with regex syntax could match other text than itself: search it everywhere.
    let mut literal_name_indexes = Vec::with_capacity(crate_names.len());
    for (name_index, crate_name) in crate_names.iter().enumerate() {
        if is_literal_crate_name(crate_name) {
            literal_name_indexes.push(name_index);
        } else {
            candidates
                .extend((0..sources.files_count()).map(|file_index| (name_index, file_index)));
        }
    }

    let patterns = literal_name_indexes
        .iter()
        .map(|&name_index| crate_names[name_index])
        .chain(ASCII_CASE_FOLDING_CHARS);
    let automaton = AhoCorasick::builder()
        .ascii_case_insensitive(true)
        .build(patterns)?;

    // The last file each pattern was found in, to report each pattern once per file.
    let mut last_file_index = vec![usize::MAX; literal_name_indexes.len()];

    for file_index in 0..sources.files_count() {
        let mut found_count = 0;
        let mut found_case_folding_char = false;

        for found in automaton.find_overlapping_iter(sources.file(file_index)) {
            let pattern_index = found.pattern().as_usize();
            let Some(&name_index) = literal_name_indexes.get(pattern_index) else {
                found_case_folding_char = true;
                continue;
            };
            if last_file_index[pattern_index] != file_index {
                last_file_index[pattern_index] = file_index;
                candidates.push((name_index, file_index));
                found_count += 1;
                if found_count == literal_name_indexes.len() {
                    break;
                }
            }
        }

        if found_case_folding_char {
            candidates.extend(
                literal_name_indexes
                    .iter()
                    .map(|&name_index| (name_index, file_index)),
            );
        }
    }

    candidates.sort_unstable();
    candidates.dedup();
    Ok(candidates)
}

/// Returns whether any of the candidate files uses the crate.
fn is_used_in_any(
    matchers: &CrateMatchers,
    candidates: &[(usize, usize)],
    sources: &SourceFiles,
    searchers: &mut Searchers,
) -> bool {
    for &(_, file_index) in candidates {
        let path = &sources.paths[file_index];
        trace!(
            "looking for {} in {}",
            matchers.crate_name,
            path.to_string_lossy()
        );
        match searchers.is_used_in(matchers, sources.file(file_index)) {
            Ok(true) => return true,
            Ok(false) => {}
            Err(err) => eprintln!("{}: {}", path.display(), err),
        }
    }
    false
}

/// A dependency, as searched for in the package's sources.
struct Dependency {
    /// The name of the key in the Cargo.toml dependency table. Can have dashes, and isn't
    /// necessarily the name in the crate registry.
    key: String,
    /// The name the crate is used with in the sources (`extern crate`, snake case).
    crate_name: String,
}

/// A package whose sources are ready to be searched for usages of its dependencies.
struct PreparedPackage {
    analysis: PackageAnalysis,
    /// Sorted by key.
    dependencies: Vec<Dependency>,
    /// The `workspace.metadata.cargo-machete.ignored` dependencies of the package's workspace.
    workspace_ignored: HashSet<String>,
    sources: SourceFiles,
    /// The sorted `(dependency index, file index)` pairs where the file may use the dependency.
    candidates: Vec<(usize, usize)>,
}

impl PreparedPackage {
    fn candidates_of(&self, dependency_index: usize) -> &[(usize, usize)] {
        let start = self
            .candidates
            .partition_point(|&(index, _)| index < dependency_index);
        let end = self
            .candidates
            .partition_point(|&(index, _)| index <= dependency_index);
        &self.candidates[start..end]
    }
}

/// Searching for a dependency of a package.
struct DependencySearch<'a> {
    crate_name: &'a str,
    package_index: usize,
    dependency_index: usize,
}

/// Returns, for each package, whether each of its dependencies is used in its sources.
///
/// Searches are grouped by crate name, since most crate names (`serde`, `log`, …) are depended
/// upon by many packages: the matchers of a crate name are compiled once, run over all the
/// packages depending on it, and dropped. Compiling them per package would dominate the run time,
/// while keeping all of them around would take hundreds of MB on large workspaces.
fn search_packages(packages: &[PreparedPackage]) -> Vec<Vec<bool>> {
    let mut searches = Vec::new();
    for (package_index, package) in packages.iter().enumerate() {
        for (dependency_index, dependency) in package.dependencies.iter().enumerate() {
            // A dependency without candidate files is unused, no need to compile anything.
            if !package.candidates_of(dependency_index).is_empty() {
                searches.push(DependencySearch {
                    crate_name: &dependency.crate_name,
                    package_index,
                    dependency_index,
                });
            }
        }
    }
    searches.sort_unstable_by_key(|search| search.crate_name);

    let used_dependencies: Vec<(usize, usize)> = searches
        .par_chunk_by(|a, b| a.crate_name == b.crate_name)
        .map_init(Searchers::new, |searchers, searches| {
            search_crate_name(searches, packages, searchers)
        })
        .flatten_iter()
        .collect();

    let mut used: Vec<Vec<bool>> = packages
        .iter()
        .map(|package| vec![false; package.dependencies.len()])
        .collect();
    for (package_index, dependency_index) in used_dependencies {
        used[package_index][dependency_index] = true;
    }
    used
}

/// Runs searches which all have the same crate name, returning the `(package index, dependency
/// index)` pairs of the dependencies which are used.
fn search_crate_name(
    searches: &[DependencySearch<'_>],
    packages: &[PreparedPackage],
    searchers: &mut Searchers,
) -> Vec<(usize, usize)> {
    let crate_name = searches
        .first()
        .expect("searches are grouped in non-empty chunks")
        .crate_name;
    let matchers = CrateMatchers::new(crate_name).expect("constructing grep context");

    let mut used_dependencies = Vec::new();
    for search in searches {
        assert!(search.crate_name == crate_name);
        let package = &packages[search.package_index];
        let candidates = package.candidates_of(search.dependency_index);
        if is_used_in_any(&matchers, candidates, &package.sources, searchers) {
            used_dependencies.push((search.package_index, search.dependency_index));
        }
    }
    used_dependencies
}

struct FullManifest {
    manifest: cargo_toml::Manifest<PackageMetadata>,
    /// The `workspace.metadata.cargo-machete` section of the workspace manifest, if any.
    workspace_metadata: Option<MetadataFields>,
    /// The canonical path to the workspace manifest, if one was found.
    workspace_manifest_path: Option<PathBuf>,
}

/// Read a manifest and try to find a workspace manifest to complete the data available in the
/// manifest.
///
/// This will look up the file tree to find the Cargo.toml workspace manifest, assuming it's on a
/// parent directory.
fn get_full_manifest(dir_path: &Path, manifest_path: &Path) -> anyhow::Result<FullManifest> {
    // HACK: we can't plain use `from_path_with_metadata` here, because it calls
    // `complete_from_path` just a bit too early (before we've had a chance to call
    // `inherit_workspace`). See https://gitlab.com/crates.rs/cargo_toml/-/issues/20 for details,
    // and a possible future fix.
    let cargo_toml_content = std::fs::read(manifest_path)?;
    let mut manifest =
        cargo_toml::Manifest::<PackageMetadata>::from_slice_with_metadata(&cargo_toml_content)?;

    let mut ws_manifest_and_path = None;

    // Canonicalize the path, so as to get the full "parenthood" of relative paths.
    let mut dir_path = std::fs::canonicalize(dir_path).unwrap_or_else(|err| {
        warn!("error when canonicalizing dir_path: {err}");
        dir_path.to_owned()
    });

    // Try to find a workspace manifest, starting from the current directory, going up to the
    // filesystem's root.
    while dir_path.pop() {
        let workspace_cargo_path = dir_path.join("Cargo.toml");
        if let Ok(workspace_manifest) =
            cargo_toml::Manifest::<PackageMetadata>::from_path_with_metadata(&workspace_cargo_path)
            && workspace_manifest.workspace.is_some()
        {
            ws_manifest_and_path = Some((workspace_manifest, workspace_cargo_path));
            break;
        }
    }

    manifest.complete_from_path_and_workspace(
        manifest_path,
        ws_manifest_and_path.as_ref().map(|(m, p)| (m, p.as_path())),
    )?;

    let (workspace_metadata, workspace_manifest_path) = match ws_manifest_and_path {
        Some((ws_manifest, ws_path)) => (
            // Look for `workspace.metadata.cargo-machete` custom metadata in the workspace
            // Cargo.toml.
            ws_manifest
                .workspace
                .and_then(|workspace| workspace.metadata?.cargo_machete),
            Some(ws_path),
        ),
        None => (None, None),
    };

    Ok(FullManifest {
        manifest,
        workspace_metadata,
        workspace_manifest_path,
    })
}

/// Searches the packages at the given manifest paths for unused dependencies.
///
/// Returns one result per manifest path, in the same order, which is `Ok(None)` for a virtual
/// manifest. The sources of all the packages are held in memory during the search.
pub(crate) fn find_unused(
    manifest_paths: &[PathBuf],
    with_cargo_metadata: UseCargoMetadata<'_>,
) -> Vec<anyhow::Result<Option<PackageAnalysis>>> {
    let manifests: Vec<anyhow::Result<Option<PackageManifest>>> = manifest_paths
        .par_iter()
        .map(|manifest_path| read_manifest(manifest_path))
        .collect();

    let metadata_manifest_paths = match with_cargo_metadata {
        UseCargoMetadata::Yes(_) => {
            let mut paths: Vec<PathBuf> = manifests
                .iter()
                .filter_map(|manifest| manifest.as_ref().ok()?.as_ref())
                .map(|manifest| manifest.metadata_manifest_path().to_owned())
                .collect();
            paths.sort_unstable();
            paths.dedup();
            paths
        }
        UseCargoMetadata::No => Vec::new(),
    };

    let prepared: Vec<anyhow::Result<Option<PreparedPackage>>> = std::thread::scope(|scope| {
        if let UseCargoMetadata::Yes(cache) = with_cargo_metadata {
            // `cargo metadata` takes a while on large workspaces: run it while reading sources.
            spawn_metadata_loads(scope, cache, &metadata_manifest_paths);
        }

        let read: Vec<anyhow::Result<Option<(PackageManifest, SourceFiles)>>> = manifests
            .into_par_iter()
            .map(|manifest| {
                Ok(manifest?.map(|manifest| {
                    let sources = read_sources(&manifest);
                    (manifest, sources)
                }))
            })
            .collect();

        // This waits for `cargo metadata` when needed.
        read.into_par_iter()
            .map(|read| match read? {
                Some((manifest, sources)) => {
                    prepare_package(manifest, sources, with_cargo_metadata).map(Some)
                }
                None => Ok(None),
            })
            .collect()
    });

    // Whether each manifest describes a package, and the packages themselves in a contiguous
    // array, in the same order.
    let mut packages = Vec::with_capacity(prepared.len());
    let outcomes: Vec<anyhow::Result<bool>> = prepared
        .into_iter()
        .map(|result| {
            result.map(|package| {
                let is_package = package.is_some();
                packages.extend(package);
                is_package
            })
        })
        .collect();

    let used = search_packages(&packages);
    let mut analyses = packages
        .into_iter()
        .zip(used)
        .map(|(package, used)| finish_package(package, &used));

    outcomes
        .into_iter()
        .map(|outcome| {
            outcome.map(|is_package| {
                is_package.then(|| analyses.next().expect("one analysis per prepared package"))
            })
        })
        .collect()
}

/// Searches a single package for unused dependencies.
#[cfg(test)]
fn find_unused_single(
    manifest_path: &Path,
    with_cargo_metadata: UseCargoMetadata<'_>,
) -> anyhow::Result<Option<PackageAnalysis>> {
    find_unused(&[manifest_path.to_owned()], with_cargo_metadata)
        .pop()
        .expect("one result per manifest path")
}

/// A package's manifest, completed with its workspace's.
struct PackageManifest {
    analysis: PackageAnalysis,
    dir_path: PathBuf,
    /// The canonical path to the manifest.
    manifest_path: PathBuf,
    /// The canonical path to the manifest of the workspace the package most likely belongs to,
    /// unless the package is a workspace root itself.
    workspace_manifest_path: Option<PathBuf>,
    workspace_metadata: Option<MetadataFields>,
}

impl PackageManifest {
    /// The manifest `cargo metadata` most likely has to run on to describe this package.
    fn metadata_manifest_path(&self) -> &Path {
        self.workspace_manifest_path
            .as_deref()
            .unwrap_or(&self.manifest_path)
    }
}

/// Reads a package's manifest. Returns `Ok(None)` for a virtual manifest.
fn read_manifest(manifest_path: &Path) -> anyhow::Result<Option<PackageManifest>> {
    let mut dir_path = manifest_path.to_path_buf();
    dir_path.pop();

    trace!("trying to open {}...", manifest_path.display());

    let FullManifest {
        manifest,
        workspace_metadata,
        workspace_manifest_path,
    } = get_full_manifest(&dir_path, manifest_path)?;

    let package_name = match manifest.package {
        Some(ref package) => package.name.clone(),
        None => return Ok(None),
    };

    debug!("handling {} ({})", package_name, dir_path.display());

    // A package that declares a workspace is that workspace's root, whatever its parent
    // directories contain.
    let workspace_manifest_path = if manifest.workspace.is_some() {
        None
    } else {
        workspace_manifest_path
    };

    Ok(Some(PackageManifest {
        analysis: PackageAnalysis::new(package_name, manifest),
        dir_path,
        manifest_path: canonicalize_or_keep(manifest_path),
        workspace_manifest_path,
        workspace_metadata,
    }))
}

fn read_sources(manifest: &PackageManifest) -> SourceFiles {
    let paths = collect_paths(&manifest.dir_path, &manifest.analysis);
    let sources = SourceFiles::read(paths);
    debug!(
        "read {} files ({} bytes) of {}",
        sources.files_count(),
        sources.bytes.len(),
        manifest.analysis.package_name
    );
    sources
}

/// Finds which source files may use each dependency of a package.
fn prepare_package(
    manifest: PackageManifest,
    sources: SourceFiles,
    with_cargo_metadata: UseCargoMetadata<'_>,
) -> anyhow::Result<PreparedPackage> {
    let PackageManifest {
        analysis,
        manifest_path,
        workspace_manifest_path,
        workspace_metadata,
        ..
    } = manifest;

    let resolved_dependencies = match with_cargo_metadata {
        UseCargoMetadata::Yes(cache) => {
            cache.for_package(&manifest_path, workspace_manifest_path.as_deref())?
        }
        UseCargoMetadata::No => None,
    };

    // TODO extend to dev dependencies + build dependencies, and be smarter in the grouping of
    // searched paths
    // Maps dependency name (the name of the key in the Cargo.toml dependency
    // table, can have dashes, not necessarily the name in the crate registry)
    // to crate name (extern crate, snake case)
    let dependencies: ResolvedDependencies = resolved_dependencies.unwrap_or_else(|| {
        analysis
            .manifest
            .dependencies
            .keys()
            .map(|k| (k.clone(), k.replace('-', "_")))
            .collect()
    });

    // Keep a list of renamed dependencies
    static NO_RENAMED: BTreeMap<Box<str>, Box<str>> = BTreeMap::new();
    let renamed = package_machete_metadata(&analysis)
        .map(|meta| &meta.renamed)
        .unwrap_or(&NO_RENAMED);

    let (workspace_ignored, workspace_renamed): (HashSet<_>, _) = workspace_metadata
        .map(|MetadataFields { ignored, renamed }| (HashSet::from_iter(ignored), renamed))
        .unwrap_or_default();

    let dependencies: Vec<Dependency> = dependencies
        .into_iter()
        .map(|(key, crate_name)| {
            // If the crate was renamed (in the current Cargo.toml), use the renamed name.
            let crate_name = renamed
                .get(key.as_str())
                // Also try to look up the renames in the custom workspace metadata
                .or_else(|| workspace_renamed.get(key.as_str()))
                // fall-back to the crate name
                .map_or(crate_name, |renamed| renamed.to_string());
            Dependency { key, crate_name }
        })
        .collect();

    let crate_names: Vec<&str> = dependencies
        .iter()
        .map(|dependency| dependency.crate_name.as_str())
        .collect();
    let candidates = candidate_files(&crate_names, &sources)?;

    Ok(PreparedPackage {
        analysis,
        dependencies,
        workspace_ignored,
        sources,
        candidates,
    })
}

/// The `package.metadata.cargo-machete` section of the package's manifest.
fn package_machete_metadata(analysis: &PackageAnalysis) -> Option<&MetadataFields> {
    analysis
        .manifest
        .package
        .as_ref()
        .and_then(|package| package.metadata.as_ref()?.cargo_machete.as_ref())
}

/// Reports the unused dependencies of a package, given whether each one is used.
fn finish_package(package: PreparedPackage, used: &[bool]) -> PackageAnalysis {
    let PreparedPackage {
        mut analysis,
        dependencies,
        workspace_ignored,
        ..
    } = package;
    assert!(used.len() == dependencies.len());

    // Keep a side-list of ignored dependencies (likely false positives).
    let ignored: HashSet<String> = package_machete_metadata(&analysis)
        .map(|meta| meta.ignored.iter().cloned().collect())
        .unwrap_or_default();

    for (dependency, &is_used) in dependencies.into_iter().zip(used) {
        if !is_used {
            if !ignored.contains(&dependency.key) && !workspace_ignored.contains(&dependency.key) {
                analysis.unused.push(dependency.key);
            }
        } else if ignored.contains(&dependency.key) {
            analysis.ignored_used.push(dependency.key);
        }
    }

    analysis
}

struct StopAfterFirstMatch {
    found: bool,
}

impl StopAfterFirstMatch {
    fn new() -> Self {
        Self { found: false }
    }
}

impl Sink for StopAfterFirstMatch {
    type Error = Box<dyn error::Error>;

    fn matched(
        &mut self,
        _searcher: &searcher::Searcher,
        matsh: &searcher::SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        let mat = std::str::from_utf8(matsh.bytes())?;
        let mat = mat.trim();

        if mat.starts_with("//") || mat.starts_with("//!") {
            // Continue if seeing what resembles a comment or doc comment. Unfortunately we can't
            // do anything better because trying to figure whether we're within a (doc) comment
            // would require actual parsing of the Rust code.
            return Ok(true);
        }

        // Otherwise, we've found it: mark to true, and return false to indicate that we can stop
        // searching.
        self.found = true;
        Ok(false)
    }
}

#[test]
fn test_regexp() -> anyhow::Result<()> {
    fn test_one(crate_name: &str, content: &str) -> anyhow::Result<bool> {
        let used =
            Searchers::new().is_used_in(&CrateMatchers::new(crate_name)?, content.as_bytes())?;

        // The prefilter must never skip a file the regular expressions match.
        let sources = SourceFiles::from_contents(&[content]);
        let is_candidate = !candidate_files(&[crate_name], &sources)?.is_empty();
        assert!(!used || is_candidate, "{crate_name} skipped in {content:?}");

        Ok(used)
    }

    assert!(!test_one("log", "use da_force_luke;")?);
    assert!(!test_one("log", "use flog;")?);
    assert!(!test_one("log", "use log_once;")?);
    assert!(!test_one("log", "use log_once::info;")?);
    assert!(!test_one("log", "use flog::flag;")?);
    assert!(!test_one("log", "flog::flag;")?);
    assert!(!test_one("log", "use ::flog;")?);
    assert!(!test_one("log", "use :log;")?);

    assert!(test_one("log", "use log;")?);
    assert!(test_one("log", "use ::log;")?);
    assert!(test_one("log", "use log::{self};")?);
    assert!(test_one("log", "use log::*;")?);
    assert!(test_one("log", "use log::info;")?);
    assert!(test_one("log", "use log as logging;")?);
    assert!(test_one("log", "extern crate log;")?);
    assert!(test_one("log", "extern crate log as logging")?);
    assert!(test_one("log", r#"log::info!("fyi")"#)?);

    assert!(test_one("Log", "use log;")?);
    assert!(test_one("Log", "use ::log;")?);
    assert!(test_one("Log", "use log::{self};")?);
    assert!(test_one("Log", "use log::*;")?);
    assert!(test_one("Log", "use log::info;")?);
    assert!(test_one("Log", "use log as logging;")?);
    assert!(test_one("Log", "extern crate log;")?);
    assert!(test_one("Log", "extern crate log as logging")?);
    assert!(test_one("Log", r#"log::info!("fyi")"#)?);

    assert!(test_one("log", "use Log;")?);
    assert!(test_one("log", "use ::Log;")?);
    assert!(test_one("log", "use Log::{self};")?);
    assert!(test_one("log", "use Log::*;")?);
    assert!(test_one("log", "use Log::info;")?);
    assert!(test_one("log", "use Log as logging;")?);
    assert!(test_one("log", "extern crate Log;")?);
    assert!(test_one("log", "extern crate Log as logging")?);
    assert!(test_one("log", r#"Log::info!("fyi")"#)?);

    assert!(test_one(
        "bitflags",
        r#"
use std::fmt;
bitflags::macro! {
"#
    )?);

    assert!(test_one(
        "Bitflags",
        r#"
use std::fmt;
bitflags::macro! {
"#
    )?);

    assert!(test_one(
        "bitflags",
        r#"
use std::fmt;
Bitflags::macro! {
"#
    )?);

    // Compound `use as` statements. Here come the nightmares...
    assert!(test_one("log", "use { log as logging };")?);
    assert!(!test_one("lol", "use { log as logging };")?);

    assert!(test_one(
        "log",
        r#"
use {
    log as logging
};
"#
    )?);

    assert!(test_one(
        "log",
        r#"
use { log as
logging
};
"#
    )?);

    assert!(test_one(
        "log",
        r#"
use { log
    as
        logging
};
"#
    )?);

    assert!(test_one(
        "log",
        r#"
use {
    x::{ y },
    log as logging,
};
"#
    )?);

    // Regex must stop at the first };
    assert!(!test_one(
        "log",
        r#"
use {
    x as y
};
type logging = u64;
fn main() {
    let func = |log: u32| {
        log as logging
    };
    func(42);
}
"#
    )?);

    // Regression test.
    // Comments and spaces are meaningful here.
    assert!(test_one(
        "static_assertions",
        r#"
    // lol
    static_assertions::assert_not_impl_all!(A: B);
    "#
    )?);

    // Regression test.
    // Comments and spaces are meaningful here.
    assert!(test_one(
        "futures",
        r#"
// the [`futures::executor::block_on`] function
pub use futures::future;

    "#
    )?);

    // multi-dep single use statements
    assert!(test_one(
        "futures",
        r#"pub use {async_trait, futures, reqwest};"#
    )?);

    // multi-dep single use statements with ::
    assert!(test_one(
        "futures",
        r#"pub use {async_trait, ::futures, reqwest};"#
    )?);

    // No false usage detection of `not_my_dep::my_dep` on compound imports
    assert!(!test_one(
        "futures",
        r#"pub use {async_trait, not_futures::futures, reqwest};"#
    )?);

    // No false usage detection of `not_my_dep::my_dep` on multiple lines
    assert!(!test_one(
        "futures",
        r#"
pub use {
    async_trait,
    not_futures::futures,
    reqwest,
};"#
    )?);

    // No false usage detection on single line `not_my_dep::my_dep`
    assert!(!test_one(
        "futures",
        r#"use not_futures::futures::stuff_in_futures;"#
    )?);

    // multi-dep single use statements with nesting
    assert!(test_one(
        "futures",
        r#"pub use {
            async_trait::{mod1, dep2},
            futures::{futures_mod1, futures_mod2::{futures_mod21, futures_mod22}},
            reqwest,
        };"#
    )?);

    // multi-dep single use statements with star import and renaming
    assert!(test_one(
        "futures",
        r#"pub use {
            async_trait::sub_mod::*,
            futures as futures_renamed,
            reqwest,
        };"#
    )?);

    // multi-dep single use statements with complex imports and renaming
    assert!(test_one(
        "futures",
        r#"pub use {
            other_dep::{
                star_mod::*,
                unnamed_import::{UnnamedTrait as _, other_mod},
                renamed_import as new_name,
                sub_import::{mod1, mod2},
            },
            futures as futures_renamed,
            reqwest,
        };"#
    )?);

    // No false usage detection of `not_my_dep::my_dep` with nesting
    assert!(!test_one(
        "futures",
        r#"pub use {
            async_trait::{mod1, dep2},
            not_futures::futures::{futures_mod1, futures_mod2::{futures_mod21, futures_mod22}},
            reqwest,
        };"#
    )?);

    // Detects top level usage
    assert!(test_one("futures", r#" ::futures::mod1"#)?);

    Ok(())
}

#[cfg(test)]
fn check_analysis<F: Fn(PackageAnalysis)>(rel_path: &str, callback: F) {
    let cache = MetadataCache::default();
    for use_cargo_metadata in UseCargoMetadata::all(&cache) {
        let analysis =
            find_unused_single(&PathBuf::from(TOP_LEVEL).join(rel_path), use_cargo_metadata)
                .expect("find_unused must return an Ok result")
                .expect("no error during processing");
        callback(analysis);
    }
}

#[test]
fn test_just_unused() {
    // a crate that simply does not use a dependency it refers to
    check_analysis("./integration-tests/just-unused/Cargo.toml", |analysis| {
        assert_eq!(analysis.unused, &["log".to_string()]);
    });
}

#[test]
fn test_just_unused_with_manifest() {
    // a crate that does not use a dependency it refers to, and uses workspace properties
    check_analysis(
        "./integration-tests/workspace-package/program/Cargo.toml",
        |analysis| {
            assert_eq!(analysis.unused, &["log".to_string()]);
        },
    );
}

#[test]
fn test_unused_transitive() {
    // lib1 has zero dependencies
    check_analysis(
        "./integration-tests/unused-transitive/lib1/Cargo.toml",
        |analysis| {
            assert!(analysis.unused.is_empty());
        },
    );

    // lib2 effectively uses lib1
    check_analysis(
        "./integration-tests/unused-transitive/lib2/Cargo.toml",
        |analysis| {
            assert!(analysis.unused.is_empty());
        },
    );

    // but top level references both lib1 and lib2, and only uses lib2
    check_analysis(
        "./integration-tests/unused-transitive/Cargo.toml",
        |analysis| {
            assert_eq!(analysis.unused, &["lib1".to_string()]);
        },
    );
}

#[test]
fn test_false_positive_macro_use() {
    // when a lib uses a dependency via a macro, there's no way we can find it by scanning the
    // source code.
    check_analysis(
        "./integration-tests/false-positive-log/Cargo.toml",
        |analysis| {
            assert_eq!(analysis.unused, &["log".to_string()]);
        },
    );
}

#[test]
fn test_with_bench() {
    // when a package has a bench file designated by binary name, it seems that `cargo_toml`
    // doesn't fill in a default path to the source code.
    check_analysis(
        "./integration-tests/with-bench/bench/Cargo.toml",
        |analysis| {
            assert!(analysis.unused.is_empty());
        },
    );
}

#[test]
fn test_renamed_field_works() -> anyhow::Result<()> {
    // cargo-machete properly handles a correct rename (rustls-webpki -> webpki), when the rename
    // happens in the crate manifest
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/renamed-dep/Cargo.toml"),
        UseCargoMetadata::No,
    )?
    .expect("no error during processing");
    assert_eq!(analysis.unused.as_slice(), &["bytes", "log"]);
    Ok(())
}

#[test]
fn test_renamed_field_workspace_works() -> anyhow::Result<()> {
    // cargo-machete properly handles a correct rename (rustls-webpki -> webpki), when the rename
    // happens in the workspace manifest
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL)
            .join("./integration-tests/renamed-dep-workspace/inner/Cargo.toml"),
        UseCargoMetadata::No,
    )?
    .expect("no error during processing");
    assert_eq!(analysis.unused.as_slice(), &["bytes", "flagset"]);
    Ok(())
}

#[test]
fn test_crate_renaming_works() -> anyhow::Result<()> {
    // when a lib like xml-rs is exposed with a different name, cargo-machete doesn't return false
    // positives.
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/renaming-works/Cargo.toml"),
        UseCargoMetadata::Yes(&MetadataCache::default()),
    )?
    .expect("no error during processing");
    assert!(analysis.unused.is_empty());

    // But when not using cargo-metadata, there's a false positive!
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/renaming-works/Cargo.toml"),
        UseCargoMetadata::No,
    )?
    .expect("no error during processing");
    assert_eq!(analysis.unused, &["xml-rs".to_string()]);

    Ok(())
}

#[test]
fn test_unused_renamed_in_registry() -> anyhow::Result<()> {
    // when a lib like xml-rs is exposed with a different name,
    // cargo-machete reports the unused spec properly.
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/unused-renamed-in-registry/Cargo.toml"),
        UseCargoMetadata::Yes(&MetadataCache::default()),
    )?
    .expect("no error during processing");
    assert_eq!(analysis.unused, &["xml-rs".to_string()]);

    Ok(())
}

#[test]
fn test_unused_renamed_in_spec() -> anyhow::Result<()> {
    // when a lib is renamed through key = { package = … },
    // cargo-machete reports the unused spec properly.
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/unused-renamed-in-spec/Cargo.toml"),
        UseCargoMetadata::Yes(&MetadataCache::default()),
    )?
    .expect("no error during processing");
    assert_eq!(analysis.unused, &["tracing".to_string()]);

    Ok(())
}

#[test]
fn test_unused_kebab_spec() -> anyhow::Result<()> {
    // when a lib uses kebab naming, cargo-machete reports the unused spec properly.
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/unused-kebab-spec/Cargo.toml"),
        UseCargoMetadata::Yes(&MetadataCache::default()),
    )?
    .expect("no error during processing");
    assert_eq!(analysis.unused, &["log-once".to_string()]);

    Ok(())
}

#[test]
fn test_ignore_deps_works() {
    // ensure that ignored deps listed in Cargo.toml package.metadata.cargo-machete.ignored are
    // correctly ignored.
    check_analysis("./integration-tests/ignored-dep/Cargo.toml", |analysis| {
        assert_eq!(analysis.unused, &["rand".to_string()]);
        assert_eq!(analysis.ignored_used, &["rand_core".to_string()]);
    });
}

#[test]
fn test_ignore_deps_workspace_works() {
    // ensure that ignored deps listed in Cargo.toml workspace.metadata.cargo-machete.ignored are
    // correctly ignored.
    check_analysis(
        "./integration-tests/ignored-dep-workspace/inner/Cargo.toml",
        |analysis| {
            assert_eq!(analysis.unused, &["rand".to_string()]);
            assert_eq!(analysis.ignored_used, &["rand_core".to_string()]);
        },
    );
}

#[test]
fn test_workspace_from_relative_path() {
    // Ensure that finding a workspace using a relative path works.
    use std::env::{current_dir, set_current_dir};

    let prev_cwd = current_dir().unwrap();

    set_current_dir(
        PathBuf::from(TOP_LEVEL).join("./integration-tests/workspace-package/program/"),
    )
    .unwrap();

    let path = Path::new("./Cargo.toml");
    let analysis = find_unused_single(path, UseCargoMetadata::No);

    // Reset the current directory *before* running any other check.
    set_current_dir(prev_cwd).unwrap();

    let analysis = analysis
        .expect("find_unused must return an Ok result")
        .expect("no error during processing");

    assert_eq!(analysis.unused, &["log".to_string()]);
    assert!(analysis.ignored_used.is_empty());
}

#[test]
fn test_multi_key_dep() {
    let analysis = find_unused_single(
        &PathBuf::from(TOP_LEVEL).join("./integration-tests/multi-key-dep/Cargo.toml"),
        UseCargoMetadata::Yes(&MetadataCache::default()),
    )
    .expect("find_unused must return an Ok result")
    .expect("no error during processing");

    assert_eq!(analysis.unused, &["cc".to_string(), "rand".to_string()]);
}

#[test]
fn test_workspace_shares_metadata() -> anyhow::Result<()> {
    let cache = MetadataCache::default();
    let root = PathBuf::from(TOP_LEVEL).join("./integration-tests/workspace-shared-metadata/");

    // A renamed dependency of a workspace member which isn't the one `cargo metadata` would
    // consider as its root.
    let analysis = find_unused_single(
        &root.join("first/Cargo.toml"),
        UseCargoMetadata::Yes(&cache),
    )?
    .expect("first is a package");
    assert_eq!(analysis.unused, &["rng".to_string()]);
    assert!(analysis.ignored_used.is_empty());

    let analysis = find_unused_single(
        &root.join("second/Cargo.toml"),
        UseCargoMetadata::Yes(&cache),
    )?
    .expect("second is a package");
    assert_eq!(analysis.unused, &["log".to_string()]);
    assert!(analysis.ignored_used.is_empty());

    // Both members got their metadata from a single `cargo metadata` invocation.
    assert_eq!(cache.loaded_count(), 1);

    // A package excluded from the workspace falls back to its own `cargo metadata` invocation.
    let analysis = find_unused_single(
        &root.join("excluded/Cargo.toml"),
        UseCargoMetadata::Yes(&cache),
    )?
    .expect("excluded is a package");
    assert_eq!(analysis.unused, &["log".to_string()]);
    assert_eq!(cache.loaded_count(), 2);

    // The virtual manifest isn't analyzed and doesn't run `cargo metadata`.
    assert!(
        find_unused_single(&root.join("Cargo.toml"), UseCargoMetadata::Yes(&cache),)?.is_none()
    );
    assert_eq!(cache.loaded_count(), 2);

    Ok(())
}

#[test]
fn test_candidate_files() -> anyhow::Result<()> {
    let sources = SourceFiles::from_contents(&[
        "use log::info;\nuse log_once;",
        "SERDE::to_string(&x);",
        "\u{212A}it::build();",
        "fn main() {}",
    ]);
    let crate_names = ["log", "log_once", "serde", "foo.bar", "kit"];

    assert_eq!(
        candidate_files(&crate_names, &sources)?,
        &[
            // `log` overlaps with `log_once`, and both are found.
            (0, 0),
            // The Kelvin sign in file 2 may match any `k`, so every name is a candidate there.
            (0, 2),
            (1, 0),
            (1, 2),
            // ASCII case is ignored.
            (2, 1),
            (2, 2),
            // A name with regex syntax is searched in every file.
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
        ]
    );

    assert!(candidate_files(&[], &sources)?.is_empty());
    assert!(candidate_files(&crate_names, &SourceFiles::from_contents(&[]))?.is_empty());
    assert!(candidate_files(&["log"], &SourceFiles::from_contents(&["use flag;"]))?.is_empty());

    // The case folding the prefilter accounts for does happen in the regular expressions.
    assert!(Searchers::new().is_used_in(
        &CrateMatchers::new("kit")?,
        "\u{212A}it::build();".as_bytes()
    )?);

    Ok(())
}

#[test]
fn test_source_files_stop_at_nul() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join(format!("cargo-machete-nul-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let binary = dir.join("binary.rs");
    let text = dir.join("text.rs");
    std::fs::write(&binary, b"use log;\x00use serde;")?;
    std::fs::write(&text, b"use rand;")?;

    let sources = SourceFiles::read(vec![binary, dir.join("missing.rs"), text]);
    std::fs::remove_dir_all(&dir)?;

    // The missing file is skipped, and the binary one is cut at its NUL byte.
    assert_eq!(sources.files_count(), 2);
    assert_eq!(sources.file(0), b"use log;");
    assert_eq!(sources.file(1), b"use rand;");

    Ok(())
}

#[test]
fn test_find_unused_batch() {
    let root = PathBuf::from(TOP_LEVEL).join("./integration-tests/workspace-shared-metadata/");
    let manifest_paths = [
        root.join("first/Cargo.toml"),
        root.join("Cargo.toml"),
        root.join("missing/Cargo.toml"),
        root.join("second/Cargo.toml"),
    ];

    let cache = MetadataCache::default();
    for use_cargo_metadata in UseCargoMetadata::all(&cache) {
        let mut results = find_unused(&manifest_paths, use_cargo_metadata).into_iter();

        // `log` is searched in both packages at once, and only used in `first`.
        let first = results
            .next()
            .unwrap()
            .unwrap()
            .expect("first is a package");
        assert_eq!(first.package_name, "first");
        assert_eq!(first.unused, &["rng".to_string()]);

        assert!(results.next().unwrap().unwrap().is_none());
        assert!(results.next().unwrap().is_err());

        let second = results
            .next()
            .unwrap()
            .unwrap()
            .expect("second is a package");
        assert_eq!(second.package_name, "second");
        assert_eq!(second.unused, &["log".to_string()]);

        assert!(results.next().is_none());
    }
}

#[test]
fn test_workspace_metadata_from_raw() -> anyhow::Result<()> {
    let dir = PathBuf::from(TOP_LEVEL).join("./integration-tests/workspace-shared-metadata/");
    let manifest_path = dir.join("first/Cargo.toml");
    let json = serde_json::json!({
        "packages": [
            {
                "id": "first-id",
                "name": "first",
                "manifest_path": manifest_path,
                "dependencies": [
                    { "name": "rand", "rename": "rng", "kind": null },
                    { "name": "log", "rename": null },
                    { "name": "log", "rename": null, "kind": "dev" },
                ],
                "unknown_field": { "is": "skipped" },
            },
            { "id": "rand-id", "name": "rand", "manifest_path": "/rand/Cargo.toml", "dependencies": [] },
            { "id": "log-id", "name": "log", "manifest_path": "/log/Cargo.toml", "dependencies": [] },
        ],
        "workspace_members": ["first-id"],
        "resolve": {
            "nodes": [{
                "id": "first-id",
                "deps": [
                    { "name": "rand", "pkg": "rand-id" },
                    { "name": "log", "pkg": "log-id" },
                ],
            }],
        },
    });

    let metadata: raw_metadata::Metadata = serde_json::from_value(json.clone())?;
    let workspace = WorkspaceMetadata::from_raw(&metadata);
    let canonical_manifest_path = canonicalize_or_keep(&manifest_path);
    assert_eq!(workspace.members.len(), 1);
    assert_eq!(
        workspace.members[&canonical_manifest_path],
        Some(BTreeMap::from([
            ("log".to_string(), "log".to_string()),
            ("rng".to_string(), "rand".to_string()),
        ]))
    );

    // Without a resolved dependency graph, the member is known but its dependencies aren't.
    let mut json = json;
    json["resolve"] = serde_json::Value::Null;
    let metadata: raw_metadata::Metadata = serde_json::from_value(json)?;
    let workspace = WorkspaceMetadata::from_raw(&metadata);
    assert_eq!(workspace.members[&canonical_manifest_path], None);

    // Missing fields are errors.
    assert!(serde_json::from_str::<raw_metadata::Metadata>(r#"{"packages": []}"#).is_err());

    Ok(())
}
