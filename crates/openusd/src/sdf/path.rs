use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{BuildHasherDefault, DefaultHasher, Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError, Weak};
use std::{array, cmp, convert::Infallible, fmt, ptr, str::FromStr};

use crate::tf;

/// Parses `str` into a validated [`Path`] — the terse form of [`Path::new`].
#[inline]
pub fn path(str: impl AsRef<str>) -> Result<Path, PathParseError> {
    let path = str.as_ref();
    Path::new(path)
}

/// Bound alias for path parameters: any argument convertible into a [`Path`]
/// via [`TryInto`] whose failure converts into a [`PathParseError`].
/// Blanket-implemented, so [`Path`], `&Path`, `&str`, and `String` all
/// satisfy it; a third-party type opts in by implementing `TryFrom<T> for
/// Path`.
pub trait IntoPath: TryInto<Path, Error: Into<PathParseError>> {}

impl<T: TryInto<Path, Error: Into<PathParseError>>> IntoPath for T {}

/// Converts a path argument into an owned [`Path`], surfacing the parse
/// error. The single conversion point the crate's generic
/// [`IntoPath`] parameters funnel through.
pub fn try_into_path(path: impl IntoPath) -> Result<Path, PathParseError> {
    path.try_into().map_err(Into::into)
}

/// `SdfPath` implementation.
///
/// # Syntax
/// - Two separators are used between parts of a path. A slash ("/")
///   following an identifier is used to introduce a namespace child.
/// - A period (".") following an identifier is used to introduce a property.
/// - A property may also have several non-sequential colons (':') in its name
///   to provide a rudimentary namespace within properties but may not end or
///   begin with a colon.
/// - Brackets ("[" and "]") are used to indicate relationship target paths for
///   relational attributes.
///
/// Parsing via [`Path::new`] (or [`FromStr`]) validates this grammar and
/// rejects malformed text with a [`PathParseError`]. The empty path is not
/// parseable; construct it with [`Path::default`].
///
/// Paths are interned, as C++ `SdfPath` pools its path nodes: every path
/// spelled the same shares one node, and equality compares node pointers.
/// Each node links to its parent's node, which [`parent`](Self::parent) and
/// [`ancestors`](Self::ancestors) follow. A node lives while a path refers
/// to it or to one of its descendants.
///
/// Hashing writes a hash of the text, computed once when the node is made.
/// Ordering compares the text. Both agree across runs.
#[derive(Debug, Clone)]
pub struct Path {
    node: Arc<Node>,
}

/// The node behind every [`Path`] spelled the same (C++ `Sdf_PathNode`).
///
/// [`NODES`] holds one live node per text. A node holds its parent's node,
/// whose text prefixes its own.
struct Node {
    text: Box<str>,
    /// The text's hash under a fixed-key hasher: equal texts hash alike in
    /// every run.
    hash: u64,
    /// The node of [`Path::parent`].
    parent: Option<Path>,
    /// Whether the path is a property path ([`Path::is_property`]).
    property: bool,
}

/// The empty path, kept so [`Path::default`] takes no lock.
static EMPTY: LazyLock<Path> = LazyLock::new(|| Path::intern(""));

/// The absolute root path, kept so [`Path::abs_root`] takes no lock.
static ABS_ROOT: LazyLock<Path> = LazyLock::new(|| Path::intern("/"));

/// Number of [`Shard`]s in [`NODES`].
const SHARDS: usize = 16;

/// The table of live [`Node`]s, split into shards each behind its own lock
/// so threads interning different paths rarely contend.
static NODES: LazyLock<[Mutex<Shard>; SHARDS]> = LazyLock::new(|| array::from_fn(|_| Mutex::default()));

/// One shard of [`NODES`]: the nodes whose hash selects it, keyed by that
/// hash. A node removes its own entry when dropped.
#[derive(Default)]
struct Shard {
    nodes: HashMap<u64, Weak<Node>, BuildHasherDefault<Prehashed>>,
    /// Nodes whose hash a different live node holds in `nodes`.
    collided: Vec<Weak<Node>>,
}

/// A hasher for keys that are hashes already: it passes the `u64` through.
#[derive(Default)]
struct Prehashed(u64);

impl Default for Path {
    /// The empty path.
    fn default() -> Self {
        EMPTY.clone()
    }
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.node, &other.node)
    }
}

impl Eq for Path {}

impl Hash for Path {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.node.hash);
    }
}

impl PartialOrd for Path {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Path {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self == other {
            return cmp::Ordering::Equal;
        }
        self.as_str().cmp(other.as_str())
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Path {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_str().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Path {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        // The empty path is a valid `Path` value (e.g. a `Reference` with no
        // prim path serializes to `""`), so round-tripping accepts it.
        if text.is_empty() {
            return Ok(Path::default());
        }
        Path::try_from(text).map_err(serde::de::Error::custom)
    }
}

/// Error produced when a string fails to parse as a [`Path`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid path {input:?}: {reason} (at byte {offset})")]
pub struct PathParseError {
    /// The offending path string.
    pub input: String,
    /// Byte offset in [`input`](Self::input) where the problem sits.
    pub offset: usize,
    /// What the string violates.
    pub reason: &'static str,
}

impl From<Infallible> for PathParseError {
    fn from(error: Infallible) -> Self {
        match error {}
    }
}

impl FromStr for Path {
    type Err = PathParseError;

    fn from_str(s: &str) -> Result<Path, Self::Err> {
        Path::validate(s)?;
        Ok(Path::intern(s))
    }
}

impl Path {
    /// Parses `path`, validating it against the path grammar described on
    /// the type. The empty path is not parseable; construct it with
    /// [`Path::default`].
    pub fn new(path: &str) -> Result<Self, PathParseError> {
        Path::from_str(path)
    }

    /// The absolute root path `/` (pseudo-root).
    #[inline]
    pub fn abs_root() -> Path {
        ABS_ROOT.clone()
    }

    /// The path spelled by already-validated `text`: its live node, or else
    /// a new node linked to its parent's, which is interned first.
    fn intern(text: &str) -> Path {
        let hash = hash_text(text);
        // The nodes `find` reads to compare, declared ahead of every shard
        // guard so they drop after it.
        let mut others = Vec::new();
        if let Some(path) = lock_shard(hash).find(hash, text, &mut others) {
            return path;
        }
        // The parent is interned outside the lock, as it may hash to the
        // same shard.
        let parent = parent_len(text).map(|len| Path::intern(&text[..len]));
        let mut shard = lock_shard(hash);
        // Another thread may have interned `text` since the lookup.
        if let Some(path) = shard.find(hash, text, &mut others) {
            return path;
        }
        let node = Arc::new(Node {
            text: text.into(),
            hash,
            parent,
            property: ends_in_property(text),
        });
        shard.insert(&node);
        Path { node }
    }

    /// Wraps `path` without validating it — the fast path for strings a
    /// derivation method recombined from already-validated paths. External
    /// input must go through [`Path::new`].
    pub(crate) fn from_str_unchecked(path: &str) -> Path {
        // Derivations of the empty path (e.g. `prim_path()` of `.bar`)
        // legitimately produce `""`, which `validate` rejects.
        debug_assert!(
            path.is_empty() || Path::validate(path).is_ok(),
            "from_str_unchecked on invalid path {path:?}"
        );
        match path {
            "" => Path::default(),
            "/" => Path::abs_root(),
            _ => Path::intern(path),
        }
    }

    #[inline]
    pub fn is_abs(&self) -> bool {
        self.as_str().starts_with('/')
    }

    /// Returns `true` if this is the absolute root path `/` (pseudo-root).
    #[inline]
    pub fn is_abs_root(&self) -> bool {
        self.as_str() == "/"
    }

    /// Whether this path's prim is a root prim — a direct child of the
    /// pseudo-root (`/Foo`, or a property of one like `/Foo.attr`): it has exactly
    /// one prim component, so its only namespace ancestor is the absolute root.
    #[inline]
    pub fn is_root_prim(&self) -> bool {
        self.prim_element_count() == 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }

    pub fn append_property(&self, property: impl Into<tf::Token>) -> Result<Path, PathParseError> {
        let property = property.into();
        let property = property.as_str();
        let fail = |offset, reason| PathParseError {
            input: format!("{self}.{property}"),
            offset,
            reason,
        };
        if self.is_property() {
            return Err(fail(0, "cannot append a property to a property path"));
        }
        // The pseudo-root, the empty path, and the `.`/`..` relative anchors
        // cannot own properties; appending to them would build an unparseable
        // path like `/.foo`.
        if self.is_empty() || self.is_abs_root() || self.as_str() == "." || self.as_str().ends_with("..") {
            return Err(fail(0, "path cannot own properties"));
        }
        if !Path::is_valid_namespace_identifier(property) {
            return Err(fail(
                self.as_str().len() + 1,
                "property name is not a valid namespaced identifier",
            ));
        }

        let mut new_path = self.as_str().to_string();
        new_path.push('.');
        new_path.push_str(property);

        Ok(Path::intern(&new_path))
    }

    /// Appends `path` (parsed if given as a string) under this path with a `/`
    /// separator (C++ `SdfPath::AppendPath`). Appending an absolute path to an
    /// absolute base, or anything to a property path, is an error.
    pub fn append_path(&self, path: impl TryInto<Path, Error: Into<PathParseError>>) -> Result<Path, PathParseError> {
        let append: Path = try_into_path(path)?;

        if self.is_abs() && append.is_abs() {
            return Err(PathParseError {
                input: append.as_str().to_string(),
                offset: 0,
                reason: "cannot append an absolute path to an absolute path",
            });
        }

        if self.is_property() {
            return Err(PathParseError {
                input: self.as_str().to_string(),
                offset: self.as_str().rfind('.').unwrap_or(0),
                reason: "cannot append a path to a property path",
            });
        }

        if append.as_str() == "." {
            return Ok(self.clone());
        }

        // The reflexive base is the identity anchor: appending to `.` yields
        // the argument itself (C++ `SdfPath::AppendPath` on the reflexive
        // relative path).
        if self.as_str() == "." {
            return Ok(append);
        }

        // A `.`-anchored argument (a `..` step or a property-relative `.attr`)
        // cannot attach under a prim namespace; the concatenation would not be
        // a valid path.
        if append.as_str().starts_with('.') {
            return Err(PathParseError {
                input: append.as_str().to_string(),
                offset: 0,
                reason: "cannot append a `.`-anchored path under a prim path",
            });
        }

        // If base is slash only.
        // "/" + "foo/bar" => "/foo/bar"
        let combined = if self.as_str() == "/" {
            format!("/{append}")
        } else if self.is_variant_selection() {
            // A prim child attaches directly to a variant selection with no
            // separator: "/A{v=s}" + "B" => "/A{v=s}B" (C++ SdfPath::AppendChild).
            format!("{self}{append}")
        } else {
            format!("{self}/{append}")
        };

        Ok(Path::intern(&combined))
    }

    /// Returns `true` if this path ends in a property name, e.g. `/A.attr`
    /// or `/A.rel[/T].attr`.
    #[inline]
    pub fn is_property(&self) -> bool {
        self.node.property
    }

    /// Returns `true` if this path's final component is a variant selection,
    /// e.g. `/Prim{set=sel}` — as opposed to a prim, property, or the root.
    ///
    /// Mirrors C++ `SdfPath::IsPrimVariantSelectionPath`. A variant selection
    /// path identifies a variant spec, not a prim, so it is not a valid target
    /// for prim authoring.
    pub fn is_variant_selection(&self) -> bool {
        self.as_str().ends_with('}')
    }

    /// Returns `true` if this path names a prim.
    ///
    /// Mirrors C++ `SdfPath::IsPrimPath`: true for an absolute or relative prim
    /// path and for the relative anchors `.` and `..`, which C++ builds as prim
    /// nodes; false for the pseudo-root, a variant selection, and every path
    /// carrying a property tail — a property, a relationship target, a
    /// relational attribute, a mapper.
    ///
    /// Decided by the final [`PathComponent`], so a tail the simpler predicates
    /// do not model still classifies correctly: `/A.rel[/Target]` leaves a
    /// remainder the prim-chain grammar cannot consume, and so is not a prim
    /// path.
    pub fn is_prim(&self) -> bool {
        // The relative anchors: `.`, and a `..(/..)*` run with nothing after it.
        if self.as_str() == "." || (!self.as_str().is_empty() && self.as_str().split('/').all(|seg| seg == "..")) {
            return true;
        }
        // Strip the anchor the grammar allows ahead of the prim chain, then ask
        // the prim-chain iterator what the last component was. A non-empty
        // remainder means a property tail the chain could not consume.
        let mut rest = self.as_str();
        if let Some(after) = rest.strip_prefix('/') {
            rest = after;
        } else {
            while let Some(after) = rest.strip_prefix("../") {
                rest = after;
            }
        }
        let mut components = PathComponents::over(rest);
        let last = components.by_ref().last();
        components.remainder().is_empty() && matches!(last, Some(PathComponent::Prim(_)))
    }

    /// Returns the variant set name of the deepest `{set=sel}` (or `{set=}`)
    /// selection in this path — `"set"` for both `/Prim{set=sel}` and the bare
    /// variant-set path `/Prim{set=}`. `None` when the path carries no variant
    /// selection.
    pub fn variant_set_name(&self) -> Option<&str> {
        self.components()
            .filter_map(|component| match component {
                PathComponent::Variant { set, .. } => Some(set),
                PathComponent::Prim(_) => None,
            })
            .last()
    }

    /// Returns `true` if any component of this path's own prim chain is a
    /// variant selection, e.g. both `/Prim{set=sel}` and `/Prim{set=sel}/Child`
    /// — including on a relative path (`../A{v=x}`).
    ///
    /// Mirrors C++ `SdfPath::ContainsPrimVariantSelection`. Unlike
    /// [`is_variant_selection`](Self::is_variant_selection),
    /// which only inspects the final component, this finds a selection embedded
    /// anywhere in the prim chain. A `{` inside a bracketed target component
    /// (`/A.rel[/T{v=x}]`) belongs to the embedded target path and does not
    /// count, matching
    /// [`strip_variant_selections`](Self::strip_variant_selections).
    pub fn has_variant_selection(&self) -> bool {
        // In a well-formed path a `{` outside a target bracket can only open a
        // `{set=sel}` segment, so a depth-0 scan decides without parsing
        // components (which relative `..` prefixes would stop short).
        let mut depth = 0usize;
        for byte in self.as_str().bytes() {
            match byte {
                b'[' => depth += 1,
                b']' => depth = depth.saturating_sub(1),
                b'{' if depth == 0 => return true,
                _ => {}
            }
        }
        false
    }

    /// Returns the prim this path names or belongs to: the path itself for a
    /// prim, the owning prim of a property, and the prim a trailing variant
    /// selection qualifies.
    pub fn prim_path(&self) -> Path {
        self.prefix(prim_len(self.as_str()))
    }

    /// The path spelled by the first `len` bytes of this one's text, which
    /// must end at an element boundary. It is `self` or the ancestor of that
    /// length when there is one, and is interned otherwise. Every parent's
    /// text prefixes its child's.
    fn prefix(&self, len: usize) -> Path {
        let mut path = self;
        loop {
            if path.as_str().len() == len {
                return path.clone();
            }
            match &path.node.parent {
                Some(parent) if parent.as_str().len() >= len => path = parent,
                _ => return Path::from_str_unchecked(&self.as_str()[..len]),
            }
        }
    }

    /// Returns the property portion of this path — the trailing segment after
    /// the owning prim, beginning with `.` (e.g. `.radius` in `/Light.radius`,
    /// `.foo[/T].bar` for a target/namespaced property). Empty for a prim
    /// (non-property) path.
    pub fn property_suffix(&self) -> &str {
        &self.as_str()[self.prim_path().as_str().len()..]
    }

    /// Splits a property path into its owning prim and the property name (the
    /// portion after the `.`), or `None` if this is not a property path. The
    /// inverse of [`append_property`](Self::append_property).
    ///
    /// The property name is returned verbatim and may carry namespaces (`:`)
    /// or further target/connection syntax (`[..]`, `.`); callers that require
    /// a plain property name should validate it.
    ///
    /// ```text
    /// "/World/Mesh.points"   -> Some(("/World/Mesh", "points"))
    /// "/Mat.inputs:diffuse"  -> Some(("/Mat", "inputs:diffuse"))
    /// "/World/Mesh"          -> None
    /// ```
    pub fn split_property(&self) -> Option<(Path, &str)> {
        if !self.is_property() {
            return None;
        }
        let prim = self.prim_path();
        let name = self.as_str()[prim.as_str().len()..].strip_prefix('.')?;
        Some((prim, name))
    }

    /// Returns the relationship/connection target path embedded in this path's
    /// first `[..]` bracket, or `None` when there is none.
    ///
    /// ```text
    /// "/A.rel[/T].attr" -> Some("/T")
    /// "/A.attr[/T.x]"   -> Some("/T.x")
    /// "/A/B"            -> None
    /// ```
    ///
    /// The closing `]` is the one balanced with the first `[`, so a target that
    /// itself carries relational-target syntax (`/A.rel[/B.rel2[/C]]`) is
    /// extracted whole. A later, separate bracket pair (e.g. a connection
    /// `.mapper[..]`) is left untouched.
    pub fn embedded_target(&self) -> Option<Path> {
        let (open, close) = self.target_brackets()?;
        Some(Path::from_str_unchecked(&self.as_str()[open + 1..close]))
    }

    /// Returns this path with the target path in its first `[..]` bracket
    /// replaced by `new_target`, or `None` when there is no bracket. The
    /// inverse of [`embedded_target`](Self::embedded_target).
    pub fn replace_embedded_target(&self, new_target: &Path) -> Option<Path> {
        let (open, close) = self.target_brackets()?;
        let mut replaced = String::with_capacity(self.as_str().len() + new_target.as_str().len());
        replaced.push_str(&self.as_str()[..=open]);
        replaced.push_str(new_target.as_str());
        replaced.push_str(&self.as_str()[close..]);
        Some(Path::from_str_unchecked(&replaced))
    }

    /// Byte positions of the first `[` and the `]` balanced with it, or `None`
    /// when there is no bracket or it is unclosed. Nested brackets are tracked by
    /// depth so a target carrying its own `[..]` is matched whole.
    fn target_brackets(&self) -> Option<(usize, usize)> {
        let open = self.as_str().find('[')?;
        let mut depth = 0u32;
        for (i, byte) in self.as_str().bytes().enumerate().skip(open) {
            match byte {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((open, i));
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Returns the final namespace element of this path and its kind, or `None`
    /// for the pseudo-root `/` and empty paths. The inverse of the
    /// `append_property` / `append_variant_selection` / child-`append_path`
    /// family: an element appended to [`parent`](Self::parent) reconstructs the
    /// path.
    ///
    /// ```text
    /// "/A/B"          -> Some(Prim("B"))
    /// "/A.points"     -> Some(Property("points"))
    /// "/A{set=sel}"   -> Some(Variant { set: "set", selection: "sel" })
    /// ```
    pub fn last_element(&self) -> Option<PathElement<'_>> {
        if self.as_str().is_empty() || self.as_str() == "/" {
            return None;
        }
        // A property's element is the whole property name (everything after
        // the owning prim's `.`), matching `parent()`/`split_property()` so an
        // element appended to `parent()` reconstructs the path — including
        // names that themselves contain dots, e.g. `append_property("foo.bar")`.
        if let Some((_, name)) = self.split_property() {
            return Some(PathElement::Property(name));
        }
        // For a prim or variant path the last element is the last component;
        // reuse the single variant-grammar definition in `components`. A
        // non-empty remainder means the path has an unparsed tail (malformed,
        // or syntax this method doesn't model), so there is no clean final
        // element.
        let mut components = self.components();
        let last = components.by_ref().last();
        if !components.remainder().is_empty() {
            return None;
        }
        last.map(|c| match c {
            PathComponent::Prim(name) => PathElement::Prim(name),
            PathComponent::Variant { set, selection } => PathElement::Variant { set, selection },
        })
    }

    /// Returns the parent path, or `None` for the pseudo-root `/` and empty
    /// paths. Mirrors C++ `SdfPath::GetParentPath`: a property's parent is its
    /// owning prim, a variant selection's parent is the prim (or enclosing
    /// variant) it qualifies, and a prim's parent is its namespace parent.
    ///
    /// ```text
    /// "/A/B/C"      -> Some("/A/B")
    /// "/A"          -> Some("/")
    /// "/A.attr"     -> Some("/A")
    /// "/A{x=y}"     -> Some("/A")
    /// "/A{x=y}{p=q}"-> Some("/A{x=y}")
    /// "/"           -> None
    /// ""            -> None
    /// ```
    #[inline]
    pub fn parent(&self) -> Option<Path> {
        self.node.parent.clone()
    }

    /// Iterates `self` followed by each strict ancestor, ending with the
    /// absolute root (C++ `SdfPath::GetAncestorsRange`).
    ///
    /// Each step is one [`parent`](Self::parent) hop, so the walk crosses
    /// property, variant-selection, and prim boundaries exactly as `parent`
    /// does. The pseudo-root and the empty path yield only themselves.
    ///
    /// ```text
    /// "/A/B/C"  -> "/A/B/C", "/A/B", "/A", "/"
    /// "/A.attr" -> "/A.attr", "/A", "/"
    /// "/"       -> "/"
    /// ```
    pub fn ancestors(&self) -> impl Iterator<Item = Path> + use<> {
        std::iter::successors(Some(self.clone()), Path::parent)
    }

    /// Iterates the strict ancestors of `self` — [`ancestors`](Self::ancestors)
    /// without `self` itself — ending with the absolute root.
    pub fn strict_ancestors(&self) -> impl Iterator<Item = Path> + use<> {
        self.ancestors().skip(1)
    }

    /// Iterates `self` and its ancestors from leaf upward, stopping before the
    /// absolute root — [`ancestors`](Self::ancestors) with the pseudo-root
    /// trimmed. Empty when `self` is the absolute root.
    pub fn ancestors_below_root(&self) -> impl Iterator<Item = Path> + use<> {
        self.ancestors().take_while(|p| !p.is_abs_root())
    }

    /// Iterates the strict ancestors of `self` from leaf upward, stopping
    /// before the absolute root — [`ancestors_below_root`](Self::ancestors_below_root)
    /// without `self` itself. Empty for a root prim and for the absolute root.
    pub fn strict_ancestors_below_root(&self) -> impl Iterator<Item = Path> + use<> {
        self.strict_ancestors().take_while(|p| !p.is_abs_root())
    }

    /// Returns the name of this path's root prim — the first prim component,
    /// whatever the depth — or `None` for the pseudo-root, the empty path, and
    /// a relative path that opens with an anchor rather than a name.
    ///
    /// ```text
    /// "/A/B/C" -> Some("A")
    /// "/A{x=y}B" -> Some("A")
    /// "/"      -> None
    /// ```
    pub fn root_prim_name(&self) -> Option<&str> {
        match self.components().next() {
            Some(PathComponent::Prim(name)) => Some(name),
            _ => None,
        }
    }

    /// Returns the final component name, or `None` for the pseudo-root and empty paths.
    ///
    /// ```text
    /// "/A/B/C" -> Some("C")
    /// "/A"     -> Some("A")
    /// "/"      -> None
    /// ""       -> None
    /// ```
    pub fn name(&self) -> Option<&str> {
        if self.as_str().is_empty() || self.as_str() == "/" {
            return None;
        }
        // The final prim name begins after the rightmost `/` or `}` (a child of
        // a variant selection attaches directly, `/A{v=s}B`). A path that *ends*
        // in a variant selection (`/A{v=s}`) has no prim after the `}`, so its
        // final component is the prim-plus-selection segment after the last `/`.
        let start = self.as_str().rfind(['/', '}']).map_or(0, |i| i + 1);
        if start < self.as_str().len() {
            Some(&self.as_str()[start..])
        } else {
            Some(self.as_str().rsplit_once('/').map_or(self.as_str(), |(_, name)| name))
        }
    }

    /// Iterates the prim-namespace components of this path — prim names and
    /// `{set=sel}` variant selections — in root → leaf order.
    ///
    /// The iterator is lenient and does no validation: it yields the raw
    /// slices between delimiters and stops at the first thing it cannot parse
    /// as a prim/variant component (a property suffix, a malformed variant, or
    /// a stray separator), leaving that tail in
    /// [`remainder`](PathComponents::remainder). It is the single definition of
    /// the prim-path grammar; validating consumers layer their checks on top.
    ///
    /// Unlike [`strip_variant_selections`](Self::strip_variant_selections),
    /// which is a blunt string strip that also removes variant segments
    /// embedded inside relationship-target brackets, this models only the
    /// structured prim namespace.
    ///
    /// ```text
    /// "/A{x=y}B" -> [Prim("A"), Variant{x, y}, Prim("B")]
    /// "/A.attr"  -> [Prim("A")]  (remainder ".attr")
    /// ```
    pub fn components(&self) -> PathComponents<'_> {
        PathComponents::over(self.as_str().strip_prefix('/').unwrap_or(self.as_str()))
    }

    /// Counts the prim-name components of this path, skipping variant
    /// selections (C++ `PcpNode_GetNonVariantPathElementCount`).
    ///
    /// This is the namespace depth used to compare composition-node strength:
    /// a `{set=sel}` segment is part of the same prim, so it does not deepen
    /// the count.
    ///
    /// ```text
    /// "/A{x=y}B" -> 2
    /// "/A/B"     -> 2
    /// "/"        -> 0
    /// ```
    pub fn prim_element_count(&self) -> usize {
        self.components()
            .filter(|c| matches!(c, PathComponent::Prim(_)))
            .count()
    }

    /// Counts every prim-namespace component of this path, including `{set=sel}`
    /// variant selections (C++ `SdfPath::GetPathElementCount`).
    ///
    /// Unlike [`prim_element_count`](Self::prim_element_count), which counts only
    /// prim names, a variant selection counts as its own element here, so
    /// `/A{x=y}B` is 3.
    pub fn element_count(&self) -> usize {
        self.components().count()
    }

    /// Returns this path with every `{set=sel}` variant segment removed from
    /// its own prim chain.
    ///
    /// Equivalent to C++ `SdfPath::StripAllVariantSelections`. Both trailing
    /// and interior variant segments are stripped, so
    /// `/A{x=y}B{p=q}C` becomes `/A/B/C`. A selection inside a bracketed
    /// target component (`/A.rel[/T{v=x}]`) belongs to the embedded target
    /// path and is left in place, as in C++, where target components are data
    /// on a path node rather than part of the outer prim chain.
    pub fn strip_variant_selections(&self) -> Path {
        if !self.has_variant_selection() {
            return self.clone();
        }
        let mut out = String::with_capacity(self.as_str().len());
        let mut rest = self.as_str();
        let mut depth = 0usize;
        let mut chars = rest.char_indices();
        // Splice out each depth-0 `{…}` span; bracketed spans pass through.
        loop {
            let Some((at, c)) = chars.next() else {
                out.push_str(rest);
                break;
            };
            match c {
                '[' => depth += 1,
                ']' => depth = depth.saturating_sub(1),
                '{' if depth == 0 => {
                    out.push_str(&rest[..at]);
                    let Some(close) = rest[at..].find('}') else {
                        out.push_str(&rest[at..]);
                        break;
                    };
                    rest = &rest[at + close + 1..];
                    // In canonical form a prim child attaches directly to the
                    // variant (`/A{v=s}B`); reintroduce the `/` separator when
                    // the stripped variant is followed by a prim name. A `{`
                    // (next variant), `.` (property), or end needs no separator.
                    if rest.starts_with(is_identifier_cont) {
                        out.push('/');
                    }
                    chars = rest.char_indices();
                }
                _ => {}
            }
        }
        Path::from_str_unchecked(&out)
    }

    /// Returns `true` if this path starts with `prefix` at a path boundary.
    ///
    /// A match requires either equality with `prefix` or that the suffix
    /// following `prefix` begins with a path separator (`/`), property
    /// separator (`.`), or variant segment opener (`{`). This avoids false positives like
    /// `/Foobar` starting with `/Foo`.
    ///
    /// ```text
    /// "/A/B".has_prefix("/A")       -> true
    /// "/A".has_prefix("/A")         -> true
    /// "/A{set=sel}".has_prefix("/A")-> true
    /// "/A.attr".has_prefix("/A")    -> true
    /// "/Ab".has_prefix("/A")        -> false
    /// "/X".has_prefix("/")          -> true
    /// ```
    pub fn has_prefix(&self, prefix: &Path) -> bool {
        if self == prefix {
            return true;
        }
        let old = prefix.as_str();
        let me = self.as_str();
        let Some(suffix) = me.strip_prefix(old) else {
            return false;
        };
        old == "/"
            || suffix.starts_with('/')
            || suffix.starts_with('.')
            || suffix.starts_with('{')
            // A prim child attaches directly after a variant selection
            // (`/A{v=s}B`), so a `}`-terminated prefix is a boundary too.
            || prefix.is_variant_selection()
    }

    /// Whether this path and `other` are nested — one is a prefix of the other,
    /// or they are equal. The symmetric form of [`has_prefix`](Self::has_prefix);
    /// used to detect relocates and composition arcs whose endpoints would
    /// contain, or be contained by, one another.
    pub fn is_nested_with(&self, other: &Path) -> bool {
        self.has_prefix(other) || other.has_prefix(self)
    }

    /// Replaces a prefix path with a new prefix, used for namespace remapping
    /// during composition (e.g. references and inherits).
    ///
    /// Returns `None` if `self` does not start with `old_prefix`.
    ///
    /// ```text
    /// "/Ref/Child".replace_prefix("/Ref", "/MyPrim") -> Some("/MyPrim/Child")
    /// "/Ref".replace_prefix("/Ref", "/MyPrim")       -> Some("/MyPrim")
    /// "/Other".replace_prefix("/Ref", "/MyPrim")     -> None
    /// ```
    pub fn replace_prefix(&self, old_prefix: &Path, new_prefix: &Path) -> Option<Path> {
        if self == old_prefix {
            return Some(new_prefix.clone());
        }
        let old = old_prefix.as_str();
        let me = self.as_str();

        // Must start with old_prefix followed by '/', '.', or '{'. Property
        // targets in connection/relationship list ops rely on prim-prefix
        // mappings crossing the property separator, e.g.
        // `/Asset.outputs:out` -> `/Instance.outputs:out`.
        let suffix = me.strip_prefix(old)?;
        // The absolute root "/" is a prefix of all absolute paths; after
        // stripping it the remainder won't start with '/' (e.g. "Foo/Bar"). A
        // prim child attaches directly after a variant selection (`/A{v=s}B`),
        // so a `}`-terminated prefix is a boundary too.
        if old != "/"
            && !suffix.starts_with('/')
            && !suffix.starts_with('.')
            && !suffix.starts_with('{')
            && !old_prefix.is_variant_selection()
        {
            return None;
        }
        // Ensure a separator between new prefix and suffix for non-root.
        if old == "/" && !suffix.is_empty() {
            let new = new_prefix.as_str();
            if new == "/" {
                return Some(Path::from_str_unchecked(&format!("/{suffix}")));
            }
            return Some(Path::from_str_unchecked(&format!("{new}/{suffix}")));
        }

        let new = new_prefix.as_str();
        // Separate the child body from its old-namespace separator. A prim child
        // carries a leading `/` (child of a non-variant prim) or attaches
        // directly to a variant selection (a bare name, `/A{v=s}B`); a property
        // `.` or nested variant `{` attaches directly in any namespace.
        let (body, is_prim_child) = if let Some(rest) = suffix.strip_prefix('/') {
            (rest, true)
        } else if old_prefix.is_variant_selection() && suffix.starts_with(is_identifier_cont) {
            (suffix, true)
        } else {
            (suffix, false)
        };
        // Re-attach with the separator the new namespace needs: a prim child gets
        // a `/`, unless the new prefix is the root (the `/` is already there) or a
        // variant selection (the child attaches directly, `/Prim{set=sel}child`).
        let separator = if is_prim_child && new != "/" && !new_prefix.is_variant_selection() {
            "/"
        } else {
            ""
        };
        Some(Path::from_str_unchecked(&format!("{new}{separator}{body}")))
    }

    /// Appends a variant selection to a prim path, producing a path like
    /// `/MyPrim{variantSet=selection}`.
    ///
    /// The set name must be non-empty and both names must use the variant
    /// alphabet ([`is_valid_variant_identifier`](Self::is_valid_variant_identifier),
    /// where an empty selection names the bare variant set); the base must be
    /// a prim or variant-selection path.
    pub fn append_variant_selection(
        &self,
        set: impl AsRef<str>,
        selection: impl AsRef<str>,
    ) -> Result<Path, PathParseError> {
        let set = set.as_ref();
        let selection = selection.as_ref();
        let fail = |input: &str, reason| PathParseError {
            input: input.to_string(),
            offset: 0,
            reason,
        };
        if set.is_empty() || !Path::is_valid_variant_identifier(set) {
            return Err(fail(set, "invalid variant set name"));
        }
        if !Path::is_valid_variant_identifier(selection) {
            return Err(fail(selection, "invalid variant selection name"));
        }
        // Only a prim or variant-selection path can carry a `{set=sel}`
        // segment; the pseudo-root, properties, targets, and the `.`/`..`
        // relative anchors cannot.
        if self.is_empty()
            || self.is_abs_root()
            || self.is_property()
            || self.as_str() == "."
            || self.as_str().ends_with("..")
            || self.as_str().ends_with(']')
        {
            return Err(fail(self.as_str(), "path cannot carry a variant selection"));
        }
        Ok(Path::from_str_unchecked(&format!("{self}{{{set}={selection}}}")))
    }

    /// Appends a raw variant segment (e.g. `{set=sel}`) directly to this path.
    ///
    /// Unlike [`Self::append_path`], no `/` separator is inserted — variant
    /// segments attach directly to the prim path to produce canonical forms
    /// like `/Prim{set=sel}`. The segment text is external input (a usdc
    /// path-table element token), so it is validated.
    pub(crate) fn append_variant_segment(&self, segment: &str) -> Result<Path, PathParseError> {
        let fail = |(offset, reason)| PathParseError {
            input: segment.to_string(),
            offset,
            reason,
        };
        let inner = segment
            .strip_prefix('{')
            .and_then(|rest| rest.strip_suffix('}'))
            .ok_or_else(|| fail((0, "variant segment must be `{set=selection}`")))?;
        let (set, selection) = inner
            .split_once('=')
            .ok_or_else(|| fail((offset_in(segment, inner), "expected `=` in variant segment")))?;
        if set.is_empty() {
            return Err(fail((offset_in(segment, set), "empty variant set name")));
        }
        validate_variant_name(segment, set).map_err(fail)?;
        validate_variant_name(segment, selection).map_err(fail)?;
        Ok(Path::from_str_unchecked(&format!("{self}{segment}")))
    }

    /// Resolve a relative path against this path as anchor.
    ///
    /// Absolute paths are returned as-is. Relative segments (`..`) walk up
    /// from the anchor's prim path.
    ///
    /// Equivalent to C++ `SdfPath::MakeAbsolutePath`. When the combination
    /// is not a representable path (e.g. a property-relative step anchored at
    /// the pseudo-root, `"/"` + `".y"`), the empty path is returned —
    /// C++'s empty-path failure mode.
    ///
    /// ```text
    /// "/A/B".make_absolute("../C")   -> "/A/C"
    /// "/A/B".make_absolute("C/D")    -> "/A/B/C/D"
    /// "/A".make_absolute("/X")       -> "/X"
    /// ```
    pub fn make_absolute(&self, target: &Path) -> Path {
        let s = target.as_str();
        if s.starts_with('/') {
            return target.clone();
        }

        // Walk up from the anchor prim for each leading `..`, using `parent` so
        // the variant-selection boundary is respected (`/A{v=s}B` → `/A{v=s}`),
        // unlike a plain `/` split.
        let mut anchor = self.prim_path();
        let mut rest = s;
        while let Some(tail) = rest.strip_prefix("..") {
            anchor = anchor.parent().unwrap_or_else(Path::abs_root);
            rest = tail.strip_prefix('/').unwrap_or(tail);
        }

        if rest.is_empty() {
            return anchor;
        }
        // A property/relational tail (`.attr`, `.outputs:surface`) attaches to
        // the prim directly via its own `.` — no separator. A prim child
        // attaches to a variant selection directly (`/A{v=s}child`) but is
        // otherwise separated by `/`; the root already carries its slash.
        let anchor = anchor.as_str();
        let sep = if rest.starts_with('.') || anchor == "/" || anchor.ends_with('}') {
            ""
        } else {
            "/"
        };
        let combined = format!("{anchor}{sep}{rest}");
        if Path::validate(&combined).is_err() {
            return Path::default();
        }
        Path::from_str_unchecked(&combined)
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.node.text
    }

    /// Whether `name` is a valid prim or property identifier: non-empty,
    /// opening with a letter or `_` and continuing with letters, digits, and
    /// `_`. ASCII rules are strict; non-ASCII characters are accepted
    /// whenever they are neither whitespace nor control characters, a
    /// superset of the UTF-8 identifiers C++ OpenUSD allows.
    pub fn is_valid_identifier(name: &str) -> bool {
        let mut chars = name.chars();
        chars.next().is_some_and(is_identifier_start) && chars.all(is_identifier_cont)
    }

    /// Whether every `:`- or `.`-separated segment of `name` is a valid
    /// identifier — the shape of a namespaced property name.
    pub fn is_valid_namespace_identifier(name: &str) -> bool {
        name.split([':', '.']).all(Self::is_valid_identifier)
    }

    /// Whether `name` may serve as a variant set or selection name:
    /// identifier characters plus `|`, `-`, and `.`, in any position. The
    /// empty string is allowed — a variant *selection* may be empty
    /// (`{set=}`); a variant set name must additionally be non-empty.
    pub fn is_valid_variant_identifier(name: &str) -> bool {
        name.chars().all(is_variant_char)
    }

    /// Checks `s` against the path grammar; `Ok` iff [`Path::new`] would
    /// accept it. The grammar covers absolute and relative prim paths,
    /// `{set=selection}` variant segments, leading `..` steps, property
    /// tails with `:` namespaces and dotted chains, and bracketed `[…]`
    /// target paths (validated recursively).
    fn validate(s: &str) -> Result<(), PathParseError> {
        validate_path(s, s, 0).map_err(|(offset, reason)| PathParseError {
            input: s.to_string(),
            offset,
            reason,
        })
    }
}

/// A prim-namespace component of a [`Path`], yielded by [`Path::components`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathComponent<'a> {
    /// A prim name (registered under its parent's `primChildren`).
    Prim(&'a str),
    /// A `{set=sel}` variant selection on the current prim.
    Variant {
        /// The variant set name (between `{` and `=`).
        set: &'a str,
        /// The selected variant (between `=` and `}`); may be empty for a
        /// variant-set path like `/Prim{set=}`.
        selection: &'a str,
    },
}

/// The final namespace element of a [`Path`], returned by
/// [`Path::last_element`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathElement<'a> {
    /// A prim name.
    Prim(&'a str),
    /// A property name (the segment after the final `.`).
    Property(&'a str),
    /// A `{set=sel}` variant selection.
    Variant {
        /// The variant set name.
        set: &'a str,
        /// The selected variant.
        selection: &'a str,
    },
}

/// Iterator over the prim-namespace components of a [`Path`]. See
/// [`Path::components`].
pub struct PathComponents<'a> {
    /// Unparsed remainder of the path string.
    rest: &'a str,
    /// `true` once a prim name has been yielded, so the *next* prim child is
    /// introduced by a `/` separator. Reset to `false` after a variant
    /// selection, whose following prim child attaches directly (`/A{v=s}B`).
    expect_slash: bool,
    /// `true` once any component has been yielded. A variant selection must
    /// attach to a prim, so a leading `{` (e.g. `/{x=y}`) is malformed.
    emitted: bool,
}

impl<'a> PathComponents<'a> {
    /// Iterates the prim chain of `rest`, the path text with its leading `/`
    /// already stripped. A relative path's `..` run is not part of the chain
    /// grammar, so a caller that must parse past one strips it too.
    fn over(rest: &'a str) -> Self {
        Self {
            rest,
            expect_slash: false,
            emitted: false,
        }
    }

    /// The unparsed tail left after iteration: empty for a well-formed prim
    /// path, otherwise the property suffix or the malformed segment at which
    /// parsing stopped.
    pub fn remainder(&self) -> &'a str {
        self.rest
    }
}

impl<'a> Iterator for PathComponents<'a> {
    type Item = PathComponent<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }

        // A variant selection attaches directly to the preceding prim or
        // variant, with no separator. Its following prim child also attaches
        // directly, so clear `expect_slash`. A leading variant (no prim to
        // attach to, e.g. `/{x=y}`) is malformed and left in the remainder.
        if let Some(after_open) = self.rest.strip_prefix('{') {
            if !self.emitted {
                return None;
            }
            let Some(close) = after_open.find('}') else {
                return None; // unterminated; left in `rest`
            };
            let Some((set, selection)) = after_open[..close].split_once('=') else {
                return None; // missing `=`; left in `rest`
            };
            self.rest = &after_open[close + 1..];
            self.expect_slash = false;
            return Some(PathComponent::Variant { set, selection });
        }

        // A prim child of another prim is introduced by a single `/`. A prim
        // child of a variant attaches directly (no separator, `expect_slash`
        // already cleared). Anything else (a property `.`, trailing or doubled
        // `/`, junk) ends the prim namespace.
        if self.expect_slash {
            match self.rest.strip_prefix('/') {
                Some(next) if !next.is_empty() && !next.starts_with('/') && !next.starts_with('.') => {
                    self.rest = next;
                }
                _ => return None,
            }
        }

        let end = self.rest.find(['/', '{', '.']).unwrap_or(self.rest.len());
        let name = &self.rest[..end];
        if name.is_empty() {
            return None;
        }
        self.rest = &self.rest[end..];
        self.expect_slash = true;
        self.emitted = true;
        Some(PathComponent::Prim(name))
    }
}

/// The hash a [`Node`] spelled `text` carries.
fn hash_text(text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Locks the shard of [`NODES`] that the node of `hash` lives in.
fn lock_shard(hash: u64) -> MutexGuard<'static, Shard> {
    // A shard's map indexes buckets by the low bits of a hash and tags them
    // by the high ones. The shard is picked by bits from the middle.
    NODES[(hash >> 32) as usize % SHARDS]
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Whether path `text` ends in a property name.
fn ends_in_property(text: &str) -> bool {
    let Some(pos) = text.rfind('.') else {
        return false;
    };
    // The final dot must be followed by a valid property name: identifier
    // characters plus `:` for namespaced properties like
    // `primvars:displayColor` — the same alphabet the path validator
    // accepts. A tail carrying structural characters (`}`, `]`) belongs to
    // a variant selection or target path, not a property.
    let tail = &text[pos + 1..];
    !tail.is_empty() && tail.chars().all(|c| c == ':' || is_identifier_cont(c))
}

/// Length of the prefix of path `text` that spells [`Path::prim_path`].
fn prim_len(text: &str) -> usize {
    // Split at last slash.
    // "/A/B/C.foo[target].bar:baz" will become "/A/B" and "C.foo[target].bar:baz"
    let Some((before, after)) = text.rsplit_once('/') else {
        // Relative single segment (no slash): strip a trailing property or
        // variant selection. `ends_in_property` separates `Foo.bar` from a
        // relative `..`, which has no property tail and is its own prim.
        if ends_in_property(text)
            && let Some(dot) = find_component_dot(text)
        {
            return dot;
        }
        if text.ends_with('}')
            && let Some(open) = text.rfind('{')
        {
            return open;
        }
        return text.len();
    };

    // For cases like ../.foo[target].bar, just return ..
    if after.starts_with('.') {
        return before.len();
    }

    // Strip the trailing variant selection, keeping any earlier variants on
    // the same component: "/A/B/C{set=sel}" => "/A/B/C" and a nested
    // "/A{x=y}B{p=q}" => "/A{x=y}B" (the last `{` opens the trailing variant).
    if after.ends_with('}')
        && let Some(pos) = after.rfind('{')
    {
        return before.len() + pos + 1;
    }

    // The prim ends at the first dot. A path with none is a prim path.
    find_component_dot(after).map_or(text.len(), |dot| before.len() + dot + 1)
}

/// Length of the prefix of path `text` that spells [`Path::parent`], or
/// `None` when the path has no parent.
fn parent_len(text: &str) -> Option<usize> {
    if text.is_empty() || text == "/" {
        return None;
    }
    // Drop a trailing `{set=sel}` variant selection.
    if text.ends_with('}')
        && let Some(open) = text.rfind('{')
    {
        return Some(open);
    }
    // A relationship/connection target path (`/A.rel[/T]`): its parent is
    // the property owning the bracket, as in C++ `SdfPath::GetParentPath`.
    if text.ends_with(']') {
        let mut nesting = 0usize;
        for (i, byte) in text.bytes().enumerate().rev() {
            match byte {
                b']' => nesting += 1,
                b'[' => {
                    nesting -= 1;
                    if nesting == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        // No `[` balances the trailing `]`; the path is malformed.
        return None;
    }
    if ends_in_property(text) {
        // A malformed text whose property dot sits inside an unclosed `{` or
        // `[` is its own prim path. It gets no parent.
        return Some(prim_len(text)).filter(|&len| len < text.len());
    }
    // Prim path: drop the final prim name. It is introduced either by a `/`
    // (a child of another prim) or directly by a `}` (a child of a variant
    // selection, `/A{v=s}B`), whichever is rightmost.
    match text.rfind(['/', '}']) {
        // Child of a variant selection: keep up to and including the `}`.
        Some(i) if text.as_bytes()[i] == b'}' => Some(i + 1),
        // A root prim's parent is the pseudo-root.
        Some(0) => Some(1),
        // Child of another prim: drop the `/` and the name.
        Some(i) => Some(i),
        None => None,
    }
}

/// Maximum `[…]` target-path nesting depth [`Path::validate`] accepts,
/// guarding its recursion against pathological input.
const MAX_TARGET_NESTING: u8 = 64;

/// Whether `c` may open an identifier: `_`, an ASCII letter, or any
/// non-ASCII character that is neither whitespace nor control. ASCII is
/// strict; the non-ASCII rule is a superset of the UTF-8 identifiers C++
/// OpenUSD allows, so no asset it reads is rejected here.
fn is_identifier_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic() || (!c.is_ascii() && !c.is_whitespace() && !c.is_control())
}

/// Whether `c` may continue an identifier: an identifier-start character or
/// an ASCII digit.
fn is_identifier_cont(c: char) -> bool {
    c.is_ascii_digit() || is_identifier_start(c)
}

/// Whether `c` may appear in a variant set or selection name: identifier
/// characters plus `|`, `-`, and `.` (C++ `SdfSchemaBase`'s variant
/// identifier alphabet).
fn is_variant_char(c: char) -> bool {
    matches!(c, '|' | '-' | '.') || is_identifier_cont(c)
}

/// Byte position of the first `.` in `segment` outside `{…}` variant
/// selections and `[…]` target brackets — the dot introducing a property, as
/// opposed to one embedded in a dotted variant name or a bracketed target
/// path.
fn find_component_dot(segment: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, byte) in segment.bytes().enumerate() {
        match byte {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b'.' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Byte offset of `part` within `full`, of which it must be a subslice.
fn offset_in(full: &str, part: &str) -> usize {
    part.as_ptr() as usize - full.as_ptr() as usize
}

/// Validates one path — the whole string or a bracketed `[…]` target span
/// of it. Error offsets are relative to `full`.
fn validate_path(full: &str, s: &str, depth: u8) -> Result<(), (usize, &'static str)> {
    let base = offset_in(full, s);
    if depth > MAX_TARGET_NESTING {
        return Err((base, "target paths nested too deeply"));
    }
    if s.is_empty() {
        return Err((base, "empty path"));
    }
    if s == "/" || s == "." {
        return Ok(());
    }

    // Anchor: `/` for an absolute path, a `..(/..)*` run for a relative one.
    // What remains opens with a prim chain or a property tail.
    let abs = s.starts_with('/');
    let mut rest = s;
    if abs {
        rest = &s[1..];
    } else {
        while let Some(after) = rest.strip_prefix("..") {
            if after.is_empty() {
                return Ok(());
            }
            let Some(after_slash) = after.strip_prefix('/') else {
                return Err((offset_in(full, after), "expected `/` after `..`"));
            };
            if after_slash.is_empty() {
                return Err((offset_in(full, after_slash), "trailing `/`"));
            }
            rest = after_slash;
        }
    }

    // The prim chain, through the same iterator `Path::components` exposes;
    // per-character identifier checks layer on its structural grammar.
    let mut components = PathComponents::over(rest);
    for component in components.by_ref() {
        match component {
            PathComponent::Prim(name) => validate_identifier(full, name)?,
            PathComponent::Variant { set, selection } => {
                if set.is_empty() {
                    return Err((offset_in(full, set), "empty variant set name"));
                }
                validate_variant_name(full, set)?;
                validate_variant_name(full, selection)?;
            }
        }
    }

    let tail = components.remainder();
    if abs && !components.emitted {
        return Err((base + 1, "expected a prim name"));
    }
    if tail.is_empty() {
        return Ok(());
    }
    match tail.as_bytes()[0] {
        b'.' => validate_tail(full, tail, depth),
        b'/' => Err((offset_in(full, tail), "stray `/`")),
        b'{' => Err((offset_in(full, tail), "malformed or misplaced variant selection")),
        _ => Err((offset_in(full, tail), "expected `.`")),
    }
}

/// Validates a property tail — the `.name` / `[target]` items following the
/// prim chain. `tail` starts with `.`; a `[` recurses into
/// [`validate_path`] for the bracketed target.
fn validate_tail(full: &str, tail: &str, depth: u8) -> Result<(), (usize, &'static str)> {
    let mut rest = tail;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('.') {
            let end = after.find(['.', '[']).unwrap_or(after.len());
            let name = &after[..end];
            if name.is_empty() {
                return Err((offset_in(full, after), "empty property name"));
            }
            validate_property_name(full, name)?;
            rest = &after[end..];
        } else if let Some(after) = rest.strip_prefix('[') {
            let mut nesting = 1u32;
            let mut close = None;
            for (i, byte) in after.bytes().enumerate() {
                match byte {
                    b'[' => nesting += 1,
                    b']' => {
                        nesting -= 1;
                        if nesting == 0 {
                            close = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(close) = close else {
                return Err((offset_in(full, rest), "unterminated `[`"));
            };
            validate_path(full, &after[..close], depth + 1)?;
            rest = &after[close + 1..];
        } else {
            return Err((offset_in(full, rest), "expected `.` or `[`"));
        }
    }
    Ok(())
}

/// Validates one prim or property-namespace identifier segment.
fn validate_identifier(full: &str, name: &str) -> Result<(), (usize, &'static str)> {
    let mut chars = name.char_indices();
    match chars.next() {
        None => return Err((offset_in(full, name), "empty identifier")),
        Some((_, c)) if !is_identifier_start(c) => {
            return Err((offset_in(full, name), "invalid identifier start"));
        }
        _ => {}
    }
    for (i, c) in chars {
        if !is_identifier_cont(c) {
            return Err((offset_in(full, name) + i, "invalid character in identifier"));
        }
    }
    Ok(())
}

/// Validates a property name: `:`-separated identifier segments, none
/// empty.
fn validate_property_name(full: &str, name: &str) -> Result<(), (usize, &'static str)> {
    for segment in name.split(':') {
        if segment.is_empty() {
            return Err((offset_in(full, segment), "empty namespace segment in property name"));
        }
        validate_identifier(full, segment)?;
    }
    Ok(())
}

/// Validates a variant set or selection name (which may be empty).
fn validate_variant_name(full: &str, name: &str) -> Result<(), (usize, &'static str)> {
    for (i, c) in name.char_indices() {
        if !is_variant_char(c) {
            return Err((offset_in(full, name) + i, "invalid character in variant name"));
        }
    }
    Ok(())
}

impl From<&Path> for Path {
    fn from(p: &Path) -> Self {
        p.clone()
    }
}

impl TryFrom<&str> for Path {
    type Error = PathParseError;

    fn try_from(s: &str) -> Result<Path, PathParseError> {
        Path::new(s)
    }
}

impl TryFrom<String> for Path {
    type Error = PathParseError;

    fn try_from(value: String) -> Result<Path, PathParseError> {
        Path::new(&value)
    }
}

impl TryFrom<&String> for Path {
    type Error = PathParseError;

    fn try_from(value: &String) -> Result<Path, PathParseError> {
        Path::new(value)
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.text, f)
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        lock_shard(self.hash).remove(self);
    }
}

impl Shard {
    /// The live node spelled `text`, whose hash is `hash`.
    ///
    /// The live nodes of that hash it reads and finds spelled otherwise go
    /// to `others`, for the caller to drop once the shard is unlocked:
    /// dropping a node's last reference locks its shard.
    fn find(&self, hash: u64, text: &str, others: &mut Vec<Arc<Node>>) -> Option<Path> {
        let mut live = |weak: &Weak<Node>| {
            let node = weak.upgrade()?;
            if *node.text == *text {
                return Some(Path { node });
            }
            others.push(node);
            None
        };
        if let Some(path) = self.nodes.get(&hash).and_then(&mut live) {
            return Some(path);
        }
        self.collided.iter().find_map(live)
    }

    /// Records `node`, whose text no live node in the shard spells.
    fn insert(&mut self, node: &Arc<Node>) {
        let weak = Arc::downgrade(node);
        match self.nodes.entry(node.hash) {
            Entry::Occupied(slot) if slot.get().strong_count() > 0 => self.collided.push(weak),
            Entry::Occupied(mut slot) => {
                slot.insert(weak);
            }
            Entry::Vacant(slot) => {
                slot.insert(weak);
            }
        }
    }

    /// Forgets `node`, which is being dropped. Its entry may already be
    /// gone, overwritten by a node of the same hash interned since it died.
    fn remove(&mut self, node: &Node) {
        if self
            .nodes
            .get(&node.hash)
            .is_some_and(|weak| ptr::eq(weak.as_ptr(), node))
        {
            self.nodes.remove(&node.hash);
        } else {
            self.collided.retain(|weak| !ptr::eq(weak.as_ptr(), node));
        }
    }
}

impl Hasher for Prehashed {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("`Prehashed` hashes only `u64` keys");
    }

    fn write_u64(&mut self, hash: u64) {
        self.0 = hash;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::thread;

    use crate::Result;

    use super::*;

    /// The empty and absolute root paths share one node each, from every
    /// constructor and derivation that reaches them.
    #[test]
    fn sentinels_shared() {
        assert!(Arc::ptr_eq(&Path::default().node, &Path::default().node));
        assert!(Arc::ptr_eq(&Path::abs_root().node, &Path::abs_root().node));
        let root = Path::new("/Root").unwrap();
        assert!(Arc::ptr_eq(&root.parent().unwrap().node, &Path::abs_root().node));
        assert!(Arc::ptr_eq(
            &Path::new(".bar").unwrap().prim_path().node,
            &Path::default().node
        ));
    }

    /// Paths spelled the same share one node however they are made.
    #[test]
    fn interned_once() {
        let path = Path::new("/Interned/Branch").unwrap();
        assert!(Arc::ptr_eq(&path.node, &path.clone().node));
        let parsed = Path::try_from(String::from("/Interned/Branch")).unwrap();
        let derived = Path::new("/Interned").unwrap().append_path("Branch").unwrap();
        let stripped = Path::new("/Interned/Branch.size").unwrap().prim_path();
        for other in [&parsed, &derived, &stripped] {
            assert!(Arc::ptr_eq(&path.node, &other.node));
        }
        let mut paths = HashSet::new();
        paths.insert(path);
        assert!(paths.contains(&derived));
    }

    /// A node holds its parent's node, which the parent's own constructors
    /// then return.
    #[test]
    fn parent_linked() {
        let attr = Path::new("/Linked{v=s}Child.rel[/T].attr").unwrap();
        let chain: Vec<_> = attr.ancestors().collect();
        assert_eq!(
            chain.iter().map(Path::as_str).collect::<Vec<_>>(),
            [
                "/Linked{v=s}Child.rel[/T].attr",
                "/Linked{v=s}Child.rel[/T]",
                "/Linked{v=s}Child.rel",
                "/Linked{v=s}Child",
                "/Linked{v=s}",
                "/Linked",
                "/",
            ]
        );
        for path in &chain {
            assert!(Arc::ptr_eq(&path.node, &Path::new(path.as_str()).unwrap().node));
        }
        // The child keeps its parent alive. Dropping the child drops the
        // parent too.
        let child = Path::new("/Orphan/Child").unwrap();
        let parent = Arc::downgrade(&child.parent().unwrap().node);
        assert!(parent.upgrade().is_some());
        drop(child);
        assert!(parent.upgrade().is_none());
    }

    /// Whether the table holds an entry for the node spelled `text`.
    fn entry(text: &str) -> bool {
        let hash = hash_text(text);
        lock_shard(hash).nodes.contains_key(&hash)
    }

    /// A node's last drop removes its entry, and its parent's when that was
    /// the parent's last reference.
    #[test]
    fn dropped_node_removed() {
        let path = Path::new("/Removed/Child.attr").unwrap();
        assert!(entry("/Removed/Child.attr") && entry("/Removed/Child"));
        let parent = path.parent().unwrap();
        drop(path);
        assert!(!entry("/Removed/Child.attr"));
        assert!(entry("/Removed/Child"));
        drop(parent);
        assert!(!entry("/Removed/Child") && !entry("/Removed"));
        assert_eq!(
            Path::new("/Removed/Child.attr").unwrap().as_str(),
            "/Removed/Child.attr"
        );
    }

    /// Nodes whose hashes collide are told apart by their text, each found
    /// and removed on its own.
    #[test]
    fn collided_hashes() {
        // A hash neither text has, which no interned path reaches.
        let hash = 0x5eed_0000_c011_1de5;
        let node = |text: &str| {
            let node = Arc::new(Node {
                text: text.into(),
                hash,
                parent: None,
                property: false,
            });
            lock_shard(hash).insert(&node);
            node
        };
        let (first, second) = (node("/CollideA"), node("/CollideB"));
        let find = |text: &str| {
            let mut others = Vec::new();
            let found = lock_shard(hash).find(hash, text, &mut others);
            (found.map(|path| path.node), others.len())
        };
        let (found, others) = find("/CollideB");
        assert!(found.is_some_and(|found| Arc::ptr_eq(&found, &second)));
        assert_eq!(others, 1);
        assert!(find("/CollideA").0.is_some_and(|found| Arc::ptr_eq(&found, &first)));
        assert!(find("/CollideC").0.is_none());
        drop(first);
        assert!(find("/CollideA").0.is_none());
        assert!(find("/CollideB").0.is_some());
        drop(second);
        let shard = lock_shard(hash);
        assert!(!shard.nodes.contains_key(&hash));
        assert!(shard.collided.iter().all(|weak| weak.strong_count() > 0));
    }

    /// Threads interning and dropping the same paths over and over leave no
    /// entry behind.
    #[test]
    fn churn_across_threads() {
        let texts: Vec<_> = (0..64).map(|i| format!("/Churn/P{}/Q{i}.attr", i % 5)).collect();
        thread::scope(|threads| {
            for _ in 0..4 {
                threads.spawn(|| {
                    for _ in 0..200 {
                        let paths: Vec<_> = texts.iter().map(|text| Path::new(text).unwrap()).collect();
                        assert!(paths.iter().zip(&texts).all(|(path, text)| path.as_str() == text));
                    }
                });
            }
        });
        assert!(!texts.iter().any(|text| entry(text)) && !entry("/Churn"));
    }

    /// Threads interning the same paths at once agree on one node each.
    #[test]
    fn interned_across_threads() {
        let texts: Vec<_> = (0..256).map(|i| format!("/Threaded/P{}/Q{i}.attr", i % 7)).collect();
        let interned: Vec<Vec<Path>> = thread::scope(|threads| {
            let handles: Vec<_> = (0..4)
                .map(|_| threads.spawn(|| texts.iter().map(|text| Path::new(text).unwrap()).collect()))
                .collect();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect()
        });
        for paths in &interned[1..] {
            for (path, first) in paths.iter().zip(&interned[0]) {
                assert!(Arc::ptr_eq(&path.node, &first.node));
                assert!(Arc::ptr_eq(&path.parent().unwrap().node, &first.parent().unwrap().node));
            }
        }
    }

    /// Ordering compares text. Equal paths hash alike.
    #[test]
    fn order_by_text() {
        let mut paths: Vec<_> = ["/B", "/A/C", "/A", "/A.x", "/"]
            .into_iter()
            .map(|text| Path::new(text).unwrap())
            .collect();
        paths.sort();
        assert_eq!(
            paths.iter().map(Path::as_str).collect::<Vec<_>>(),
            ["/", "/A", "/A.x", "/A/C", "/B"]
        );
        let hash = |path: &Path| {
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&paths[1]), hash(&Path::new("/A").unwrap()));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_as_string() {
        for path in [
            Path::default(),
            Path::abs_root(),
            Path::new("/Root.rel[/Target]").unwrap(),
        ] {
            let encoded = serde_json::to_string(&path).unwrap();
            assert_eq!(encoded, serde_json::to_string(path.as_str()).unwrap());
            assert_eq!(serde_json::from_str::<Path>(&encoded).unwrap(), path);
        }
        assert!(serde_json::from_str::<Path>("\"/invalid path\"").is_err());
    }

    /// `is_prim` classifies by the path's final element, so every tail
    /// shape lands where C++ `SdfPath::IsPrimPath` puts it — including `.` and
    /// `..`, which C++ builds as prim nodes, and `/A.rel[/T]`, which the
    /// simpler `is_property` predicate cannot recognize as non-prim.
    #[test]
    fn is_prim_path_forms() {
        for text in ["/A", "/A/B", "/A{x=y}B", "../foo", "..", "../..", "."] {
            assert!(raw(text).is_prim(), "{text} names a prim");
        }
        for text in [
            "",
            "/",
            "/A{x=y}",
            "/A{x=}",
            "/A.attr",
            "/A.primvars:displayColor",
            "/A.rel[/T]",
            "/A.rel[/T].attr",
            "/A.attr.mapper[/T]",
            "/A.attr.expression",
        ] {
            assert!(!raw(text).is_prim(), "{text} does not name a prim");
        }
    }

    /// Builds a `Path` directly from `path`, skipping validation — for
    /// exercising lenient read-side behavior on malformed input.
    fn raw(path: &str) -> Path {
        Path::intern(path)
    }

    #[test]
    fn prim_element_count() {
        assert_eq!(Path::new("/A/B").unwrap().prim_element_count(), 2);
        assert_eq!(Path::new("/A{x=y}B").unwrap().prim_element_count(), 2);
        assert_eq!(Path::new("/A{x=y}").unwrap().prim_element_count(), 1);
        assert_eq!(Path::abs_root().prim_element_count(), 0);
    }

    #[test]
    fn embedded_target_get() {
        assert_eq!(
            Path::new("/A.rel[/T].attr").unwrap().embedded_target(),
            Some(Path::new("/T").unwrap())
        );
        assert_eq!(
            Path::new("/A.attr[/T.x]").unwrap().embedded_target(),
            Some(Path::new("/T.x").unwrap())
        );
        // A nested target bracket is matched whole, not truncated at the inner `]`.
        assert_eq!(
            Path::new("/A.rel[/B.rel2[/C]].attr").unwrap().embedded_target(),
            Some(Path::new("/B.rel2[/C]").unwrap())
        );
        // An unclosed bracket has no balanced close.
        assert_eq!(raw("/A.rel[/T").embedded_target(), None);
        assert_eq!(Path::new("/A/B").unwrap().embedded_target(), None);
    }

    #[test]
    fn embedded_target_replace() {
        let t = Path::new("/Source/Child").unwrap();
        assert_eq!(
            Path::new("/A.rel[/T].attr")
                .unwrap()
                .replace_embedded_target(&t)
                .unwrap()
                .as_str(),
            "/A.rel[/Source/Child].attr"
        );
        assert_eq!(
            Path::new("/A.attr[/T.x]")
                .unwrap()
                .replace_embedded_target(&t)
                .unwrap()
                .as_str(),
            "/A.attr[/Source/Child]"
        );
        assert!(Path::new("/A/B").unwrap().replace_embedded_target(&t).is_none());
    }

    #[test]
    fn test_append_property() {
        let base = Path::new("/foo").unwrap();

        assert_eq!(base.append_property("prop").unwrap().as_str(), "/foo.prop");
        assert_eq!(
            base.append_property("prop:foo:bar").unwrap().as_str(),
            "/foo.prop:foo:bar"
        );

        let base = Path::new("/foo.prop").unwrap();
        assert!(base.append_property("prop2").is_err());
        assert!(base.append_property("prop2:foo:bar").is_err());
    }

    #[test]
    fn test_append_path() -> Result<()> {
        assert_eq!(Path::new("/prim")?.append_path(".")?.as_str(), "/prim");

        assert_eq!(Path::new("/")?.append_path("foo/bar.attr")?.as_str(), "/foo/bar.attr");
        assert_eq!(
            Path::new("/")?.append_path("foo/bar.attr:argle:bargle")?.as_str(),
            "/foo/bar.attr:argle:bargle"
        );

        assert_eq!(Path::new("/foo")?.append_path("bar.attr")?.as_str(), "/foo/bar.attr");
        assert_eq!(
            Path::new("/foo")?.append_path("bar.attr:argle:bargle")?.as_str(),
            "/foo/bar.attr:argle:bargle"
        );
        assert_eq!(
            Path::new("/foo")?.append_path("bar.rel[/target].attr")?.as_str(),
            "/foo/bar.rel[/target].attr"
        );

        assert_eq!(
            Path::new("/foo")?
                .append_path("bar.rel[/target].attr:argle:bargle")?
                .as_str(),
            "/foo/bar.rel[/target].attr:argle:bargle"
        );

        assert_eq!(
            Path::new("/foo")?.append_path("bar.attr[/target.attr]")?.as_str(),
            "/foo/bar.attr[/target.attr]"
        );

        assert_eq!(
            Path::new("/foo")?
                .append_path("bar.attr[/target.attr:argle:bargle]")?
                .as_str(),
            "/foo/bar.attr[/target.attr:argle:bargle]"
        );

        assert_eq!(
            Path::new("/foo")?.append_path("bar.attr.mapper[/target].arg")?.as_str(),
            "/foo/bar.attr.mapper[/target].arg"
        );

        Ok(())
    }

    #[test]
    fn test_append_invalid_path() -> Result<()> {
        assert!(Path::new("/prim")?.append_path("/abs").is_err());
        assert!(Path::new("/prim.attr")?.append_path("abs").is_err());

        Ok(())
    }

    #[test]
    fn test_prim_path() {
        #[rustfmt::skip]
        let cases = [
            ("/A/B/C", "/A/B/C"),

            ("/A/B{set=sel}C", "/A/B{set=sel}C"),
            ("/A/B/C{set=sel}", "/A/B/C"),

            // A variant set on a variant-direct child: strip only the trailing
            // variant, keeping the earlier one on the same component.
            ("/A{x=y}B{p=q}", "/A{x=y}B"),

            // A property authored on a variant-direct child or on the variant.
            ("/A{x=y}B.attr", "/A{x=y}B"),
            ("/A{x=y}.attr", "/A{x=y}"),

            ("/A/B/C.foo", "/A/B/C"),
            ("/A/B/C.foo:bar:baz", "/A/B/C"),

            ("/A/B/C.foo[target].bar", "/A/B/C"),
            ("/A/B/C.foo[target].bar:baz", "/A/B/C"),

            ("A/B/C.foo[target].bar", "A/B/C"),
            ("A/B/C.foo[target].bar:baz", "A/B/C"),

            ("../C.foo", "../C"),
            ("../C.foo:bar:baz", "../C"),

            ("../.foo[target].bar", ".."),
            ("../.foo[target].bar:baz", ".."),

            // Relative single segment (no slash): strip the property/variant.
            ("Foo.bar", "Foo"),
            ("Foo{x=y}", "Foo"),
            ("Foo", "Foo"),
            (".bar", ""),
            ("..", ".."),
        ];

        for (path, expected) in cases {
            assert_eq!(
                Path::new(path).unwrap().prim_path().as_str(),
                expected,
                "Unable to parse: {path}",
            );
        }
    }

    #[test]
    fn test_is_property() {
        #[rustfmt::skip]
        let cases = [
            ("/Foo/Bar.baz", true),
            ("Foo", false),
            ("Foo/Bar", false),
            ("Foo.bar", true),
            ("Foo/Bar.bar", true),
            (".bar", true),
            ("/Some/Kinda/Long/Path/Just/To/Make/Sure", false),
            ("Some/Kinda/Long/Path/Just/To/Make/Sure.property", true),
            ("../Some/Kinda/Long/Path/Just/To/Make/Sure", false),
            ("../../Some/Kinda/Long/Path/Just/To/Make/Sure.property", true),
            ("/Foo/Bar.baz[targ].boom", true),
            ("Foo.bar[targ].boom", true),
            (".bar[targ].boom", true),
            ("Foo.bar[targ.attr].boom", true),
            ("/A/B/C.rel3[/Blah].attr3", true),
            ("A/B.rel2[/A/B/C.rel3[/Blah].attr3].attr2", true),
            ("/A.rel1[/A/B.rel2[/A/B/C.rel3[/Blah].attr3].attr2].attr1", true),
        ];

        for (path, expected) in cases {
            assert_eq!(Path::new(path).unwrap().is_property(), expected);
        }
    }

    #[test]
    fn test_strip_all_variant_selections() {
        let cases: &[(&str, &str)] = &[
            ("/A/B/C", "/A/B/C"),
            ("/A{set=sel}", "/A"),
            ("/A{set=sel}B", "/A/B"),
            ("/A{x=y}B{p=q}C", "/A/B/C"),
            ("/A/B{p=q}C.attr", "/A/B/C.attr"),
            ("/", "/"),
        ];
        for (input, expected) in cases {
            let p = Path::new(input).unwrap();
            assert_eq!(p.strip_variant_selections().as_str(), *expected, "input {input}");
        }
    }

    /// A selection inside a bracketed target component belongs to the embedded
    /// target path, so the strip removes only the outer prim chain's segments —
    /// on relative paths too.
    #[test]
    fn strip_skips_bracket_targets() {
        let cases: &[(&str, &str)] = &[
            ("/A.rel[/T{v=x}B]", "/A.rel[/T{v=x}B]"),
            ("/A{v=x}B.rel[/T{w=y}]", "/A/B.rel[/T{w=y}]"),
            ("/A{v=x}.rel[/T{w=y}].attr", "/A.rel[/T{w=y}].attr"),
            ("../A{v=x}", "../A"),
        ];
        for (input, expected) in cases {
            let p = raw(input);
            assert_eq!(p.strip_variant_selections().as_str(), *expected, "input {input}");
        }
    }

    #[test]
    fn has_variant_selection() {
        // A selection anywhere in the prim chain counts — on relative paths
        // too; a `{` that is not a prim-namespace variant (a
        // relationship-target bracket) does not.
        let cases: &[(&str, bool)] = &[
            ("/A/B", false),
            ("/A{set=sel}", true),
            ("/A{set=sel}/Child", true),
            ("/A{x=y}B{p=q}C", true),
            ("/A.rel[/B{x=y}]", false),
            ("../A{v=x}", true),
            ("..", false),
        ];
        for (input, expected) in cases {
            assert_eq!(raw(input).has_variant_selection(), *expected, "input {input}");
        }
    }

    #[test]
    fn test_last_element() {
        // Render the element as an owned string so it doesn't borrow the
        // temporary `Path`: `name`, `.name` for a property, `{set=sel}`.
        let last = |s: &str| -> Option<String> {
            raw(s).last_element().map(|e| match e {
                PathElement::Prim(name) => name.to_owned(),
                PathElement::Property(name) => format!(".{name}"),
                PathElement::Variant { set, selection } => format!("{{{set}={selection}}}"),
            })
        };

        assert_eq!(last("/A/B").as_deref(), Some("B"));
        assert_eq!(last("/A").as_deref(), Some("A"));
        assert_eq!(last("/A.points").as_deref(), Some(".points"));
        assert_eq!(last("/A.inputs:diffuse").as_deref(), Some(".inputs:diffuse"));
        // The whole property name, even one containing dots (so parent() plus
        // this element reconstruct the path) — `append_property("foo.bar")`.
        assert_eq!(last("/A.foo.bar").as_deref(), Some(".foo.bar"));
        assert_eq!(last("/A{set=sel}").as_deref(), Some("{set=sel}"));
        assert_eq!(last("/A{x=y}{p=q}").as_deref(), Some("{p=q}"));
        assert_eq!(last("/"), None);
        assert_eq!(last(""), None);
        // An unparsed tail (malformed, or target syntax this method doesn't
        // model) has no clean final element — None, not the last parsed prim.
        assert_eq!(last("/A{bad"), None);
        assert_eq!(last("/A.rel[/Target]"), None);
        assert_eq!(last("/{x=y}"), None);
    }

    #[test]
    fn test_split_property() {
        let split = |s: &str| {
            raw(s)
                .split_property()
                .map(|(p, n)| (p.as_str().to_owned(), n.to_owned()))
        };
        let owned = |p: &str, n: &str| Some((p.to_owned(), n.to_owned()));

        assert_eq!(split("/World/Mesh.points"), owned("/World/Mesh", "points"));
        assert_eq!(split("/Mat.inputs:diffuse"), owned("/Mat", "inputs:diffuse"));
        // Relative property (no slash) decomposes too — the inverse of
        // append_property.
        assert_eq!(split("Foo.bar"), owned("Foo", "bar"));
        // Not a property path.
        assert_eq!(split("/World/Mesh"), None);
        assert_eq!(split("/"), None);
    }

    #[test]
    fn test_property_suffix() {
        let suffix = |s: &str| raw(s).property_suffix().to_owned();

        assert_eq!(suffix("/A.points"), ".points");
        assert_eq!(suffix("/A/B.inputs:diffuse"), ".inputs:diffuse");
        // Relative property (no slash).
        assert_eq!(suffix("Foo.bar"), ".bar");
        // A prim path has no property suffix.
        assert_eq!(suffix("/A"), "");
    }

    #[test]
    fn test_components() {
        // Render each component as an owned string so the result doesn't borrow
        // the temporary `Path`: a prim is its name, a variant is `{set=sel}`.
        let parse = |s: &str| -> (Vec<String>, String) {
            let p = raw(s);
            let mut it = p.components();
            let items = it
                .by_ref()
                .map(|c| match c {
                    PathComponent::Prim(name) => name.to_owned(),
                    PathComponent::Variant { set, selection } => format!("{{{set}={selection}}}"),
                })
                .collect();
            (items, it.remainder().to_owned())
        };
        let case = |items: &[&str]| items.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();

        // Prim names and variant selections, in order, fully consumed. A prim
        // child attaches directly to a variant selection (canonical form).
        assert_eq!(parse("/A/B/C"), (case(&["A", "B", "C"]), String::new()));
        assert_eq!(parse("/A{x=y}B"), (case(&["A", "{x=y}", "B"]), String::new()));
        assert_eq!(parse("/A{x=y}B/C"), (case(&["A", "{x=y}", "B", "C"]), String::new()));
        assert_eq!(
            parse("/A{x=y}{p=q}C"),
            (case(&["A", "{x=y}", "{p=q}", "C"]), String::new())
        );
        assert_eq!(parse("/A{x=y}{p=q}"), (case(&["A", "{x=y}", "{p=q}"]), String::new()));
        // An empty selection (variant-set path) is yielded verbatim.
        assert_eq!(parse("/A{x=}"), (case(&["A", "{x=}"]), String::new()));

        // A property suffix ends the prim namespace and stays in the remainder.
        assert_eq!(parse("/A.attr"), (case(&["A"]), ".attr".to_owned()));
        assert_eq!(parse("/A{x=y}B.attr"), (case(&["A", "{x=y}", "B"]), ".attr".to_owned()));
        // The non-canonical slash-after-variant form is not consumed past the
        // variant: the stray `/B` is left in the remainder.
        assert_eq!(parse("/A{x=y}/B"), (case(&["A", "{x=y}"]), "/B".to_owned()));

        // Malformed input stops parsing, leaving the bad tail in the remainder.
        assert_eq!(parse("/A{x=y"), (case(&["A"]), "{x=y".to_owned()));
        assert_eq!(parse("/A{x}"), (case(&["A"]), "{x}".to_owned()));
        assert_eq!(parse("/A/"), (case(&["A"]), "/".to_owned()));
        assert_eq!(parse("/A//B"), (case(&["A"]), "//B".to_owned()));

        // The root and empty paths have no components.
        assert_eq!(parse("/"), (Vec::<String>::new(), String::new()));
    }

    #[test]
    fn test_has_prefix() {
        let cases: &[(&str, &str, bool)] = &[
            ("/A/B/C", "/A", true),
            ("/A/B/C", "/A/B", true),
            ("/A/B/C", "/A/B/C", true),
            ("/A/B/C", "/A/B/D", false),
            ("/Foobar", "/Foo", false),
            ("/A{set=sel}", "/A", true),
            // A child attaches directly to a variant selection (canonical form).
            ("/A{set=sel}B", "/A{set=sel}", true),
            ("/A{set=sel}B", "/A", true),
            ("/A.attr", "/A", true),
            ("/A", "/", true),
            ("/", "/", true),
            ("/A/B", "/A/B/C", false),
        ];
        for (path, prefix, expected) in cases {
            let p = Path::new(path).unwrap();
            let pre = Path::new(prefix).unwrap();
            assert_eq!(p.has_prefix(&pre), *expected, "{path}.has_prefix({prefix})");
        }
    }

    #[test]
    fn test_path_cmp() {
        // Less then
        assert!(Path::from_str("aaa").unwrap() < Path::from_str("aab").unwrap());
        assert!(Path::from_str("/").unwrap() < Path::from_str("/a").unwrap());

        // Greater then
        assert!(Path::from_str("aab").unwrap() > Path::from_str("aaa").unwrap());

        // Less equal
        assert!(Path::from_str("aaa").unwrap() <= Path::from_str("aab").unwrap());
        assert!(Path::from_str("aaa").unwrap() <= Path::from_str("aaa").unwrap());

        // Greater equal
        assert!(Path::from_str("aab").unwrap() >= Path::from_str("aaa").unwrap());
        assert!(Path::from_str("aaa").unwrap() >= Path::from_str("aaa").unwrap());
    }

    #[test]
    fn test_parent() {
        #[rustfmt::skip]
        let cases: &[(&str, Option<&str>)] = &[
            ("/A/B/C",       Some("/A/B")),
            ("/A/B",         Some("/A")),
            ("/A",           Some("/")),
            ("/",            None),
            ("",             None),
            // A property's parent is its owning prim.
            ("/A.attr",      Some("/A")),
            ("/A/B.attr",    Some("/A/B")),
            ("/A.foo:bar",   Some("/A")),
            // A variant selection's parent is the prim (or enclosing variant).
            ("/A{x=y}",      Some("/A")),
            ("/A/B{x=y}",    Some("/A/B")),
            ("/A{x=y}{p=q}",  Some("/A{x=y}")),
            // A child attaches directly to a variant selection (canonical form).
            ("/A{x=y}B",     Some("/A{x=y}")),
            ("/A{x=y}B/C",   Some("/A{x=y}B")),
            ("/A{x=y}.attr", Some("/A{x=y}")),
            // Relative paths must not self-parent (would loop a parent walk).
            ("Foo.bar",      Some("Foo")),
            ("Foo",          None),
            ("..",           None),
        ];

        for &(path, expected) in cases {
            let parent = raw(path).parent();
            let parent = parent.as_ref().map(|p| p.as_str());
            assert_eq!(parent, expected, "parent of {path:?}");
            // A parent must make progress — never return the path itself.
            assert_ne!(parent, Some(path), "self-parent on {path:?}");
        }
    }

    #[test]
    fn ancestors_walk() {
        let p = Path::new("/A/B/C.attr").unwrap();
        let strs = |it: &mut dyn Iterator<Item = Path>| it.map(|a| a.as_str().to_string()).collect::<Vec<_>>();
        assert_eq!(strs(&mut p.ancestors()), ["/A/B/C.attr", "/A/B/C", "/A/B", "/A", "/"]);
        // `strict_ancestors` drops self; `ancestors_below_root` drops the root.
        assert_eq!(strs(&mut p.strict_ancestors()), ["/A/B/C", "/A/B", "/A", "/"]);
        assert_eq!(
            strs(&mut p.ancestors_below_root()),
            ["/A/B/C.attr", "/A/B/C", "/A/B", "/A"]
        );
        assert_eq!(strs(&mut p.strict_ancestors_below_root()), ["/A/B/C", "/A/B", "/A"]);
        // The pseudo-root yields only itself, no strict ancestors, nothing below.
        assert_eq!(Path::abs_root().ancestors().count(), 1);
        assert_eq!(Path::abs_root().strict_ancestors().count(), 0);
        assert_eq!(Path::abs_root().ancestors_below_root().count(), 0);
        assert_eq!(Path::abs_root().strict_ancestors_below_root().count(), 0);
        assert_eq!(Path::new("/A").unwrap().strict_ancestors_below_root().count(), 0);
    }

    #[test]
    fn test_name() {
        #[rustfmt::skip]
        let cases: &[(&str, Option<&str>)] = &[
            ("/A/B/C", Some("C")),
            ("/A/B",   Some("B")),
            ("/A",     Some("A")),
            // A child attaches directly to a variant selection (canonical form).
            ("/A{x=y}B", Some("B")),
            ("/",      None),
            ("",       None),
            ("Foo",    Some("Foo")),
        ];

        for &(path, expected) in cases {
            assert_eq!(raw(path).name(), expected, "name of {path:?}",);
        }
    }

    #[test]
    fn test_replace_prefix() {
        let p = |s| Path::new(s).unwrap();

        // Exact match.
        assert_eq!(
            p("/Ref").replace_prefix(&p("/Ref"), &p("/MyPrim")).unwrap().as_str(),
            "/MyPrim"
        );

        // Child remapping.
        assert_eq!(
            p("/Ref/Child")
                .replace_prefix(&p("/Ref"), &p("/MyPrim"))
                .unwrap()
                .as_str(),
            "/MyPrim/Child"
        );

        // Deeper nesting.
        assert_eq!(
            p("/Ref/A/B").replace_prefix(&p("/Ref"), &p("/X")).unwrap().as_str(),
            "/X/A/B"
        );

        // Remap to root.
        assert_eq!(
            p("/Ref/Child").replace_prefix(&p("/Ref"), &p("/")).unwrap().as_str(),
            "/Child"
        );

        // Property paths still live under the owning prim namespace for
        // composition maps; the `.` separator must count as a prefix boundary.
        assert_eq!(
            p("/Ref.outputs:out")
                .replace_prefix(&p("/Ref"), &p("/MyPrim"))
                .unwrap()
                .as_str(),
            "/MyPrim.outputs:out"
        );

        // No match.
        assert!(p("/Other").replace_prefix(&p("/Ref"), &p("/MyPrim")).is_none());

        // Partial name overlap must not match (e.g. /RefExtra should not match /Ref).
        assert!(p("/RefExtra").replace_prefix(&p("/Ref"), &p("/X")).is_none());

        // A variant-direct child (`/A{v=s}B`) remapped off the variant regains a
        // `/` separator; remapped onto another variant it still attaches directly.
        assert_eq!(
            p("/A{v=s}B").replace_prefix(&p("/A{v=s}"), &p("/A")).unwrap().as_str(),
            "/A/B"
        );
        assert_eq!(
            p("/A{v=s}B")
                .replace_prefix(&p("/A{v=s}"), &p("/X{w=t}"))
                .unwrap()
                .as_str(),
            "/X{w=t}B"
        );
        // A deeper child under the variant-direct prim keeps its own `/`.
        assert_eq!(
            p("/A{v=s}B/C")
                .replace_prefix(&p("/A{v=s}"), &p("/A"))
                .unwrap()
                .as_str(),
            "/A/B/C"
        );
    }

    #[test]
    fn test_append_variant_selection() {
        let p = Path::new("/MyPrim").unwrap();
        assert_eq!(
            p.append_variant_selection("model", "high").unwrap().as_str(),
            "/MyPrim{model=high}"
        );
    }

    #[test]
    fn validate_identifier() {
        // Valid identifiers
        assert!(Path::is_valid_identifier("_"));
        assert!(Path::is_valid_identifier("x"));
        assert!(Path::is_valid_identifier("_1"));
        assert!(Path::is_valid_identifier("a1"));
        assert!(Path::is_valid_identifier("test"));
        assert!(Path::is_valid_identifier("_test"));
        assert!(Path::is_valid_identifier("test123"));
        assert!(Path::is_valid_identifier("Test"));
        assert!(Path::is_valid_identifier("teST"));
        assert!(Path::is_valid_identifier("TEST"));

        // Invalid ones
        assert!(!Path::is_valid_identifier(""));
        assert!(!Path::is_valid_identifier(" "));
        assert!(!Path::is_valid_identifier("?"));
        assert!(!Path::is_valid_identifier("1"));
        assert!(!Path::is_valid_identifier("x!"));
        assert!(!Path::is_valid_identifier("_abc?"));
        assert!(!Path::is_valid_identifier("_!"));
        assert!(!Path::is_valid_identifier("test "));
        assert!(!Path::is_valid_identifier(" test"));
        assert!(!Path::is_valid_identifier("te st"));
        assert!(!Path::is_valid_identifier("te.st"));
        assert!(!Path::is_valid_identifier("te:st"));
    }

    #[test]
    fn make_absolute() {
        let abs = |anchor, target| raw(anchor).make_absolute(&raw(target));

        assert_eq!(abs("/A/B", "/X/Y").as_str(), "/X/Y");
        assert_eq!(abs("/A/B", "../C").as_str(), "/A/C");
        assert_eq!(abs("/A/B/C", "../../D").as_str(), "/A/D");
        assert_eq!(abs("/A", "../X").as_str(), "/X");
        assert_eq!(abs("/A/B", "..").as_str(), "/A");
        assert_eq!(abs("/A/B", "C/D").as_str(), "/A/B/C/D");

        // The `..` walk respects the variant-selection boundary: the parent of
        // `/A{v=s}B` is `/A{v=s}`, and a child reattaches directly after `}`.
        assert_eq!(abs("/A{v=s}B", "../C").as_str(), "/A{v=s}C");
        assert_eq!(abs("/A{v=s}B", "C").as_str(), "/A{v=s}B/C");
        assert_eq!(abs("/A{v=s}B", "..").as_str(), "/A{v=s}");

        // A property-relative tail attaches to the prim via `.`, not `/`.
        assert_eq!(abs("/Shader", ".outputs:surface").as_str(), "/Shader.outputs:surface");
        assert_eq!(abs("/A/B", "../.attr").as_str(), "/A.attr");
    }

    #[test]
    fn append_variant_segment() {
        let p = raw;

        // Variant set and selection attach directly without a slash separator.
        assert_eq!(p("/A").append_variant_segment("{v=sel}").unwrap().as_str(), "/A{v=sel}");
        assert_eq!(
            p("/A/B").append_variant_segment("{color=red}").unwrap().as_str(),
            "/A/B{color=red}"
        );
        // Empty selection (variant set path).
        assert_eq!(p("/A").append_variant_segment("{v=}").unwrap().as_str(), "/A{v=}");
        // Nested variant segments stack.
        assert_eq!(
            p("/A{v=x}").append_variant_segment("{w=y}").unwrap().as_str(),
            "/A{v=x}{w=y}"
        );
    }

    #[test]
    fn variant_segment_rejects() {
        let p = raw("/A");
        assert!(p.append_variant_segment("{v=sel").is_err());
        assert!(p.append_variant_segment("v=sel}").is_err());
        assert!(p.append_variant_segment("{vsel}").is_err());
        assert!(p.append_variant_segment("{=x}").is_err());
        assert!(p.append_variant_segment("{v=s l}").is_err());
    }

    #[test]
    fn parse_accepts() {
        #[rustfmt::skip]
        let cases = [
            "/", ".", "..", "../..", "../C.foo", "Foo", "A/B/C",
            "/A.foo:bar:baz", "/A{x=y}", "/A{x=}", "/A{x=y}{p=q}",
            "/A{x=y}B", "/A{x=y}.attr", ".bar", "../.foo[target].bar",
            "/A.rel[/T].attr", "/A.rel1[/A/B.rel2[/C].attr2].attr1",
            "/foo/bar.attr.mapper[/target].arg", "/A.attr[/target.attr]",
            "/A.foo.bar", "/_underscore/_1", "/Ünïcode/日本",
        ];
        for case in cases {
            assert!(Path::new(case).is_ok(), "should parse: {case}");
        }
    }

    #[test]
    fn parse_rejects() {
        #[rustfmt::skip]
        let cases = [
            "", "/foo bar", "//A", "/A/", "../", "/A{x=y", "/A{x}",
            "/{x=y}", "/.foo", "./A", "..foo", "/A.", "/A.foo::bar",
            "/A.foo:", "/A.rel[/T", "/A[3]", "/A{x=y}/B", "/A{=y}",
            "/A.rel[]", "/A..b", "/1digit", "/A.foo/bar", "/A{x=y}}",
        ];
        for case in cases {
            assert!(Path::new(case).is_err(), "should reject: {case}");
        }
    }

    #[test]
    fn parse_error_details() {
        let err = Path::new("/World/foo bar").unwrap_err();
        assert_eq!(err.offset, 10);
        assert_eq!(err.reason, "invalid character in identifier");

        let err = Path::new("").unwrap_err();
        assert_eq!(err.reason, "empty path");

        // The offset points into the embedded target of a bracketed path.
        let err = Path::new("/A.rel[/T x].attr").unwrap_err();
        assert_eq!(err.offset, 9);
    }

    #[test]
    fn parse_depth_cap() {
        // Reasonable nesting parses; pathological nesting is refused rather
        // than recursing without bound.
        let nested = |n: usize| {
            let mut path = String::from("/B");
            for _ in 0..n {
                path = format!("/A.r[{path}]");
            }
            path
        };
        assert!(Path::new(&nested(8)).is_ok());
        assert!(Path::new(&nested(70)).is_err());
    }

    #[test]
    fn dotted_variant_names() {
        // A dot inside a variant name is not a property separator.
        let p = Path::new("/A{v=1.5}.attr").unwrap();
        assert_eq!(p.prim_path().as_str(), "/A{v=1.5}");
        assert_eq!(p.parent().unwrap().as_str(), "/A{v=1.5}");
        assert_eq!(Path::new("Foo{v=1.5}.attr").unwrap().prim_path().as_str(), "Foo{v=1.5}");
    }

    #[test]
    fn append_property_rejects_base() {
        // Bases that cannot own properties error instead of minting an
        // unparseable path.
        assert!(Path::abs_root().append_property("y").is_err());
        assert!(raw("").append_property("y").is_err());
        assert!(Path::new("..").unwrap().append_property("y").is_err());
        assert!(Path::new(".").unwrap().append_property("y").is_err());
        // A relational-attribute target still owns properties.
        assert!(Path::new("/A.rel[/T]").unwrap().append_property("x").is_ok());
    }

    #[test]
    fn append_path_rejects_anchored() {
        // `.`-anchored arguments cannot attach under a prim namespace.
        assert!(Path::new("/A/B").unwrap().append_path("../C").is_err());
        assert!(Path::new("/A").unwrap().append_path(".attr").is_err());
        // The reflexive base is the identity anchor.
        assert_eq!(Path::new(".").unwrap().append_path("C/D").unwrap().as_str(), "C/D");
    }

    #[test]
    fn variant_selection_rejects() {
        let p = Path::new("/A").unwrap();
        assert!(p.append_variant_selection("bad name", "x").is_err());
        assert!(p.append_variant_selection("v", "bad sel").is_err());
        assert!(p.append_variant_selection("", "x").is_err());
        assert!(Path::abs_root().append_variant_selection("v", "x").is_err());
        assert!(
            Path::new("/A.attr")
                .unwrap()
                .append_variant_selection("v", "x")
                .is_err()
        );
        // The variant alphabet allows `.`, `-`, and `|`.
        assert_eq!(
            p.append_variant_selection("v", "hi-res.2|b").unwrap().as_str(),
            "/A{v=hi-res.2|b}"
        );
    }

    #[test]
    fn make_absolute_unrepresentable() {
        // A property-relative step anchored at the pseudo-root has no
        // representable result: the empty path, as in C++.
        let anchor = Path::new("/X.r").unwrap();
        assert!(anchor.make_absolute(&Path::new("../.y").unwrap()).is_empty());
        assert_eq!(anchor.make_absolute(&Path::new("../Y.y").unwrap()).as_str(), "/Y.y");
    }

    #[test]
    fn unicode_alphabet_agreement() {
        // The lenient non-ASCII identifier rule holds across the read-side
        // classifiers, not just the validator.
        let p = Path::new("/A.p\u{2603}").unwrap();
        assert!(p.is_property());
        assert_eq!(p.parent().unwrap().as_str(), "/A");
        let stripped = Path::new("/A{v=x}\u{2206}B").unwrap().strip_variant_selections();
        assert_eq!(stripped.as_str(), "/A/\u{2206}B");
    }

    /// Every derivation of a parsed path must itself re-validate — the
    /// contract `from_str_unchecked` debug-asserts.
    #[test]
    fn derived_revalidate() {
        for case in ["/A/B/C.attr", "/A{x=y}B", "../C.foo", "/A.rel[/T].attr"] {
            let path = Path::new(case).unwrap();
            for derived in [path.prim_path(), path.strip_variant_selections()] {
                let text = derived.as_str();
                assert!(text.is_empty() || Path::new(text).is_ok(), "derived {text} of {case}");
            }
            for ancestor in path.ancestors() {
                assert!(Path::new(ancestor.as_str()).is_ok(), "ancestor {ancestor} of {case}");
            }
        }
    }
}
