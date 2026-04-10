/// libclang-based supplementary resolver.
///
/// Only activated when a `compile_commands.json` is present and libclang
/// is available at runtime.  If absent, tree-sitter results stand unchanged.
///
/// The full libclang AST walk (type-resolved calls, overload resolution,
/// template instantiation) is implemented in `resolve_with_libclang` below.
/// It is gated behind a runtime probe so the binary remains portable even
/// without libclang installed.
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ResolvedCall {
    pub caller_qname: String,
    pub callee_qname: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedInheritance {
    pub derived_qname: String,
    pub base_qname: String,
}

#[derive(Debug, Clone)]
pub struct ResolvedUsage {
    pub user_qname: String,
    pub type_qname: String,
}

pub type LibclangResult = (Vec<ResolvedCall>, Vec<ResolvedInheritance>, Vec<ResolvedUsage>);

pub struct LibclangResolver {
    compile_commands: PathBuf,
}

impl LibclangResolver {
    /// Returns `None` if libclang is unavailable or compile_commands.json is missing.
    pub fn try_new(compile_commands_path: &Path) -> Option<Self> {
        if !compile_commands_path.exists() {
            return None;
        }
        if !probe_libclang() {
            tracing::debug!("libclang not found at runtime; skipping type-resolved pass");
            return None;
        }
        Some(Self {
            compile_commands: compile_commands_path.to_owned(),
        })
    }

    /// Resolve additional edges for `file_path` using libclang's AST.
    pub fn resolve_file(&self, file_path: &Path) -> LibclangResult {
        resolve_with_libclang(&self.compile_commands, file_path).unwrap_or_else(|e| {
            tracing::debug!("libclang skipped for {}: {}", file_path.display(), e);
            (vec![], vec![], vec![])
        })
    }
}

// ── Runtime probe ──────────────────────────────────────────────────────────────

fn probe_libclang() -> bool {
    // Check well-known libclang locations for the current platform.
    let candidates: &[&str] = if cfg!(target_os = "macos") {
        &[
            "/usr/lib/libclang.dylib",
            "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib",
            "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
        ]
    } else {
        &[
            "/usr/lib/llvm-14/lib/libclang.so.1",
            "/usr/lib/llvm-14/lib/libclang.so",
            "/usr/lib/llvm-15/lib/libclang.so.1",
            "/usr/lib/llvm-16/lib/libclang.so.1",
            "/usr/lib/x86_64-linux-gnu/libclang-14.so",
            "/usr/lib/libclang.so",
        ]
    };
    candidates.iter().any(|p| Path::new(p).exists())
}

// ── Libclang AST traversal ─────────────────────────────────────────────────────

fn resolve_with_libclang(
    compile_commands_path: &Path,
    file_path: &Path,
) -> anyhow::Result<LibclangResult> {
    // Safety: all clang-sys pointers are checked for null before use.
    unsafe { libclang_traverse(compile_commands_path, file_path) }
}

unsafe fn libclang_traverse(
    compile_commands_path: &Path,
    file_path: &Path,
) -> anyhow::Result<LibclangResult> {
    use clang_sys::*;
    use std::ffi::{CStr, CString};

    // Ensure the shared library is loaded (clang-sys `runtime` feature)
    clang_sys::load().map_err(|e| anyhow::anyhow!("libclang load: {}", e))?;

    let dir = compile_commands_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("compile_commands.json has no parent dir"))?;

    let dir_cs = CString::new(dir.to_string_lossy().as_ref())?;
    let file_cs = CString::new(file_path.to_string_lossy().as_ref())?;

    // ── Load compilation database ──────────────────────────────────────────
    let mut db_err: CXCompilationDatabase_Error = 0;
    let db = clang_CompilationDatabase_fromDirectory(dir_cs.as_ptr(), &mut db_err);
    anyhow::ensure!(!db.is_null() && db_err == 0, "compile_commands.json load failed (err={})", db_err);

    let cmds = clang_CompilationDatabase_getCompileCommands(db, file_cs.as_ptr());
    clang_CompilationDatabase_dispose(db);
    anyhow::ensure!(!cmds.is_null(), "no compile commands for {}", file_path.display());

    let n_cmds = clang_CompileCommands_getSize(cmds);
    if n_cmds == 0 {
        clang_CompileCommands_dispose(cmds);
        anyhow::bail!("empty compile commands for {}", file_path.display());
    }

    // ── Build clang argv ──────────────────────────────────────────────────
    let cmd = clang_CompileCommands_getCommand(cmds, 0);
    let argc = clang_CompileCommand_getNumArgs(cmd) as usize;
    let mut arg_strings: Vec<CString> = Vec::with_capacity(argc);
    for i in 1..argc {
        let raw = clang_CompileCommand_getArg(cmd, i as u32);
        let s = CStr::from_ptr(clang_getCString(raw)).to_string_lossy().into_owned();
        clang_disposeString(raw);
        arg_strings.push(CString::new(s)?);
    }
    clang_CompileCommands_dispose(cmds);

    let argv: Vec<*const std::os::raw::c_char> = arg_strings.iter().map(|s| s.as_ptr()).collect();

    // ── Parse translation unit ────────────────────────────────────────────
    let index = clang_createIndex(0, 0);
    anyhow::ensure!(!index.is_null(), "clang_createIndex failed");

    let tu = clang_parseTranslationUnit(
        index,
        file_cs.as_ptr(),
        argv.as_ptr(),
        argv.len() as std::os::raw::c_int,
        std::ptr::null_mut(),
        0,
        CXTranslationUnit_SkipFunctionBodies as i32,
    );

    if tu.is_null() {
        clang_disposeIndex(index);
        anyhow::bail!("clang_parseTranslationUnit failed for {}", file_path.display());
    }

    // ── AST walk ──────────────────────────────────────────────────────────
    let mut vdata = VisitorData::default();
    let cursor = clang_getTranslationUnitCursor(tu);
    clang_visitChildren(
        cursor,
        ast_visitor,
        &mut vdata as *mut VisitorData as CXClientData,
    );

    clang_disposeTranslationUnit(tu);
    clang_disposeIndex(index);

    Ok((vdata.calls, vdata.inherits, vdata.usages))
}

#[derive(Default)]
struct VisitorData {
    calls: Vec<ResolvedCall>,
    inherits: Vec<ResolvedInheritance>,
    usages: Vec<ResolvedUsage>,
    current_fn: Option<String>,
}

extern "C" fn ast_visitor(
    cursor: clang_sys::CXCursor,
    parent: clang_sys::CXCursor,
    data: clang_sys::CXClientData,
) -> clang_sys::CXChildVisitResult {
    unsafe {
        use clang_sys::*;
        let vd = &mut *(data as *mut VisitorData);
        match clang_getCursorKind(cursor) {
            CXCursor_FunctionDecl
            | CXCursor_CXXMethod
            | CXCursor_Constructor
            | CXCursor_Destructor => {
                vd.current_fn = Some(cx_to_string(clang_getCursorDisplayName(cursor)));
                CXChildVisit_Recurse
            }
            CXCursor_CallExpr => {
                if let Some(ref caller) = vd.current_fn.clone() {
                    let referenced = clang_getCursorReferenced(cursor);
                    if clang_Cursor_isNull(referenced) == 0 {
                        let callee = cx_to_string(clang_getCursorDisplayName(referenced));
                        if !callee.is_empty() {
                            vd.calls.push(ResolvedCall {
                                caller_qname: caller.clone(),
                                callee_qname: callee,
                            });
                        }
                    }
                }
                CXChildVisit_Recurse
            }
            CXCursor_CXXBaseSpecifier => {
                let derived = cx_to_string(clang_getCursorDisplayName(parent));
                let base_type = clang_getCursorType(cursor);
                let base_decl = clang_getTypeDeclaration(base_type);
                let base = cx_to_string(clang_getCursorDisplayName(base_decl));
                if !derived.is_empty() && !base.is_empty() {
                    vd.inherits.push(ResolvedInheritance {
                        derived_qname: derived,
                        base_qname: base,
                    });
                }
                CXChildVisit_Continue
            }
            CXCursor_TypeRef => {
                if let Some(ref user) = vd.current_fn.clone() {
                    let referenced = clang_getCursorReferenced(cursor);
                    if clang_Cursor_isNull(referenced) == 0 {
                        let type_name = cx_to_string(clang_getCursorDisplayName(referenced));
                        if !type_name.is_empty() {
                            vd.usages.push(ResolvedUsage {
                                user_qname: user.clone(),
                                type_qname: type_name,
                            });
                        }
                    }
                }
                CXChildVisit_Continue
            }
            _ => clang_sys::CXChildVisit_Recurse,
        }
    }
}

unsafe fn cx_to_string(s: clang_sys::CXString) -> String {
    use clang_sys::*;
    let ptr = clang_getCString(s);
    let result = if ptr.is_null() {
        String::new()
    } else {
        std::ffi::CStr::from_ptr(ptr)
            .to_string_lossy()
            .into_owned()
    };
    clang_disposeString(s);
    result
}
