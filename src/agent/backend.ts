import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import fg from "fast-glob";
import {
  LocalShellBackend,
  type GlobResult,
  type GrepResult,
  type FileInfo,
  type LocalShellBackendOptions,
} from "deepagents";

/**
 * Directories that must never be traversed by discovery tools. Scanning them is
 * both wasteful for documentation work and a common source of symlink cycles
 * (e.g. pnpm's `node_modules/<self> -> ..`), which crash naive walkers with
 * `ENAMETOOLONG`.
 */
const IGNORED_DIRECTORIES: string[] = ["**/node_modules/**", "**/.git/**"];

const DEFAULT_MAX_FILE_SIZE_MB = 10;

type LiteralMatches = Record<string, Array<[number, string]>>;

/**
 * `LocalShellBackend` whose `glob`/`grep` never follow symbolic links and always
 * skip `node_modules`/`.git`.
 *
 * The upstream backend calls `fast-glob` with its default `followSymbolicLinks:
 * true` and without any ignore list, so a self-referential symlink inside
 * `node_modules` makes the walk recurse forever until the path exceeds the OS
 * limit and throws an uncaught `ENAMETOOLONG`, aborting the whole run. These
 * overrides make discovery resilient to such repositories.
 */
export class SafeLocalShellBackend extends LocalShellBackend {
  private readonly safeCwd: string;
  private readonly safeVirtualMode: boolean;
  private readonly safeMaxFileSizeBytes: number;

  constructor(options: LocalShellBackendOptions = {}) {
    super(options);
    this.safeCwd = options.rootDir
      ? path.resolve(options.rootDir)
      : process.cwd();
    this.safeVirtualMode = options.virtualMode ?? false;
    const maxFileSizeMb =
      (options as { maxFileSizeMb?: number }).maxFileSizeMb ??
      DEFAULT_MAX_FILE_SIZE_MB;
    this.safeMaxFileSizeBytes = maxFileSizeMb * 1024 * 1024;
  }

  override async glob(
    pattern: string,
    searchPath = "/",
  ): Promise<GlobResult> {
    const normalizedPattern = pattern.startsWith("/")
      ? pattern.substring(1)
      : pattern;
    const resolvedSearchPath = this.resolveSearchPath(searchPath);

    try {
      if (!(await stat(resolvedSearchPath)).isDirectory()) {
        return { files: [] };
      }
    } catch {
      return { files: [] };
    }

    const globOptions = {
      cwd: resolvedSearchPath,
      absolute: false,
      dot: true,
      followSymbolicLinks: false,
      ignore: [...IGNORED_DIRECTORIES],
    };

    let fileMatches: string[];
    let dirMatches: string[];

    try {
      [fileMatches, dirMatches] = await Promise.all([
        fg(normalizedPattern, { ...globOptions, onlyFiles: true }),
        fg(normalizedPattern, { ...globOptions, onlyDirectories: true }),
      ]);
    } catch {
      return { files: [] };
    }

    const files: FileInfo[] = [];

    for (const match of fileMatches) {
      const info = await this.describeEntry(resolvedSearchPath, match, false);
      if (info) {
        files.push(info);
      }
    }

    for (const match of dirMatches) {
      const info = await this.describeEntry(resolvedSearchPath, match, true);
      if (info) {
        files.push(info);
      }
    }

    files.sort((left, right) => left.path.localeCompare(right.path));
    return { files };
  }

  override async grep(
    pattern: string,
    dirPath = "/",
    glob: string | null = null,
  ): Promise<GrepResult> {
    let baseFull: string;

    try {
      baseFull = this.resolveBackendPath(dirPath || ".");
    } catch {
      return { matches: [] };
    }

    try {
      await stat(baseFull);
    } catch {
      return { matches: [] };
    }

    // Prefer the inherited ripgrep search: it honors .gitignore and does not
    // follow symlinks, so it is already safe on cyclic repositories.
    let results = await this.runRipgrep(pattern, baseFull, glob);

    if (results === null) {
      results = await this.safeLiteralSearch(pattern, baseFull, glob);
    }

    const matches: GrepResult["matches"] = [];

    for (const [filePath, items] of Object.entries(results)) {
      for (const [line, text] of items) {
        matches.push({ path: filePath, line, text });
      }
    }

    return { matches };
  }

  private resolveSearchPath(searchPath: string): string {
    if (searchPath === "/" || searchPath === "") {
      return this.safeCwd;
    }

    if (this.safeVirtualMode) {
      return path.resolve(this.safeCwd, searchPath.replace(/^\//, ""));
    }

    return path.resolve(this.safeCwd, searchPath);
  }

  private resolveBackendPath(key: string): string {
    return (
      this as unknown as { resolvePath(key: string): string }
    ).resolvePath(key);
  }

  private async runRipgrep(
    pattern: string,
    baseFull: string,
    includeGlob: string | null,
  ): Promise<LiteralMatches | null> {
    const ripgrepSearch = (
      this as unknown as {
        ripgrepSearch?: (
          pattern: string,
          baseFull: string,
          includeGlob: string | null,
        ) => Promise<LiteralMatches | null>;
      }
    ).ripgrepSearch;

    if (typeof ripgrepSearch !== "function") {
      return null;
    }

    return ripgrepSearch.call(this, pattern, baseFull, includeGlob);
  }

  private toVirtualPath(absolutePath: string): string | null {
    const relative = path.relative(this.safeCwd, absolutePath);

    if (relative.startsWith("..") || path.isAbsolute(relative)) {
      return null;
    }

    return `/${relative.split(path.sep).join("/")}`;
  }

  private async describeEntry(
    base: string,
    relativeMatch: string,
    expectDirectory: boolean,
  ): Promise<FileInfo | null> {
    try {
      const entryStat = await stat(path.join(base, relativeMatch));

      if (expectDirectory ? !entryStat.isDirectory() : !entryStat.isFile()) {
        return null;
      }

      return {
        path: this.safeVirtualMode ? `/${relativeMatch}` : relativeMatch,
        is_dir: expectDirectory,
        size: expectDirectory ? 0 : entryStat.size,
        modified_at: entryStat.mtime.toISOString(),
      };
    } catch {
      return null;
    }
  }

  private async safeLiteralSearch(
    pattern: string,
    baseFull: string,
    includeGlob: string | null,
  ): Promise<LiteralMatches> {
    const results: LiteralMatches = {};

    let searchRoot = baseFull;

    try {
      if (!(await stat(baseFull)).isDirectory()) {
        searchRoot = path.dirname(baseFull);
      }
    } catch {
      return results;
    }

    let files: string[];

    try {
      files = await fg(includeGlob ?? "**/*", {
        cwd: searchRoot,
        absolute: true,
        onlyFiles: true,
        dot: true,
        followSymbolicLinks: false,
        ignore: [...IGNORED_DIRECTORIES],
      });
    } catch {
      return results;
    }

    for (const filePath of files) {
      try {
        if ((await stat(filePath)).size > this.safeMaxFileSizeBytes) {
          continue;
        }

        const content = await readFile(filePath);

        if (isProbablyBinary(content)) {
          continue;
        }

        const lines = content.toString("utf-8").split("\n");
        const key = this.safeVirtualMode
          ? this.toVirtualPath(filePath)
          : filePath;

        if (key === null) {
          continue;
        }

        for (let index = 0; index < lines.length; index += 1) {
          if (lines[index].includes(pattern)) {
            (results[key] ??= []).push([index + 1, lines[index]]);
          }
        }
      } catch {
        continue;
      }
    }

    return results;
  }
}

/**
 * Cheap binary sniff: a NUL byte within the first chunk reliably marks a file as
 * non-text, so we skip it during the literal (ripgrep-less) fallback search.
 */
function isProbablyBinary(content: Buffer): boolean {
  const sampleLength = Math.min(content.length, 8000);

  for (let index = 0; index < sampleLength; index += 1) {
    if (content[index] === 0) {
      return true;
    }
  }

  return false;
}
