# Platform Support Policy

| Platform | Role | Required validation |
|---|---|---|
| Windows AMD64 | Primary production | Native build and tests on Windows runner |
| Linux AMD64 | Compatibility | Native build and tests |
| Linux ARM64 | Compatibility | Native ARM64 tests when runner exists; otherwise cross-build clearly marked |

## Portability rules

- Core packages must not require CGo.
- SIMD code must be isolated by architecture/build constraints and runtime feature checks.
- BLAS/LAPACK adapters are optional and must not block Pure Go builds.
- Use `filepath` and platform-safe file operations.
- Do not assume Unix signals, file locks, mmap semantics, or atomic rename behavior are identical across platforms.
- Validate Windows DLL loading only for optional integrations; core should not depend on vendor DLLs.
