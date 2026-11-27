package compute

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/adhitiad/gotorch/src/errs"
)

// ExecutionPolicy budgets process-wide BLAS threads against concurrent GoTorch workers.
// GoWorkers is the maximum number of GoTorch operations that may enter compute concurrently.
type ExecutionPolicy struct {
	GoWorkers  int
	BLASThreads int
}

// DefaultExecutionPolicy gives one operation the available logical processors.
func DefaultExecutionPolicy() ExecutionPolicy {
	return ExecutionPolicy{GoWorkers: 1, BLASThreads: runtime.GOMAXPROCS(0)}
}

// Validate rejects ambiguous zero or negative concurrency limits.
func (p ExecutionPolicy) Validate() error {
	if p.GoWorkers < 1 || p.BLASThreads < 1 {
		return fmt.Errorf("%w: GoWorkers and BLASThreads must be positive", errs.ErrInvalidArgument)
	}
	return nil
}

// EffectiveBLASThreads caps nested BLAS parallelism so GoWorkers*BLASThreads
// never exceeds GOMAXPROCS (except when GoWorkers itself exceeds that limit).
func (p ExecutionPolicy) EffectiveBLASThreads() int {
	procs := runtime.GOMAXPROCS(0)
	workers, threads := p.GoWorkers, p.BLASThreads
	if workers < 1 {
		workers = 1
	}
	if threads < 1 {
		threads = 1
	}
	budget := procs / workers
	if budget < 1 {
		budget = 1
	}
	if threads > budget {
		threads = budget
	}
	return threads
}

// OpenBLASConfig explicitly opts into a dynamically loaded OpenBLAS library.
type OpenBLASConfig struct {
	LibraryPath string
	Policy      ExecutionPolicy
}

// DefaultOpenBLASConfig reads the opt-in library path from GOTORCH_OPENBLAS_LIBRARY.
func DefaultOpenBLASConfig() OpenBLASConfig {
	return OpenBLASConfig{
		LibraryPath: strings.TrimSpace(os.Getenv("GOTORCH_OPENBLAS_LIBRARY")),
		Policy:      DefaultExecutionPolicy(),
	}
}
