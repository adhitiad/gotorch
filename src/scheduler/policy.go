package scheduler

import (
	"fmt"
	"runtime"

	"github.com/adhitiad/gotorch/src/errs"
)

// Policy adalah anggaran bersama antara worker GoTorch dan thread BLAS.
// Rumusnya sengaja sama dengan compute.ExecutionPolicy agar adapter BLAS dan
// scheduler tidak saling melampaui anggaran CPU yang sama.
type Policy struct {
	GoWorkers   int
	BLASThreads int
}

// Validate menolak jumlah worker atau thread yang tidak positif.
func (p Policy) Validate() error {
	if p.GoWorkers < 1 || p.BLASThreads < 1 {
		return fmt.Errorf("%w: GoWorkers and BLASThreads must be positive", errs.ErrInvalidArgument)
	}
	return nil
}

// EffectiveBLASThreads membatasi thread bersarang agar GoWorkers*BLASThreads
// tidak melebihi GOMAXPROCS, kecuali GoWorkers sendiri sudah melebihi jumlah itu.
func (p Policy) EffectiveBLASThreads() int {
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

// Policy mengembalikan anggaran untuk scheduler ini. Nilai blasThreads <= 0
// berarti memakai GOMAXPROCS seperti compute.DefaultExecutionPolicy.
func (s *Scheduler) Policy(blasThreads int) Policy {
	if s == nil {
		return Policy{}
	}
	if blasThreads <= 0 {
		blasThreads = runtime.GOMAXPROCS(0)
	}
	return Policy{GoWorkers: s.config.Workers, BLASThreads: blasThreads}
}
