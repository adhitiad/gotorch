package optim

import (
	"fmt"
	"math"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
)

// Scheduler mengubah learning rate optimizer setelah setiap pemanggilan Step.
type Scheduler interface {
	Step(optimizer Optimizer) error
	LastLearningRate() float64
	Reset() error
}

type baseScheduler struct {
	mu       sync.Mutex
	base     float64
	current  float64
	step     int64
	evaluate func(int64) float64
}

func newBaseScheduler(base float64, evaluate func(int64) float64) (*baseScheduler, error) {
	if base <= 0 || evaluate == nil {
		return nil, errs.ErrInvalidArgument
	}
	return &baseScheduler{base: base, current: base, evaluate: evaluate}, nil
}

func (s *baseScheduler) Step(optimizer Optimizer) error {
	if optimizer == nil {
		return errs.ErrInvalidArgument
	}
	s.mu.Lock()
	s.step++
	s.current = s.evaluate(s.step)
	value := s.current
	s.mu.Unlock()
	return optimizer.SetLearningRate(value)
}

func (s *baseScheduler) LastLearningRate() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

func (s *baseScheduler) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.step = 0
	s.current = s.base
	return nil
}

// ConstantSchedule menjaga learning rate tetap.
func ConstantSchedule(base float64) (Scheduler, error) {
	return newBaseScheduler(base, func(int64) float64 { return base })
}

// ExponentialSchedule mengalikan learning rate dengan gamma setiap langkah.
func ExponentialSchedule(base, gamma float64) (Scheduler, error) {
	if gamma <= 0 || gamma >= 1 {
		return nil, fmt.Errorf("%w: exponential gamma %g outside (0,1)", errs.ErrInvalidArgument, gamma)
	}
	return newBaseScheduler(base, func(step int64) float64 {
		return base * math.Pow(gamma, float64(step))
	})
}

// StepLR menurunkan learning rate dengan faktor gamma setiap stepSize langkah.
func StepLR(base float64, stepSize int64, gamma float64) (Scheduler, error) {
	if stepSize <= 0 || gamma <= 0 || gamma >= 1 {
		return nil, fmt.Errorf("%w: invalid step schedule", errs.ErrInvalidArgument)
	}
	return newBaseScheduler(base, func(step int64) float64 {
		return base * math.Pow(gamma, math.Floor(float64(step)/float64(stepSize)))
	})
}

// LinearSchedule berubah linier dari base ke end selama totalSteps langkah.
func LinearSchedule(base, end float64, totalSteps int64) (Scheduler, error) {
	if totalSteps <= 0 || end < 0 || end > base {
		return nil, fmt.Errorf("%w: invalid linear schedule", errs.ErrInvalidArgument)
	}
	return newBaseScheduler(base, func(step int64) float64 {
		fraction := float64(step) / float64(totalSteps)
		if fraction > 1 {
			fraction = 1
		}
		return base + (end-base)*fraction
	})
}

// CosineSchedule mengikuti annealing kosinus hingga minimum yang ditentukan.
func CosineSchedule(base, minimum float64, totalSteps int64) (Scheduler, error) {
	if totalSteps <= 0 || minimum < 0 || minimum > base {
		return nil, fmt.Errorf("%w: invalid cosine schedule", errs.ErrInvalidArgument)
	}
	return newBaseScheduler(base, func(step int64) float64 {
		fraction := float64(step) / float64(totalSteps)
		if fraction > 1 {
			fraction = 1
		}
		return minimum + 0.5*(base-minimum)*(1+math.Cos(math.Pi*fraction))
	})
}

// WarmupSchedule menaikkan learning rate linier dari start ke base selama warmupSteps.
func WarmupSchedule(start, base float64, warmupSteps int64) (Scheduler, error) {
	if warmupSteps <= 0 || start < 0 || start > base {
		return nil, fmt.Errorf("%w: invalid warmup schedule", errs.ErrInvalidArgument)
	}
	return newBaseScheduler(base, func(step int64) float64 {
		if step >= warmupSteps {
			return base
		}
		fraction := float64(step) / float64(warmupSteps)
		return start + (base-start)*fraction
	})
}
