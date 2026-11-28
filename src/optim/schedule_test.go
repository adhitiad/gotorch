package optim

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

// rateRecorder merekam learning rate tanpa memvalidasi rentangnya, sehingga
// jadwal bisa diuji sampai ke nilai minimum yang ditolak optimizer sungguhan.
type rateRecorder struct {
	rate  float64
	calls int
}

func (r *rateRecorder) Name() string { return "recorder" }
func (r *rateRecorder) Step(*nn.Context, []Param) error {
	return nil
}
func (r *rateRecorder) SetLearningRate(value float64) error {
	r.rate = value
	r.calls++
	return nil
}
func (r *rateRecorder) LearningRate() float64 { return r.rate }
func (r *rateRecorder) MarshalState([]Param) ([]byte, error) {
	return nil, nil
}
func (r *rateRecorder) UnmarshalState([]Param, []byte) error { return nil }
func (r *rateRecorder) Close() error                         { return nil }

func TestSchedulesProduceExpectedSequence(t *testing.T) {
	cases := []struct {
		name     string
		make     func() (Scheduler, error)
		base     float64
		sequence []float64
	}{
		{
			name:     "constant",
			make:     func() (Scheduler, error) { return ConstantSchedule(0.1) },
			base:     0.1,
			sequence: []float64{0.1, 0.1, 0.1},
		},
		{
			name:     "exponential",
			make:     func() (Scheduler, error) { return ExponentialSchedule(0.1, 0.5) },
			base:     0.1,
			sequence: []float64{0.05, 0.025, 0.0125},
		},
		{
			name:     "steplr",
			make:     func() (Scheduler, error) { return StepLR(0.1, 2, 0.5) },
			base:     0.1,
			sequence: []float64{0.1, 0.05, 0.05, 0.025},
		},
		{
			name:     "linear",
			make:     func() (Scheduler, error) { return LinearSchedule(0.2, 0.02, 4) },
			base:     0.2,
			sequence: []float64{0.155, 0.11, 0.065, 0.02, 0.02},
		},
		{
			name: "cosine",
			make: func() (Scheduler, error) { return CosineSchedule(1, 0, 4) },
			base: 1,
			sequence: []float64{
				0.8535533905932737, 0.5, 0.14644660940672627, 0, 0,
			},
		},
		{
			name:     "warmup",
			make:     func() (Scheduler, error) { return WarmupSchedule(0.01, 0.1, 3) },
			base:     0.1,
			sequence: []float64{0.04, 0.07, 0.1, 0.1},
		},
	}

	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			scheduler, err := item.make()
			if err != nil {
				t.Fatalf("constructor: %v", err)
			}
			if got := scheduler.LastLearningRate(); got != item.base {
				t.Fatalf("nilai awal: got %g, want %g", got, item.base)
			}
			recorder := &rateRecorder{}
			for index, want := range item.sequence {
				if err := scheduler.Step(recorder); err != nil {
					t.Fatalf("Step %d: %v", index+1, err)
				}
				if got := scheduler.LastLearningRate(); mathClose(got, want) == false {
					t.Fatalf("LastLearningRate %d: got %.17g, want %.17g", index+1, got, want)
				}
				if mathClose(recorder.rate, want) == false {
					t.Fatalf("optimizer rate %d: got %.17g, want %.17g", index+1, recorder.rate, want)
				}
			}
			if recorder.calls != len(item.sequence) {
				t.Fatalf("jumlah pemanggilan SetLearningRate: got %d, want %d", recorder.calls, len(item.sequence))
			}
			if err := scheduler.Reset(); err != nil {
				t.Fatalf("Reset: %v", err)
			}
			if got := scheduler.LastLearningRate(); got != item.base {
				t.Fatalf("nilai setelah Reset: got %g, want %g", got, item.base)
			}
			if err := scheduler.Step(recorder); err != nil {
				t.Fatalf("Step setelah Reset: %v", err)
			}
			if !mathClose(recorder.rate, item.sequence[0]) {
				t.Fatalf("Step setelah Reset: got %g, want %g", recorder.rate, item.sequence[0])
			}
		})
	}
}

func TestSchedulerRejectsInvalidConfiguration(t *testing.T) {
	cases := []struct {
		name string
		make func() (Scheduler, error)
	}{
		{"constant nol", func() (Scheduler, error) { return ConstantSchedule(0) }},
		{"exponential gamma satu", func() (Scheduler, error) { return ExponentialSchedule(0.1, 1) }},
		{"exponential gamma nol", func() (Scheduler, error) { return ExponentialSchedule(0.1, 0) }},
		{"step size nol", func() (Scheduler, error) { return StepLR(0.1, 0, 0.5) }},
		{"step gamma satu", func() (Scheduler, error) { return StepLR(0.1, 2, 1) }},
		{"linear menaik", func() (Scheduler, error) { return LinearSchedule(0.1, 0.2, 10) }},
		{"linear total nol", func() (Scheduler, error) { return LinearSchedule(0.1, 0, 0) }},
		{"cosine menaik", func() (Scheduler, error) { return CosineSchedule(0.1, 0.2, 10) }},
		{"cosine total negatif", func() (Scheduler, error) { return CosineSchedule(0.1, 0, -1) }},
		{"warmup langkah nol", func() (Scheduler, error) { return WarmupSchedule(0.01, 0.1, 0) }},
		{"warmup start melebihi base", func() (Scheduler, error) { return WarmupSchedule(0.2, 0.1, 3) }},
		{"warmup start negatif", func() (Scheduler, error) { return WarmupSchedule(-1, 0.1, 3) }},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			if _, err := item.make(); !errors.Is(err, errs.ErrInvalidArgument) {
				t.Fatalf("got %v, want ErrInvalidArgument", err)
			}
		})
	}
}

func TestSchedulerIntegratesWithRealOptimizer(t *testing.T) {
	optimizer, err := NewSGD(SGDConfig{LearningRate: 0.1})
	if err != nil {
		t.Fatalf("NewSGD: %v", err)
	}
	defer optimizer.Close()
	schedulers := []struct {
		name string
		make func() (Scheduler, error)
	}{
		{"constant", func() (Scheduler, error) { return ConstantSchedule(0.1) }},
		{"exponential", func() (Scheduler, error) { return ExponentialSchedule(0.1, 0.5) }},
		{"warmup", func() (Scheduler, error) { return WarmupSchedule(0.02, 0.1, 2) }},
	}
	for _, item := range schedulers {
		t.Run(item.name, func(t *testing.T) {
			scheduler, err := item.make()
			if err != nil {
				t.Fatalf("constructor: %v", err)
			}
			for index := 0; index < 3; index++ {
				if err := scheduler.Step(optimizer); err != nil {
					t.Fatalf("Step %d: %v", index+1, err)
				}
				if got := optimizer.LearningRate(); !mathClose(got, scheduler.LastLearningRate()) {
					t.Fatalf("Step %d: optimizer %g, scheduler %g", index+1, got, scheduler.LastLearningRate())
				}
			}
			if err := scheduler.Reset(); err != nil {
				t.Fatalf("Reset: %v", err)
			}
			if err := scheduler.Step(optimizer); err != nil {
				t.Fatalf("Step setelah Reset: %v", err)
			}
		})
	}
}

func TestSchedulerNilOptimizerDoesNotAdvance(t *testing.T) {
	scheduler, err := ConstantSchedule(0.1)
	if err != nil {
		t.Fatalf("ConstantSchedule: %v", err)
	}
	if err := scheduler.Step(nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Step(nil): got %v", err)
	}
	if got := scheduler.LastLearningRate(); got != 0.1 {
		t.Fatalf("nilai setelah Step(nil): got %g", got)
	}
	recorder := &rateRecorder{}
	if err := scheduler.Step(recorder); err != nil {
		t.Fatalf("Step: %v", err)
	}
	if recorder.calls != 1 {
		t.Fatalf("jumlah pemanggilan: got %d", recorder.calls)
	}
}
