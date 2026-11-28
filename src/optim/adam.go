package optim

import (
	"fmt"
	"math"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

// AdamConfig mengonfigurasi Adam/AdamW. AdamW memakai weight decay terpisah.
type AdamConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	WeightDecay  float64
	AdamW        bool
}

// Adam mengimplementasikan Adam dan AdamW dengan state per parameter.
type Adam struct {
	mu      sync.Mutex
	rate    float64
	beta1   float64
	beta2   float64
	epsilon float64
	decay   float64
	adamW   bool
	steps   map[string]int64
	moments optimizerState
	closed  bool
}

// NewAdam membuat optimizer Adam atau AdamW.
func NewAdam(config AdamConfig) (*Adam, error) {
	if err := validateLearningRate(config.LearningRate); err != nil {
		return nil, err
	}
	beta1, beta2 := config.Beta1, config.Beta2
	if beta1 == 0 {
		beta1 = 0.9
	}
	if beta2 == 0 {
		beta2 = 0.999
	}
	epsilon := config.Epsilon
	if epsilon == 0 {
		epsilon = 1e-8
	}
	if beta1 <= 0 || beta1 >= 1 || beta2 <= 0 || beta2 >= 1 || epsilon <= 0 || config.WeightDecay < 0 {
		return nil, fmt.Errorf("%w: invalid Adam configuration", errs.ErrInvalidArgument)
	}
	return &Adam{
		rate:    config.LearningRate,
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
		decay:   config.WeightDecay,
		adamW:   config.AdamW,
		steps:   make(map[string]int64),
		moments: make(optimizerState),
	}, nil
}

// Name mengembalikan identifier optimizer.
func (o *Adam) Name() string {
	if o.adamW {
		return "adamw"
	}
	return "adam"
}

// LearningRate mengembalikan learning rate saat ini.
func (o *Adam) LearningRate() float64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.rate
}

// SetLearningRate mengubah learning rate untuk langkah berikutnya.
func (o *Adam) SetLearningRate(rate float64) error {
	if err := validateLearningRate(rate); err != nil {
		return err
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return errs.ErrInvalidState
	}
	o.rate = rate
	return nil
}

// Step memperbarui semua parameter yang memiliki gradien pada Context.
func (o *Adam) Step(ctx *nn.Context, parameters []Param) error {
	ordered, err := validateParameters(parameters)
	if err != nil {
		return err
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return errs.ErrInvalidState
	}
	for _, item := range ordered {
		count, err := paramCount(item.Parameter)
		if err != nil {
			return err
		}
		gradient, hasGradient, err := gradientValues(ctx, item.Parameter)
		if err != nil {
			return err
		}
		entry := o.moments[item.Name]
		if len(entry.vectors) != 2 || len(entry.vectors[0]) != count || len(entry.vectors[1]) != count {
			entry = stateEntry{vectors: zeroVectors(count, count)}
		}
		if !hasGradient {
			o.moments[item.Name] = entry
			continue
		}
		if len(gradient) != count {
			return fmt.Errorf("%w: gradient for %q has %d values, want %d", errs.ErrGradientShapeMismatch, item.Name, len(gradient), count)
		}
		step := o.steps[item.Name] + 1
		biasCorrection1 := 1 - math.Pow(o.beta1, float64(step))
		biasCorrection2 := 1 - math.Pow(o.beta2, float64(step))
		if biasCorrection1 == 0 || biasCorrection2 == 0 {
			return fmt.Errorf("%w: Adam bias correction underflow for %q", errs.ErrInvalidState, item.Name)
		}
		current, err := readFloatValues(item.Parameter.Value())
		if err != nil {
			return err
		}
		applied := make([]float64, count)
		for index := 0; index < count; index++ {
			value := gradient[index]
			if o.decay != 0 && !o.adamW {
				value += o.decay * current[index]
			}
			entry.vectors[0][index] = o.beta1*entry.vectors[0][index] + (1-o.beta1)*value
			entry.vectors[1][index] = o.beta2*entry.vectors[1][index] + (1-o.beta2)*value*value
			correctedFirst := entry.vectors[0][index] / biasCorrection1
			correctedSecond := entry.vectors[1][index] / biasCorrection2
			applied[index] = correctedFirst / (math.Sqrt(correctedSecond) + o.epsilon)
			if o.adamW && o.decay != 0 {
				applied[index] += o.decay * current[index]
			}
		}
		o.steps[item.Name] = step
		o.moments[item.Name] = entry
		if err := item.Parameter.Update(func(values []float64) ([]float64, error) {
			for index := range values {
				values[index] -= o.rate * applied[index]
			}
			return values, nil
		}); err != nil {
			return err
		}
	}
	return nil
}

// MarshalState menyandikan langkah dan momen pertama/kedua setiap parameter.
func (o *Adam) MarshalState(parameters []Param) ([]byte, error) {
	ordered, err := validateParameters(parameters)
	if err != nil {
		return nil, err
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	entries := make(optimizerState, len(ordered))
	for _, item := range ordered {
		count, err := paramCount(item.Parameter)
		if err != nil {
			return nil, err
		}
		entry := o.moments[item.Name]
		if len(entry.vectors) != 2 || len(entry.vectors[0]) != count || len(entry.vectors[1]) != count {
			entry = stateEntry{vectors: zeroVectors(count, count)}
		}
		entries[item.Name] = stateEntry{
			scalars: []float64{float64(o.steps[item.Name])},
			vectors: [][]float64{
				append([]float64(nil), entry.vectors[0]...),
				append([]float64(nil), entry.vectors[1]...),
			},
		}
	}
	return encodeState(entries)
}

// UnmarshalState memulihkan momen dan menolak state yang tidak cocok.
func (o *Adam) UnmarshalState(parameters []Param, data []byte) error {
	ordered, err := validateParameters(parameters)
	if err != nil {
		return err
	}
	decoded, err := decodeState(data)
	if err != nil {
		return err
	}
	restored := make(optimizerState, len(ordered))
	steps := make(map[string]int64, len(ordered))
	for _, item := range ordered {
		count, err := paramCount(item.Parameter)
		if err != nil {
			return err
		}
		entry, exists := decoded[item.Name]
		if !exists {
			return fmt.Errorf("%w: missing optimizer state %q", errs.ErrInvalidArgument, item.Name)
		}
		if err := validateStateEntry(item.Name, entry, 1, count, count); err != nil {
			return err
		}
		step := entry.scalars[0]
		// float64(math.MaxInt64) pembulatannya menjadi 2^63, jadi batas atas
		// harus eksklusif agar tidak dikonversi menjadi MinInt64.
		if step < 0 || step >= float64(math.MaxInt64) || math.Trunc(step) != step {
			return fmt.Errorf("%w: invalid optimizer step for %q", errs.ErrInvalidState, item.Name)
		}
		steps[item.Name] = int64(step)
		restored[item.Name] = stateEntry{vectors: [][]float64{
			append([]float64(nil), entry.vectors[0]...),
			append([]float64(nil), entry.vectors[1]...),
		}}
	}
	if len(decoded) != len(ordered) {
		return fmt.Errorf("%w: optimizer state contains unknown parameters", errs.ErrInvalidArgument)
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return errs.ErrInvalidState
	}
	o.steps = steps
	o.moments = restored
	return nil
}

// Close membersihkan state dan membuat optimizer tidak bisa dipakai lagi.
func (o *Adam) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.steps = nil
	o.moments = nil
	o.closed = true
	return nil
}
