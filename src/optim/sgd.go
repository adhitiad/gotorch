package optim

import (
	"fmt"
	"sync"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
)

// SGDConfig mengonfigurasi SGD. Weight decay bersifat coupled L2.
type SGDConfig struct {
	LearningRate float64
	Momentum     float64
	WeightDecay  float64
	Nesterov     bool
}

// SGD adalah stochastic gradient descent dengan momentum opsional.
type SGD struct {
	mu         sync.Mutex
	rate       float64
	momentum   float64
	decay      float64
	nesterov   bool
	velocities map[string][]float64
	closed     bool
}

// NewSGD membuat optimizer SGD dan memvalidasi konfigurasinya.
func NewSGD(config SGDConfig) (*SGD, error) {
	if err := validateLearningRate(config.LearningRate); err != nil {
		return nil, err
	}
	if config.Momentum < 0 || config.Momentum >= 1 || config.WeightDecay < 0 {
		return nil, fmt.Errorf("%w: invalid SGD momentum or weight decay", errs.ErrInvalidArgument)
	}
	return &SGD{
		rate:       config.LearningRate,
		momentum:   config.Momentum,
		decay:      config.WeightDecay,
		nesterov:   config.Nesterov,
		velocities: make(map[string][]float64),
	}, nil
}

// Name mengembalikan identifier optimizer.
func (o *SGD) Name() string { return "sgd" }

// LearningRate mengembalikan learning rate saat ini.
func (o *SGD) LearningRate() float64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.rate
}

// SetLearningRate mengubah learning rate untuk langkah berikutnya.
func (o *SGD) SetLearningRate(rate float64) error {
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
func (o *SGD) Step(ctx *nn.Context, parameters []Param) error {
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
		velocity := o.velocities[item.Name]
		if len(velocity) != count {
			velocity = make([]float64, count)
		}
		if !hasGradient {
			o.velocities[item.Name] = velocity
			continue
		}
		if len(gradient) != count {
			return fmt.Errorf("%w: gradient for %q has %d values, want %d", errs.ErrGradientShapeMismatch, item.Name, len(gradient), count)
		}
		current, err := readFloatValues(item.Parameter.Value())
		if err != nil {
			return err
		}
		applied := make([]float64, count)
		for index := 0; index < count; index++ {
			value := gradient[index]
			if o.decay != 0 {
				value += o.decay * current[index]
			}
			if o.momentum == 0 {
				applied[index] = value
				continue
			}
			velocity[index] = o.momentum*velocity[index] + value
			applied[index] = velocity[index]
			if o.nesterov {
				applied[index] += o.momentum * velocity[index]
			}
		}
		o.velocities[item.Name] = velocity
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

// MarshalState menyandikan velocity setiap parameter.
func (o *SGD) MarshalState(parameters []Param) ([]byte, error) {
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
		velocity := append([]float64(nil), o.velocities[item.Name]...)
		if len(velocity) != count {
			velocity = make([]float64, count)
		}
		entries[item.Name] = stateEntry{vectors: [][]float64{velocity}}
	}
	return encodeState(entries)
}

// UnmarshalState memulihkan velocity dan menolak parameter yang tidak cocok.
func (o *SGD) UnmarshalState(parameters []Param, data []byte) error {
	ordered, err := validateParameters(parameters)
	if err != nil {
		return err
	}
	decoded, err := decodeState(data)
	if err != nil {
		return err
	}
	restored := make(map[string][]float64, len(ordered))
	for _, item := range ordered {
		count, err := paramCount(item.Parameter)
		if err != nil {
			return err
		}
		entry, exists := decoded[item.Name]
		if !exists {
			return fmt.Errorf("%w: missing optimizer state %q", errs.ErrInvalidArgument, item.Name)
		}
		if err := validateStateEntry(item.Name, entry, 0, count); err != nil {
			return err
		}
		restored[item.Name] = append([]float64(nil), entry.vectors[0]...)
	}
	if len(decoded) != len(ordered) {
		return fmt.Errorf("%w: optimizer state contains unknown parameters", errs.ErrInvalidArgument)
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return errs.ErrInvalidState
	}
	o.velocities = restored
	return nil
}

// Close membersihkan state dan membuat optimizer tidak bisa dipakai lagi.
func (o *SGD) Close() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.velocities = nil
	o.closed = true
	return nil
}
