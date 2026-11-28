package peft

import (
	"fmt"
	"sort"
	"sync"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/nn"
	"github.com/adhitiad/gotorch/src/tensor"
)

const mergedStateKey = "__merged"

// Adapter mengumpulkan beberapa LoRALinear dan mengelola state serta merge-nya.
type Adapter struct {
	*nn.ModuleBase
	mu       sync.RWMutex
	wrappers []*LoRALinear
}

// NewAdapter membuat wadah adapter dengan nama yang divalidasi.
func NewAdapter(name string) (*Adapter, error) {
	base, err := nn.NewModule(name)
	if err != nil {
		return nil, err
	}
	return &Adapter{ModuleBase: base}, nil
}

// WrapLinear membungkus base dengan LoRA dan mendaftarkannya pada adapter.
func (a *Adapter) WrapLinear(name string, base Projector, config LoRAConfig) (*LoRALinear, error) {
	if a == nil || a.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	wrapper, err := NewLoRALinear(name, base, config)
	if err != nil {
		return nil, err
	}
	if err := a.ModuleBase.RegisterModule(wrapper); err != nil {
		_ = wrapper.Close()
		return nil, err
	}
	a.mu.Lock()
	a.wrappers = append(a.wrappers, wrapper)
	a.mu.Unlock()
	return wrapper, nil
}

// Wrappers mengembalikan salinan daftar adapter dalam urutan nama.
func (a *Adapter) Wrappers() []*LoRALinear {
	if a == nil {
		return nil
	}
	a.mu.RLock()
	result := append([]*LoRALinear(nil), a.wrappers...)
	a.mu.RUnlock()
	sort.Slice(result, func(i, j int) bool { return result[i].Name() < result[j].Name() })
	return result
}

// Merge menyatukan semua adapter. Jika satu gagal, yang sudah menyatu dikembalikan.
func (a *Adapter) Merge() error {
	return a.setMerged(true)
}

// Unmerge memisahkan semua adapter. Jika satu gagal, yang sudah dipisah disatukan kembali.
func (a *Adapter) Unmerge() error {
	return a.setMerged(false)
}

func (a *Adapter) setMerged(target bool) error {
	wrappers := a.Wrappers()
	completed := make([]*LoRALinear, 0, len(wrappers))
	for _, wrapper := range wrappers {
		if target {
			err := wrapper.Merge()
			if err != nil {
				a.rollback(completed, false)
				return err
			}
		} else {
			err := wrapper.Unmerge()
			if err != nil {
				a.rollback(completed, true)
				return err
			}
		}
		completed = append(completed, wrapper)
	}
	return nil
}

func (a *Adapter) rollback(completed []*LoRALinear, merge bool) {
	for index := len(completed) - 1; index >= 0; index-- {
		if merge {
			_ = completed[index].Merge()
		} else {
			_ = completed[index].Unmerge()
		}
	}
}

// MarshalState menyandikan parameter adapter dan status merge.
func (a *Adapter) MarshalState() ([]byte, error) {
	if a == nil || a.ModuleBase == nil {
		return nil, errs.ErrInvalidArgument
	}
	state, err := nn.StateDictOf(a)
	if err != nil {
		return nil, err
	}
	defer state.Close()
	merged := int64(0)
	if len(a.Wrappers()) > 0 && a.Wrappers()[0].IsMerged() {
		merged = 1
	}
	flag, err := tensor.NewTensor(dtype.Int64)
	if err != nil {
		return nil, err
	}
	if err := tensor.Write(flag, []int64{merged}); err != nil {
		_ = flag.Close()
		return nil, err
	}
	err = state.Put(mergedStateKey, flag)
	_ = flag.Close()
	if err != nil {
		return nil, err
	}
	return state.MarshalBinary()
}

// LoadState memulihkan parameter adapter dan status merge tanpa menyentuh base.
func (a *Adapter) LoadState(data []byte) error {
	if a == nil || a.ModuleBase == nil {
		return errs.ErrInvalidArgument
	}
	decoded := nn.NewStateDict()
	defer decoded.Close()
	if err := decoded.UnmarshalBinary(data); err != nil {
		return err
	}
	flag, err := decoded.Get(mergedStateKey)
	if err != nil {
		return fmt.Errorf("%w: missing adapter merge flag", err)
	}
	values, err := tensor.Read[int64](flag)
	_ = flag.Close()
	if err != nil || len(values) != 1 || (values[0] != 0 && values[0] != 1) {
		return fmt.Errorf("%w: invalid adapter merge flag", errs.ErrInvalidSerialization)
	}
	filtered := nn.NewStateDict()
	defer filtered.Close()
	for _, name := range decoded.Names() {
		if name == mergedStateKey {
			continue
		}
		value, err := decoded.Get(name)
		if err != nil {
			return err
		}
		err = filtered.Put(name, value)
		_ = value.Close()
		if err != nil {
			return err
		}
	}
	if err := nn.LoadStateDict(a, filtered); err != nil {
		return err
	}
	for _, wrapper := range a.Wrappers() {
		wrapper.mu.Lock()
		wrapper.merged = values[0] == 1
		wrapper.mu.Unlock()
	}
	return nil
}
