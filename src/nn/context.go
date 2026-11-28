package nn

import (
	"sync"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Context mengikat satu Tape dengan variabel parameter yang dipakai pada langkah
// forward/backward tersebut. Satu Context dipakai untuk satu siklus pelatihan.
type Context struct {
	Tape      *autograd.Tape
	mu        sync.Mutex
	variables map[*Parameter]*autograd.Variable
}

// NewContext membuat Context untuk tape yang diberikan.
func NewContext(tape *autograd.Tape) (*Context, error) {
	if tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	return &Context{Tape: tape, variables: make(map[*Parameter]*autograd.Variable)}, nil
}

// Variable mengembalikan variabel autograd untuk parameter dan membuatnya sekali
// per Context. Parameter dengan requiresGrad=false tetap dibuat sebagai leaf.
func (c *Context) Variable(parameter *Parameter) (*autograd.Variable, error) {
	if c == nil || c.Tape == nil || parameter == nil {
		return nil, errs.ErrInvalidArgument
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if variable := c.variables[parameter]; variable != nil {
		return variable, nil
	}
	value := parameter.Value()
	if value == nil {
		return nil, errs.ErrInvalidState
	}
	variable, err := c.Tape.NewVariable(value, parameter.RequiresGrad())
	if err != nil {
		return nil, err
	}
	c.variables[parameter] = variable
	return variable, nil
}

// Gradient mengembalikan salinan gradien parameter pada Context ini.
func (c *Context) Gradient(parameter *Parameter) (*tensor.Tensor, error) {
	variable, err := c.Variable(parameter)
	if err != nil {
		return nil, err
	}
	return variable.Grad()
}

// HasGradient melaporkan apakah gradien parameter sudah tersedia.
func (c *Context) HasGradient(parameter *Parameter) bool {
	gradient, err := c.Gradient(parameter)
	if err != nil {
		return false
	}
	return gradient.Close() == nil
}

// Parameters mengembalikan parameter yang sudah terikat pada Context.
func (c *Context) Parameters() []*Parameter {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([]*Parameter, 0, len(c.variables))
	for parameter := range c.variables {
		result = append(result, parameter)
	}
	return result
}
