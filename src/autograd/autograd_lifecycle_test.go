package autograd

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func TestBackwardRejectsInvalidCustomGradient(t *testing.T) {
	tape := NewTape()
	defer tape.Close()
	if err := tape.RegisterOperation(Operation{
		Name: "bad_vjp",
		Backward: func(_ context.Context, _ *tensor.Tensor, _ []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
			return []*tensor.Tensor{tensorMustFloat32(t, []float32{1}, 1)}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.From([]float32{1, 2}, 2)
	defer input.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	output, _ := tensor.From([]float32{1}, 1)
	defer output.Close()
	y, err := tape.Apply("bad_vjp", []*Variable{x}, output)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(context.Background(), y); !errors.Is(err, errs.ErrGradientShapeMismatch) {
		t.Fatalf("Backward with malformed custom gradient = %v", err)
	}
	if _, err := x.Grad(); !errors.Is(err, errs.ErrNoGradient) {
		t.Fatalf("gradient committed after malformed VJP = %v", err)
	}
}

func TestJVPRequiresRegisteredForwardRule(t *testing.T) {
	tape := NewTape()
	defer tape.Close()
	if err := tape.RegisterOperation(Operation{
		Name: "backward_only",
		Backward: func(_ context.Context, upstream *tensor.Tensor, _ []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
			copyValue, err := upstream.Copy()
			return []*tensor.Tensor{copyValue}, err
		},
	}); err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.From([]float32{2}, 1)
	defer input.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	output, _ := tensor.From([]float32{2}, 1)
	defer output.Close()
	y, err := tape.Apply("backward_only", []*Variable{x}, output)
	if err != nil {
		t.Fatal(err)
	}
	seed, _ := tensor.From([]float32{1}, 1)
	defer seed.Close()
	if _, err := tape.JVP(context.Background(), y, map[*Variable]*tensor.Tensor{x: seed}); !errors.Is(err, errs.ErrUnsupportedOperation) {
		t.Fatalf("JVP without rule = %v", err)
	}
}

func TestTapeRejectsMutationAndCloseDuringBackward(t *testing.T) {
	tape := NewTape()
	defer tape.Close()
	entered := make(chan struct{})
	release := make(chan struct{})
	if err := tape.RegisterOperation(Operation{
		Name: "blocking",
		Backward: func(_ context.Context, upstream *tensor.Tensor, _ []*tensor.Tensor, _ *tensor.Tensor) ([]*tensor.Tensor, error) {
			close(entered)
			<-release
			copyValue, err := upstream.Copy()
			return []*tensor.Tensor{copyValue}, err
		},
	}); err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.From([]float32{3}, 1)
	defer input.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	output, _ := tensor.From([]float32{3}, 1)
	defer output.Close()
	y, err := tape.Apply("blocking", []*Variable{x}, output)
	if err != nil {
		t.Fatal(err)
	}

	done := make(chan error, 1)
	go func() { done <- tape.Backward(context.Background(), y) }()
	<-entered
	if err := tape.Close(); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("Tape.Close during backward = %v", err)
	}
	if _, err := tape.NewVariable(input, true); !errors.Is(err, errs.ErrInvalidState) {
		t.Fatalf("NewVariable during backward = %v", err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestElementwiseOperationsRejectBroadcastShapes(t *testing.T) {
	inputA, _ := tensor.From([]float32{1, 2}, 2)
	defer inputA.Close()
	inputB, _ := tensor.From([]float32{1, 2, 3, 4}, 2, 2)
	defer inputB.Close()
	tape := NewTape()
	defer tape.Close()
	a, err := tape.NewVariable(inputA, true)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tape.NewVariable(inputB, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tape.Add(a, b); !errors.Is(err, errs.ErrGradientShapeMismatch) {
		t.Fatalf("Add shape mismatch = %v", err)
	}
}

func tensorMustFloat32(t *testing.T, values []float32, dimensions ...int64) *tensor.Tensor {
	t.Helper()
	value, err := tensor.From(values, dimensions...)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

type typedNilContext struct{}

func (*typedNilContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (*typedNilContext) Done() <-chan struct{}       { return nil }
func (*typedNilContext) Err() error                  { return nil }
func (*typedNilContext) Value(any) any               { return nil }

func TestJVPRejectsTypedNilContext(t *testing.T) {
	tape := NewTape()
	defer tape.Close()
	input, err := tensor.From([]float32{1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	x, err := tape.NewVariable(input, true)
	if err != nil {
		t.Fatal(err)
	}
	var typedNil *typedNilContext
	var ctx context.Context = typedNil
	if _, err := tape.JVP(ctx, x, nil); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("JVP typed-nil context error = %v, want ErrInvalidArgument", err)
	}
	if err := tape.Backward(ctx, x); !errors.Is(err, errs.ErrInvalidArgument) {
		t.Fatalf("Backward typed-nil context error = %v, want ErrInvalidArgument", err)
	}
}
