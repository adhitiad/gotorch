# Autograd API (P2)

This document describes the first Go-native automatic-differentiation engine in GoTorch. It is a small, explicit API intended to establish graph, gradient, ownership and numerical contracts before neural-network layers and broad CPU kernels are introduced.

## Scope

P2 provides:

- An append-only reverse-mode graph (Tape) and immutable Variable handles.
- requiresGrad state for input and derived variables.
- Reverse-mode vector-Jacobian products, gradient accumulation and ZeroGrad.
- Built-in differentiable Add, Mul and all-elements Sum for float32/float64.
- Per-Tape custom operation registration with backward and optional forward-mode rules.
- Forward-mode Jacobian-vector products (JVP).
- A central finite-difference CheckGradient helper for scalar-valued functions.

This is not yet a neural-network/training API. Tensor broadcasting, matrix multiplication, convolution, complex/low-precision gradients, higher-order differentiation, graph compilation and parallel backward scheduling are not part of P2. The built-in arithmetic is a small Pure Go reference path; P3 owns the general CPU kernel registry, dispatch and optimization strategy.

## Create variables and record operations

    input, err := tensor.From([]float64{1, 2, 3}, 3)
    if err != nil {
        return err
    }
    defer input.Close()

    tape := autograd.NewTape()
    defer tape.Close()

    x, err := tape.NewVariable(input, true)
    if err != nil {
        return err
    }
    squared, err := tape.Mul(x, x)
    if err != nil {
        return err
    }
    shifted, err := tape.Add(squared, x)
    if err != nil {
        return err
    }
    loss, err := tape.Sum(shifted)
    if err != nil {
        return err
    }
    if err := tape.Backward(context.Background(), loss); err != nil {
        return err
    }

    gradient, err := x.Grad()
    if err != nil {
        return err
    }
    defer gradient.Close()
    values, err := tensor.Read[float64](gradient) // [3, 5, 7]

The initial elementwise Add and Mul require exactly matching shapes and dtypes. Sum reduces every element to a rank-zero scalar. All three operations support float32 and float64 only. There is no implicit dtype conversion or broadcasting.

## Tensor and graph ownership

- Tape.NewVariable deep-copies its input. Changing or closing the caller's original Tensor cannot change the value saved by the graph.
- Tape.Apply deep-copies the operation output and stores parent node IDs. Parents must belong to the same Tape and must already exist; append order is topological, so graph cycles cannot be introduced through the public API.
- Variable.Value and Variable.Grad return independent Tensor copies. The caller owns and must close those copies.
- Tape owns all graph values and committed gradient buffers. Tape.Close releases them and invalidates its Variables. Close is idempotent.
- A Tape cannot be closed, mutated, or zeroed while Backward or JVP is running. Graph execution is serialized per Tape; separate Tapes can execute independently.
- Every differentiable node retains its forward value until Tape.Close. This deliberately favors predictable first-version lifetime semantics over activation-memory optimization.

## Reverse-mode gradients

Backward(ctx, target) seeds target with ones, then applies registered vector-Jacobian products in reverse topological order. This makes it convenient for scalar losses while remaining defined for non-scalar targets.

BackwardWithGradient(ctx, target, seed) accepts an explicit upstream Tensor with exactly the target's shape and dtype.

Gradients accumulate by default across repeated Backward calls. ZeroGrad() clears all gradients; ZeroGrad(x, y) clears only the selected Variables. Intermediate and leaf Variables retain their gradients until cleared or the Tape is closed. A Variable without a committed gradient returns errs.ErrNoGradient.

A successful Backward commits gradients atomically at the Tape level: cancellation or a VJP/shape error leaves previously committed gradients unchanged. The engine validates the shape and dtype of each VJP result before accumulation.

Only float32 and float64 are differentiable in P2. Bool, integer, Float16Bits, BFloat16Bits and complex tensors may still exist in P1 but cannot set requiresGrad.

## Custom operations

Register operations per Tape; there is no global mutable registry:

    err := tape.RegisterOperation(autograd.Operation{
        Name: "scale_two",
        Backward: func(
            ctx context.Context,
            gradOutput *tensor.Tensor,
            inputs []*tensor.Tensor,
            output *tensor.Tensor,
        ) ([]*tensor.Tensor, error) {
            // Return one newly allocated gradient Tensor per input.
            // Return nil only for an input that is not differentiable.
            return []*tensor.Tensor{scaledGradient}, nil
        },
        Forward: func(
            ctx context.Context,
            inputs []*tensor.Tensor,
            tangents []*tensor.Tensor,
            output *tensor.Tensor,
        ) (*tensor.Tensor, error) {
            // tangents[i] is nil when that input has no tangent.
            return outputTangent, nil
        },
    })

Then compute the forward result with the caller's operation and record it:

    result, err := tape.Apply("scale_two", []*autograd.Variable{x}, forwardResult)

Backward callbacks receive independently owned snapshots of gradOutput, input values and output. Treat them as read-only and do not close them. They must return a slice with one entry per input; each non-nil returned Tensor transfers ownership to the engine. The engine checks shape/dtype, accumulates and closes returned gradients. Callback errors are wrapped with the operation name.

The backward rule is mandatory when registering a custom operation. Forward-mode support is optional; requesting JVP through a node without a forward rule returns errs.ErrUnsupportedOperation. Do not retain callback snapshots after the callback returns.

## Forward-mode JVP

Tape.JVP(ctx, target, seeds) propagates tangent tensors in graph order and returns an independently owned tangent for target. The seed map may contain leaf or intermediate Variables from the same Tape. Unseeded values contribute zero tangent. If target has no incoming tangent, JVP returns a zero Tensor with target's shape/dtype.

Each operation reached by a non-zero tangent must have a registered Forward rule. The built-in Add, Mul and Sum operations have both VJP and JVP rules.

## Finite-difference gradient checking

autograd.CheckGradient compares reverse-mode gradients to central finite differences for a function that returns a rank-zero scalar:

- Defaults are dtype-aware: float32 uses a step of 1e-3; float64 uses 1e-6.
- Absolute/relative tolerances can be supplied in GradientCheckOptions.
- The default maximum is 256 input elements to bound the number of function evaluations.
- The result includes per-element analytical/numerical values and maximum errors. A numerical mismatch sets Passed=false; invalid inputs and execution errors return an error.

Finite differences are a development/test tool, not a production training primitive. Non-smooth functions and poorly scaled objectives may need explicit step/tolerance settings.

## Concurrency and cancellation

Tensor snapshots are isolated from external mutation. Each Tape protects graph structure and committed gradient state. Backward and JVP execute one at a time per Tape; graph mutation, ZeroGrad and Close return errs.ErrInvalidState while an execution is active. Custom callbacks should honor ctx cancellation and must not call graph mutation methods on the same Tape. Separate Tapes have independent locks and may be used concurrently.

## P2 limitations and next phase

P2 deliberately does not claim broadcast-gradient reductions, matrix multiplication, convolution, batched/parallel backward, checkpointing, graph pruning, higher-order derivatives or low-precision gradients. Add/Mul/Sum are correctness primitives only. P3 will define the general CPU operation registry and Pure Go reference kernels, then optional SIMD/BLAS paths.
