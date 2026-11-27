//go:build goexperiment.simd && amd64

package compute

import (
	"context"

	"simd/archsimd"
)

func simdBinaryFloat32(ctx context.Context, dst, a, b []float32, operation string) error {
	const lanes = 8
	i := 0
	for ; i+lanes <= len(dst); i += lanes {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		x, y := archsimd.LoadFloat32x8(a[i:]), archsimd.LoadFloat32x8(b[i:])
		var z archsimd.Float32x8
		switch operation {
		case OpAdd:
			z = x.Add(y)
		case OpSub:
			z = x.Sub(y)
		case OpMul:
			z = x.Mul(y)
		}
		z.Store(dst[i:])
	}
	for ; i < len(dst); i++ {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		switch operation {
		case OpAdd:
			dst[i] = a[i] + b[i]
		case OpSub:
			dst[i] = a[i] - b[i]
		case OpMul:
			dst[i] = a[i] * b[i]
		}
	}
	return ctx.Err()
}

func simdBinaryFloat64(ctx context.Context, dst, a, b []float64, operation string) error {
	const lanes = 4
	i := 0
	for ; i+lanes <= len(dst); i += lanes {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		x, y := archsimd.LoadFloat64x4(a[i:]), archsimd.LoadFloat64x4(b[i:])
		var z archsimd.Float64x4
		switch operation {
		case OpAdd:
			z = x.Add(y)
		case OpSub:
			z = x.Sub(y)
		case OpMul:
			z = x.Mul(y)
		}
		z.Store(dst[i:])
	}
	for ; i < len(dst); i++ {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		switch operation {
		case OpAdd:
			dst[i] = a[i] + b[i]
		case OpSub:
			dst[i] = a[i] - b[i]
		case OpMul:
			dst[i] = a[i] * b[i]
		}
	}
	return ctx.Err()
}
