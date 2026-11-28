package nn

import (
	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

func intControl(ctx *Context, values ...int64) (*autograd.Variable, error) {
	if ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	value, err := intTensor(values...)
	if err != nil {
		return nil, err
	}
	variable, err := ctx.Tape.NewVariable(value, false)
	_ = value.Close()
	return variable, err
}

func floatControl(ctx *Context, elementType dtype.DType, dims []int64, values ...float64) (*autograd.Variable, error) {
	if ctx == nil || ctx.Tape == nil {
		return nil, errs.ErrInvalidArgument
	}
	value, err := tensor.NewTensor(elementType, dims...)
	if err != nil {
		return nil, err
	}
	if err := writeValues(value, values); err != nil {
		_ = value.Close()
		return nil, err
	}
	variable, err := ctx.Tape.NewVariable(value, false)
	_ = value.Close()
	return variable, err
}

// Reshape mengubah bentuk tensor tanpa mengubah urutan data.
func Reshape(ctx *Context, input *autograd.Variable, dims ...int64) (*autograd.Variable, error) {
	control, err := intControl(ctx, dims...)
	if err != nil {
		return nil, err
	}
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpReshape, []*autograd.Variable{input, control}, func() (tensorData, error) {
		return reshapeData(data, dims)
	})
}

// Transpose memutar sumbu tensor menurut permutasi axes yang eksplisit.
func Transpose(ctx *Context, input *autograd.Variable, axes ...int64) (*autograd.Variable, error) {
	control, err := intControl(ctx, axes...)
	if err != nil {
		return nil, err
	}
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpTranspose, []*autograd.Variable{input, control}, func() (tensorData, error) {
		return transposeData(data, axes)
	})
}

// MatMul mengalikan dua matriks rank-2.
func MatMul(ctx *Context, left, right *autograd.Variable) (*autograd.Variable, error) {
	leftData, err := variableData(left)
	if err != nil {
		return nil, err
	}
	rightData, err := variableData(right)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpMatMul, []*autograd.Variable{left, right}, func() (tensorData, error) {
		return matMulData(leftData, rightData)
	})
}

// BiasAdd menambahkan bias rank-1 pada setiap baris matriks.
func BiasAdd(ctx *Context, input, bias *autograd.Variable) (*autograd.Variable, error) {
	inputData, err := variableData(input)
	if err != nil {
		return nil, err
	}
	biasData, err := variableData(bias)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpBiasAdd, []*autograd.Variable{input, bias}, func() (tensorData, error) {
		if err := requireRank(inputData, 2); err != nil {
			return tensorData{}, err
		}
		if err := requireRank(biasData, 1); err != nil {
			return tensorData{}, err
		}
		if biasData.dims[0] != inputData.dims[1] || biasData.dtype != inputData.dtype {
			return tensorData{}, errs.ErrGradientShapeMismatch
		}
		values := floatSlice(inputData.values)
		columns := int(biasData.dims[0])
		for row := 0; row < int(inputData.dims[0]); row++ {
			for column := 0; column < columns; column++ {
				values[row*columns+column] += biasData.values[column]
			}
		}
		return inputData.withValues(values), nil
	})
}

// Scale mengalikan setiap elemen dengan skalar konstan.
func Scale(ctx *Context, input *autograd.Variable, scale float64) (*autograd.Variable, error) {
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	control, err := floatControl(ctx, data.dtype, nil, scale)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpScale, []*autograd.Variable{input, control}, func() (tensorData, error) {
		return scaleData(data, scale), nil
	})
}

// ScaleLast mengalikan kolom terakhir dengan vektor skala yang diberikan.
func ScaleLast(ctx *Context, input, scale *autograd.Variable) (*autograd.Variable, error) {
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	scaleDataValue, err := variableData(scale)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpScaleLast, []*autograd.Variable{input, scale}, func() (tensorData, error) {
		return scaleLastData(data, scaleDataValue)
	})
}

// Softmax menghitung probabilitas softmax pada sumbu terakhir secara stabil.
func Softmax(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpSoftmax, []*autograd.Variable{input}, func() (tensorData, error) {
		return softmaxForward(data)
	})
}

// LogSoftmax menghitung log probabilitas softmax pada sumbu terakhir secara stabil.
func LogSoftmax(ctx *Context, input *autograd.Variable) (*autograd.Variable, error) {
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpLogSoftmax, []*autograd.Variable{input}, func() (tensorData, error) {
		return logSoftmaxForward(data)
	})
}

// ConcatTimePrefix menambahkan prompt [time, dim] di awal urutan [batch, time, dim].
func ConcatTimePrefix(ctx *Context, prompt, sequence *autograd.Variable) (*autograd.Variable, error) {
	promptData, err := variableData(prompt)
	if err != nil {
		return nil, err
	}
	sequenceData, err := variableData(sequence)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpConcatTimePrefix, []*autograd.Variable{sequence, prompt}, func() (tensorData, error) {
		return concatTimePrefixData(sequenceData, promptData)
	})
}

// ConcatTime menggabungkan prefix [batch, time, dim] dengan prompt [time, dim].
func ConcatTime(ctx *Context, prefix, prompt *autograd.Variable) (*autograd.Variable, error) {
	prefixData, err := variableData(prefix)
	if err != nil {
		return nil, err
	}
	promptData, err := variableData(prompt)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpConcatTime, []*autograd.Variable{prefix, prompt}, func() (tensorData, error) {
		return concatTimeData(prefixData, promptData)
	})
}

// Embedding mengambil baris weight berdasarkan indeks int64.
func Embedding(ctx *Context, weight, indices *autograd.Variable) (*autograd.Variable, error) {
	weightData, err := variableData(weight)
	if err != nil {
		return nil, err
	}
	values, err := variableIntData(indices)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpEmbedding, []*autograd.Variable{weight, indices}, func() (tensorData, error) {
		return embeddingData(weightData, values)
	})
}

// RoPE menerapkan rotary position embedding pada tensor [batch, heads, time, dim].
func RoPE(ctx *Context, input, cosine, sine *autograd.Variable) (*autograd.Variable, error) {
	data, err := variableData(input)
	if err != nil {
		return nil, err
	}
	cosineData, err := variableData(cosine)
	if err != nil {
		return nil, err
	}
	sineData, err := variableData(sine)
	if err != nil {
		return nil, err
	}
	return applyOperation(ctx, OpRoPE, []*autograd.Variable{input, cosine, sine}, func() (tensorData, error) {
		return ropeData(data, cosineData, sineData)
	})
}

// Attention menghitung scaled dot-product attention dengan mask opsional.
func Attention(ctx *Context, query, key, value, mask *autograd.Variable) (*autograd.Variable, error) {
	queryData, err := variableData(query)
	if err != nil {
		return nil, err
	}
	keyData, err := variableData(key)
	if err != nil {
		return nil, err
	}
	valueData, err := variableData(value)
	if err != nil {
		return nil, err
	}
	maskData := tensorData{}
	if mask != nil {
		maskData, err = variableData(mask)
		if err != nil {
			return nil, err
		}
	}
	inputs := []*autograd.Variable{query, key, value}
	if mask != nil {
		inputs = append(inputs, mask)
	}
	return applyOperation(ctx, OpAttention, inputs, func() (tensorData, error) {
		return attentionData(queryData, keyData, valueData, maskData)
	})
}

func variableIntData(variable *autograd.Variable) ([]int64, error) {
	if variable == nil {
		return nil, errs.ErrInvalidArgument
	}
	value, err := variable.Value()
	if err != nil {
		return nil, err
	}
	defer value.Close()
	return readIntTensor(value)
}
