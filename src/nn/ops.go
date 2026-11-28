package nn

import (
	"errors"
	"fmt"
	"math"

	"github.com/adhitiad/gotorch/src/autograd"
	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/tensor"
)

// Nama operasi diferensial yang dipakai layer P5.
const (
	OpLinear           = "nn.linear"
	OpLinearNoBias     = "nn.linear.no_bias"
	OpMatMul           = "nn.matmul"
	OpBiasAdd          = "nn.bias_add"
	OpScale            = "nn.scale"
	OpScaleLast        = "nn.scale_last"
	OpConcatTime       = "nn.concat_time"
	OpConcatTimePrefix = "nn.concat_time_prefix"
	OpReshape          = "nn.reshape"
	OpTranspose        = "nn.transpose"
	OpReLU             = "nn.relu"
	OpSigmoid          = "nn.sigmoid"
	OpTanh             = "nn.tanh"
	OpGELU             = "nn.gelu"
	OpSiLU             = "nn.silu"
	OpSoftmax          = "nn.softmax"
	OpLogSoftmax       = "nn.log_softmax"
	OpLayerNorm        = "nn.layer_norm"
	OpRMSNorm          = "nn.rms_norm"
	OpMSELossMean      = "nn.mse.mean"
	OpMSELossSum       = "nn.mse.sum"
	OpMSELossNone      = "nn.mse.none"
	OpCrossEntropy     = "nn.cross_entropy"
	OpConv2D           = "nn.conv2d"
	OpMaxPool2D        = "nn.max_pool2d"
	OpAvgPool2D        = "nn.avg_pool2d"
	OpEmbedding        = "nn.embedding"
	OpRoPE             = "nn.rope"
	OpAttention        = "nn.attention"
)

var nnOperations = []autograd.Operation{
	{Name: OpLinear, Backward: linearBackward},
	{Name: OpLinearNoBias, Backward: linearNoBiasBackward},
	{Name: OpMatMul, Backward: matMulBackward},
	{Name: OpBiasAdd, Backward: biasAddBackward},
	{Name: OpScale, Backward: scaleBackward},
	{Name: OpScaleLast, Backward: scaleLastBackward},
	{Name: OpConcatTime, Backward: concatTimeBackward},
	{Name: OpConcatTimePrefix, Backward: concatTimePrefixBackward},
	{Name: OpReshape, Backward: reshapeBackward},
	{Name: OpTranspose, Backward: transposeBackward},
	{Name: OpReLU, Backward: reluBackward},
	{Name: OpSigmoid, Backward: sigmoidBackward},
	{Name: OpTanh, Backward: tanhBackward},
	{Name: OpGELU, Backward: geluBackward},
	{Name: OpSiLU, Backward: siluBackward},
	{Name: OpSoftmax, Backward: softmaxBackward},
	{Name: OpLogSoftmax, Backward: logSoftmaxBackward},
	{Name: OpLayerNorm, Backward: layerNormBackward},
	{Name: OpRMSNorm, Backward: rmsNormBackward},
	{Name: OpMSELossMean, Backward: mseMeanBackward},
	{Name: OpMSELossSum, Backward: mseSumBackward},
	{Name: OpMSELossNone, Backward: mseNoneBackward},
	{Name: OpCrossEntropy, Backward: crossEntropyBackward},
	{Name: OpConv2D, Backward: conv2DBackward},
	{Name: OpMaxPool2D, Backward: maxPool2DBackward},
	{Name: OpAvgPool2D, Backward: avgPool2DBackward},
	{Name: OpEmbedding, Backward: embeddingBackward},
	{Name: OpRoPE, Backward: ropeBackward},
	{Name: OpAttention, Backward: attentionBackward},
}

// NewTape membuat autograd Tape dengan seluruh operasi diferensial P5 terdaftar.
func NewTape() (*autograd.Tape, error) {
	tape := autograd.NewTape()
	if err := RegisterOperations(tape); err != nil {
		return nil, err
	}
	return tape, nil
}

// RegisterOperations mendaftarkan operasi P5 ke tape. Pemanggilan ulang aman.
func RegisterOperations(tape *autograd.Tape) error {
	if tape == nil {
		return errs.ErrInvalidArgument
	}
	for _, operation := range nnOperations {
		err := tape.RegisterOperation(operation)
		if err != nil && !errors.Is(err, errs.ErrOperationAlreadyRegistered) {
			return err
		}
	}
	return nil
}

type tensorData struct {
	values []float64
	dtype  dtype.DType
	dims   []int64
}

func tensorResults(values ...tensorData) ([]*tensor.Tensor, error) {
	result := make([]*tensor.Tensor, len(values))
	for index, value := range values {
		if value.dtype == dtype.Invalid {
			result[index] = nil
			continue
		}
		converted, err := value.tensor()
		if err != nil {
			for _, item := range result {
				if item != nil {
					_ = item.Close()
				}
			}
			return nil, err
		}
		result[index] = converted
	}
	return result, nil
}

func dataOf(value *tensor.Tensor) (tensorData, error) {
	return readTensorData(value)
}

func readDataPair(gradOutput *tensor.Tensor, inputs []*tensor.Tensor) (tensorData, tensorData, error) {
	grad, err := dataOf(gradOutput)
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	input, err := dataOf(inputs[0])
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	return grad, input, nil
}

func readDataTriple(gradOutput *tensor.Tensor, inputs []*tensor.Tensor) (tensorData, tensorData, tensorData, error) {
	grad, input, err := readDataPair(gradOutput, inputs)
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	weight, err := dataOf(inputs[1])
	if err != nil {
		return tensorData{}, tensorData{}, tensorData{}, err
	}
	return grad, input, weight, nil
}

func readBackwardPair(gradOutput *tensor.Tensor, inputs []*tensor.Tensor) (tensorData, tensorData, error) {
	grad, input, err := readDataPair(gradOutput, inputs)
	if err != nil {
		return tensorData{}, tensorData{}, err
	}
	if err := requireSameDims(grad, input); err != nil {
		return tensorData{}, tensorData{}, err
	}
	return grad, input, nil
}

func readTensorData(value *tensor.Tensor) (tensorData, error) {
	if value == nil {
		return tensorData{}, errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return tensorData{}, err
	}
	values, err := readValues(value)
	if err != nil {
		return tensorData{}, err
	}
	return tensorData{values: values, dtype: metadata.DType(), dims: cloneDims(metadata.Shape().Dims())}, nil
}

func (d tensorData) tensor() (*tensor.Tensor, error) {
	value, err := tensor.NewTensor(d.dtype, d.dims...)
	if err != nil {
		return nil, err
	}
	if err := writeValues(value, d.values); err != nil {
		_ = value.Close()
		return nil, err
	}
	return value, nil
}

func (d tensorData) withValues(values []float64) tensorData {
	return tensorData{values: values, dtype: d.dtype, dims: cloneDims(d.dims)}
}

func floatSlice(values []float64) []float64 {
	return append([]float64(nil), values...)
}

func cloneDims(dims []int64) []int64 {
	return append([]int64(nil), dims...)
}

func dimsEqual(left, right []int64) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func elementCount(dims []int64) (int, error) {
	count := int64(1)
	for _, dim := range dims {
		if dim < 0 {
			return 0, errs.ErrInvalidShape
		}
		if dim != 0 && count > math.MaxInt64/dim {
			return 0, errs.ErrShapeOverflow
		}
		count *= dim
	}
	if count > int64(math.MaxInt) {
		return 0, errs.ErrShapeOverflow
	}
	return int(count), nil
}

func rowColumns(dims []int64) (int, int, error) {
	if len(dims) == 0 {
		return 1, 1, nil
	}
	columns := dims[len(dims)-1]
	if columns < 0 {
		return 0, 0, errs.ErrInvalidShape
	}
	rows, err := elementCount(dims[:len(dims)-1])
	if err != nil || columns > int64(math.MaxInt) {
		return 0, 0, errs.ErrShapeOverflow
	}
	if rows != 0 && columns != 0 && int64(math.MaxInt)/int64(rows) < columns {
		return 0, 0, errs.ErrShapeOverflow
	}
	return rows, int(columns), nil
}

func requireSameDims(values ...tensorData) error {
	if len(values) == 0 {
		return errs.ErrInvalidArgument
	}
	for _, value := range values[1:] {
		if !dimsEqual(values[0].dims, value.dims) {
			return errs.ErrGradientShapeMismatch
		}
		if values[0].dtype != value.dtype {
			return fmt.Errorf("%w: mixed dtypes %s and %s", errs.ErrUnsupportedDType, values[0].dtype, value.dtype)
		}
	}
	return nil
}

func requireRank(value tensorData, rank int) error {
	if len(value.dims) != rank {
		return fmt.Errorf("%w: expected rank %d, got %d", errs.ErrInvalidShape, rank, len(value.dims))
	}
	return nil
}

func intTensor(values ...int64) (*tensor.Tensor, error) {
	return tensor.From(values, int64(len(values)))
}

func readIntTensor(value *tensor.Tensor) ([]int64, error) {
	if value == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := value.Metadata()
	if err != nil {
		return nil, err
	}
	if metadata.DType() != dtype.Int64 {
		return nil, fmt.Errorf("%w: expected int64 control tensor, got %s", errs.ErrUnsupportedDType, metadata.DType())
	}
	return tensor.Read[int64](value)
}
