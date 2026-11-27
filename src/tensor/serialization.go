package tensor

import (
	"encoding/binary"
	"fmt"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
)

var binaryMagic = [4]byte{'G', 'T', 'N', 'S'}

const (
	binaryFormatVersion = byte(1)
	binaryFixedHeader   = 16
)

// MarshalBinary implements encoding.BinaryMarshaler. The versioned format stores
// dtype, dimensions and canonical row-major logical bytes; it never serializes
// allocator internals, storage handles or view strides.
func (t *Tensor) MarshalBinary() ([]byte, error) {
	if t == nil {
		return nil, errs.ErrInvalidArgument
	}
	metadata, err := t.Metadata()
	if err != nil {
		return nil, err
	}
	dimensions := metadata.Shape().Dims()
	if len(dimensions) > int(^uint16(0)) {
		return nil, errs.ErrShapeOverflow
	}
	payload, err := t.readAll()
	if err != nil {
		return nil, err
	}
	if len(dimensions) > (int(^uint(0)>>1)-binaryFixedHeader)/8 {
		return nil, errs.ErrShapeOverflow
	}
	headerSize := binaryFixedHeader + len(dimensions)*8
	if len(payload) > int(^uint(0)>>1)-headerSize {
		return nil, errs.ErrShapeOverflow
	}
	result := make([]byte, headerSize+len(payload))
	copy(result[:4], binaryMagic[:])
	result[4] = binaryFormatVersion
	result[5] = byte(metadata.DType())
	binary.LittleEndian.PutUint16(result[6:8], uint16(len(dimensions)))
	for axis, dimension := range dimensions {
		binary.LittleEndian.PutUint64(result[8+axis*8:], uint64(dimension))
	}
	payloadLengthOffset := 8 + len(dimensions)*8
	binary.LittleEndian.PutUint64(result[payloadLengthOffset:payloadLengthOffset+8], uint64(len(payload)))
	copy(result[headerSize:], payload)
	return result, nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler. It accepts only a zero
// or closed Tensor receiver and leaves the receiver unchanged on any error.
// The decoded allocation is always newly owned and contiguous.
func (t *Tensor) UnmarshalBinary(data []byte) error {
	if t == nil {
		return errs.ErrInvalidArgument
	}
	t.mu.RLock()
	active := !t.closed && !nilDependency(t.storage)
	t.mu.RUnlock()
	if active {
		return fmt.Errorf("%w: UnmarshalBinary requires a zero or closed Tensor", errs.ErrInvalidState)
	}

	metadata, payload, err := decodeBinaryTensor(data)
	if err != nil {
		return err
	}
	created, err := NewTensor(metadata.DType(), metadata.Shape().Dims()...)
	if err != nil {
		return err
	}
	if err := created.writeAll(payload); err != nil {
		_ = created.Close()
		return err
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.closed && !nilDependency(t.storage) {
		_ = created.Close()
		return fmt.Errorf("%w: UnmarshalBinary requires a zero or closed Tensor", errs.ErrInvalidState)
	}
	t.metadata = created.metadata
	t.storage = created.storage
	t.offset = 0
	t.closed = false
	created.storage = nil
	created.closed = true
	return nil
}

func decodeBinaryTensor(data []byte) (Metadata, []byte, error) {
	if len(data) < binaryFixedHeader {
		return Metadata{}, nil, errs.ErrInvalidSerialization
	}
	if string(data[:4]) != string(binaryMagic[:]) {
		return Metadata{}, nil, errs.ErrInvalidSerialization
	}
	if data[4] != binaryFormatVersion {
		return Metadata{}, nil, errs.ErrUnsupportedSerializationVersion
	}
	elementType := dtype.DType(data[5])
	if !elementType.Valid() {
		return Metadata{}, nil, errs.ErrInvalidSerialization
	}
	rank := int(binary.LittleEndian.Uint16(data[6:8]))
	headerSize := binaryFixedHeader + rank*8
	if len(data) < headerSize {
		return Metadata{}, nil, errs.ErrInvalidSerialization
	}
	dimensions := make([]int64, rank)
	for axis := range dimensions {
		dimensions[axis] = int64(binary.LittleEndian.Uint64(data[8+axis*8:]))
	}
	dimensionsShape, err := shape.New(dimensions...)
	if err != nil {
		return Metadata{}, nil, fmt.Errorf("%w: invalid tensor shape: %v", errs.ErrInvalidSerialization, err)
	}
	metadata, err := NewMetadata(elementType, dimensionsShape)
	if err != nil {
		return Metadata{}, nil, fmt.Errorf("%w: invalid tensor metadata: %v", errs.ErrInvalidSerialization, err)
	}
	lengthOffset := 8 + rank*8
	declaredLength := binary.LittleEndian.Uint64(data[lengthOffset : lengthOffset+8])
	expectedLength, err := metadata.packedByteSize()
	if err != nil {
		return Metadata{}, nil, err
	}
	if uint64(expectedLength) != declaredLength || uint64(len(data)-headerSize) != declaredLength {
		return Metadata{}, nil, errs.ErrInvalidSerialization
	}
	payload := data[headerSize:]
	if elementType == dtype.Bool {
		for index, value := range payload {
			if value > 1 {
				return Metadata{}, nil, fmt.Errorf("%w: non-canonical bool at payload byte %d", errs.ErrInvalidSerialization, index)
			}
		}
	}
	return metadata, payload, nil
}
