package optim

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"

	"github.com/adhitiad/gotorch/src/errs"
)

var optimizerStateMagic = [8]byte{'G', 'O', 'T', 'O', 'P', 'T', 'S', 'T'}

const optimizerStateVersion uint32 = 1

type stateEntry struct {
	scalars []float64
	vectors [][]float64
}

type optimizerState map[string]stateEntry

func encodeState(entries optimizerState) ([]byte, error) {
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	buffer := make([]byte, 0, 128)
	buffer = append(buffer, optimizerStateMagic[:]...)
	buffer = appendUint32(buffer, optimizerStateVersion)
	buffer = appendUint32(buffer, uint32(len(names)))
	for _, name := range names {
		entry := entries[name]
		if len(name) > math.MaxUint32 || len(entry.vectors) > math.MaxUint32 || len(entry.scalars) > math.MaxUint32 {
			return nil, errs.ErrShapeOverflow
		}
		buffer = appendUint32(buffer, uint32(len(name)))
		buffer = append(buffer, name...)
		buffer = appendUint64(buffer, uint64(len(entry.scalars)))
		for _, value := range entry.scalars {
			buffer = appendUint64(buffer, math.Float64bits(value))
		}
		buffer = appendUint32(buffer, uint32(len(entry.vectors)))
		for _, vector := range entry.vectors {
			if uint64(len(vector)) > uint64(math.MaxUint32) {
				return nil, errs.ErrShapeOverflow
			}
			buffer = appendUint32(buffer, uint32(len(vector)))
			for _, value := range vector {
				buffer = appendUint64(buffer, math.Float64bits(value))
			}
		}
	}
	return buffer, nil
}

func decodeState(data []byte) (optimizerState, error) {
	if len(data) < len(optimizerStateMagic)+8 {
		return nil, errs.ErrInvalidArgument
	}
	for index, item := range optimizerStateMagic {
		if data[index] != item {
			return nil, fmt.Errorf("%w: invalid optimizer state magic", errs.ErrInvalidSerialization)
		}
	}
	version := binary.LittleEndian.Uint32(data[len(optimizerStateMagic):])
	if version != optimizerStateVersion {
		return nil, fmt.Errorf("%w: optimizer state version %d", errs.ErrUnsupportedSerializationVersion, version)
	}
	count := int(binary.LittleEndian.Uint32(data[len(optimizerStateMagic)+4:]))
	offset := len(optimizerStateMagic) + 8
	// Setiap entri minimal memuat panjang nama, satu byte nama, panjang skalar,
	// dan jumlah vektor. Batasi alokasi sebelum mempercayai header.
	if uint64(count) > uint64(len(data)-offset)/17 {
		return nil, errs.ErrInvalidSerialization
	}
	entries := make(optimizerState, count)
	for item := 0; item < count; item++ {
		name, next, err := decodeName(data, offset)
		if err != nil {
			return nil, err
		}
		offset = next
		scalars, next, err := decodeVector(data, offset)
		if err != nil {
			return nil, err
		}
		offset = next
		if offset+4 > len(data) {
			return nil, errs.ErrInvalidSerialization
		}
		vectorCount := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
		if uint64(vectorCount) > uint64(len(data)-offset)/4 {
			return nil, errs.ErrInvalidSerialization
		}
		vectors := make([][]float64, 0, vectorCount)
		for index := 0; index < vectorCount; index++ {
			vector, next, err := decodeFloatVector(data, offset)
			if err != nil {
				return nil, err
			}
			offset = next
			vectors = append(vectors, vector)
		}
		if _, exists := entries[name]; exists {
			return nil, fmt.Errorf("%w: duplicate optimizer state %q", errs.ErrInvalidSerialization, name)
		}
		entries[name] = stateEntry{scalars: scalars, vectors: vectors}
	}
	if offset != len(data) {
		return nil, errs.ErrInvalidSerialization
	}
	return entries, nil
}

func decodeName(data []byte, offset int) (string, int, error) {
	if offset+4 > len(data) {
		return "", 0, errs.ErrInvalidSerialization
	}
	length := int(binary.LittleEndian.Uint32(data[offset:]))
	offset += 4
	if length < 1 || length > len(data)-offset {
		return "", 0, errs.ErrInvalidSerialization
	}
	name := string(data[offset : offset+length])
	return name, offset + length, nil
}

func decodeVector(data []byte, offset int) ([]float64, int, error) {
	if offset+8 > len(data) {
		return nil, 0, errs.ErrInvalidSerialization
	}
	count := int(binary.LittleEndian.Uint64(data[offset:]))
	offset += 8
	if uint64(count) > uint64(len(data)-offset)/8 {
		return nil, 0, errs.ErrInvalidSerialization
	}
	values := make([]float64, count)
	for index := range values {
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(data[offset:]))
		offset += 8
	}
	return values, offset, nil
}

// decodeFloatVector membaca vektor dengan panjang uint32, sesuai penulisan
// pada encodeState. decodeVector memakai panjang uint64 untuk daftar skalar.
func decodeFloatVector(data []byte, offset int) ([]float64, int, error) {
	if offset+4 > len(data) {
		return nil, 0, errs.ErrInvalidSerialization
	}
	count := int(binary.LittleEndian.Uint32(data[offset:]))
	offset += 4
	if uint64(count) > uint64(len(data)-offset)/8 {
		return nil, 0, errs.ErrInvalidSerialization
	}
	values := make([]float64, count)
	for index := range values {
		values[index] = math.Float64frombits(binary.LittleEndian.Uint64(data[offset:]))
		offset += 8
	}
	return values, offset, nil
}

func appendUint32(buffer []byte, value uint32) []byte {
	var encoded [4]byte
	binary.LittleEndian.PutUint32(encoded[:], value)
	return append(buffer, encoded[:]...)
}

func appendUint64(buffer []byte, value uint64) []byte {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], value)
	return append(buffer, encoded[:]...)
}

func validateStateEntry(name string, entry stateEntry, scalarCount int, vectorSizes ...int) error {
	if len(entry.scalars) != scalarCount || len(entry.vectors) != len(vectorSizes) {
		return fmt.Errorf("%w: optimizer state %q has incompatible layout", errs.ErrInvalidState, name)
	}
	for index, size := range vectorSizes {
		if len(entry.vectors[index]) != size {
			return fmt.Errorf("%w: optimizer state %q vector %d has %d values, want %d", errs.ErrInvalidState, name, index, len(entry.vectors[index]), size)
		}
	}
	return nil
}

func zeroVectors(sizes ...int) [][]float64 {
	result := make([][]float64, len(sizes))
	for index, size := range sizes {
		result[index] = make([]float64, size)
	}
	return result
}
