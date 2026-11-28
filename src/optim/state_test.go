package optim

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func stateHeader(count uint32) []byte {
	data := append([]byte(nil), optimizerStateMagic[:]...)
	data = appendUint32(data, optimizerStateVersion)
	return appendUint32(data, count)
}

func TestDecodeStateRejectsHostileEntryCount(t *testing.T) {
	data := stateHeader(^uint32(0))
	if _, err := decodeState(data); !errors.Is(err, errs.ErrInvalidSerialization) {
		t.Fatalf("error = %v, diharapkan ErrInvalidSerialization", err)
	}
}

func TestDecodeStateRejectsHostileVectorCount(t *testing.T) {
	data := stateHeader(1)
	data = appendUint32(data, 1)
	data = append(data, 'p')
	data = appendUint64(data, 0)
	data = appendUint32(data, ^uint32(0))
	if _, err := decodeState(data); !errors.Is(err, errs.ErrInvalidSerialization) {
		t.Fatalf("error = %v, diharapkan ErrInvalidSerialization", err)
	}
}
