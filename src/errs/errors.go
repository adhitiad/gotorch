// Package errs defines stable sentinel errors shared by GoTorch contracts.
package errs

import "errors"

var (
	ErrInvalidArgument                 = errors.New("gotorch: invalid argument")
	ErrInvalidShape                    = errors.New("gotorch: invalid shape")
	ErrShapeOverflow                   = errors.New("gotorch: shape or storage size overflow")
	ErrUnsupportedDType                = errors.New("gotorch: unsupported dtype")
	ErrInvalidLayout                   = errors.New("gotorch: invalid layout")
	ErrOutOfBounds                     = errors.New("gotorch: storage range out of bounds")
	ErrStorageReleased                 = errors.New("gotorch: storage handle released")
	ErrInvalidState                    = errors.New("gotorch: invalid lifecycle state")
	ErrInvalidDistributedID            = errors.New("gotorch: invalid distributed identity")
	ErrUnsupportedOperation            = errors.New("gotorch: unsupported operation")
	ErrUnsupportedBackend              = errors.New("gotorch: unsupported backend")
	ErrOverlappingWrite                = errors.New("gotorch: bulk write to overlapping tensor view")
	ErrInvalidSerialization            = errors.New("gotorch: invalid tensor serialization")
	ErrUnsupportedSerializationVersion = errors.New("gotorch: unsupported tensor serialization version")
	ErrNoGradient                      = errors.New("gotorch: gradient is not available")
	ErrGradientShapeMismatch           = errors.New("gotorch: gradient shape or dtype mismatch")
	ErrOperationAlreadyRegistered      = errors.New("gotorch: autograd operation already registered")
)
