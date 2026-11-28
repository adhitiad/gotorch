package data

import (
	"fmt"

	"github.com/adhitiad/gotorch/src/errs"
)

// Encoding adalah hasil immutable dari encode teks. ID, token, offset, dan
// masker special token semua memiliki panjang yang sama. Slice yang dikembalikan
// adalah salinan defensif dan tidak boleh diubah oleh pemanggil.
type Encoding struct {
	ids         []int64
	tokens      []string
	offsets     [][2]int64
	specialMask []bool
}

// NewEncoding membentuk Encoding. Semua slice harus memiliki panjang yang sama.
// Slice masukan disalin sehingga perubahan berikutnya tidak memengaruhi encoding.
func NewEncoding(ids []int64, tokens []string, offsets [][2]int64, specialMask []bool) (*Encoding, error) {
	length := len(ids)
	if len(tokens) != length || len(offsets) != length || len(specialMask) != length {
		return nil, fmt.Errorf("%w: encoding slices must have equal length: ids=%d tokens=%d offsets=%d mask=%d",
			errs.ErrInvalidArgument, length, len(tokens), len(offsets), len(specialMask))
	}
	return &Encoding{
		ids:         append([]int64(nil), ids...),
		tokens:      append([]string(nil), tokens...),
		offsets:     append([][2]int64(nil), offsets...),
		specialMask: append([]bool(nil), specialMask...),
	}, nil
}

// IDs mengembalikan ID token.
func (e *Encoding) IDs() []int64 {
	if e == nil {
		return nil
	}
	return append([]int64(nil), e.ids...)
}

// Tokens mengembalikan string token.
func (e *Encoding) Tokens() []string {
	if e == nil {
		return nil
	}
	return append([]string(nil), e.tokens...)
}

// Offsets mengembalikan offset karakter [start, end) per token.
func (e *Encoding) Offsets() [][2]int64 {
	if e == nil {
		return nil
	}
	return append([][2]int64(nil), e.offsets...)
}

// SpecialTokensMask mengembalikan true untuk token yang merupakan special token.
func (e *Encoding) SpecialTokensMask() []bool {
	if e == nil {
		return nil
	}
	return append([]bool(nil), e.specialMask...)
}

// Len mengembalikan jumlah token.
func (e *Encoding) Len() int {
	if e == nil {
		return 0
	}
	return len(e.ids)
}

// Clone mengembalikan salinan mandiri dari encoding.
func (e *Encoding) Clone() *Encoding {
	if e == nil {
		return nil
	}
	cloned, err := NewEncoding(e.ids, e.tokens, e.offsets, e.specialMask)
	if err != nil {
		// Tidak mungkin terjadi karena slice sudah divalidasi saat konstruksi.
		return nil
	}
	return cloned
}

// HasSpecial melaporkan apakah encoding mengandung minimal satu special token.
func (e *Encoding) HasSpecial() bool {
	if e == nil {
		return false
	}
	for _, special := range e.specialMask {
		if special {
			return true
		}
	}
	return false
}
