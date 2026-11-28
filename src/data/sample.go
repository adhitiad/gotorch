package data

import (
	"fmt"
	"sort"

	"github.com/adhitiad/gotorch/src/errs"
)

// Sample adalah satu rekaman dataset yang terdiri dari field bernama. Nilai nol
// merupakan sample kosong yang valid. NewSample menyalin peta field, tetapi nilai
// field dipinjam: pemanggil tidak boleh mengubahnya selama sample atau batch yang
// memuatnya masih dipakai. Nilai hanya dapat dibaca lewat Sample dan tidak pernah
// ditutup oleh Sample.
type Sample struct {
	fields map[string]any
}

// NewSample membentuk sample dari field bernama. Nama field tidak boleh kosong
// dan nilai tidak boleh nil. Peta masukan disalin sehingga perubahan berikutnya
// tidak memengaruhi sample.
func NewSample(fields map[string]any) (Sample, error) {
	copied := make(map[string]any, len(fields))
	for name, value := range fields {
		if name == "" || value == nil {
			return Sample{}, fmt.Errorf("%w: sample field %q must have a name and value", errs.ErrInvalidArgument, name)
		}
		copied[name] = value
	}
	return Sample{fields: copied}, nil
}

// Len mengembalikan jumlah field.
func (s Sample) Len() int {
	return len(s.fields)
}

// Names mengembalikan nama field dalam urutan leksikografis. Slice yang
// dikembalikan adalah salinan baru.
func (s Sample) Names() []string {
	names := make([]string, 0, len(s.fields))
	for name := range s.fields {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Value mengembalikan nilai field dan status keberadaannya. Nilai dipinjam dan
// tidak boleh diubah oleh pemanggil.
func (s Sample) Value(name string) (any, bool) {
	value, ok := s.fields[name]
	return value, ok
}

// Clone mengembalikan salinan mandiri dari peta field. Nilai field tetap dipinjam
// dan dibagikan dengan sample asal.
func (s Sample) Clone() Sample {
	if len(s.fields) == 0 {
		return Sample{}
	}
	copied := make(map[string]any, len(s.fields))
	for name, value := range s.fields {
		copied[name] = value
	}
	return Sample{fields: copied}
}
