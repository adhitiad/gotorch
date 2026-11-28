// Package data mendefinisikan kontrak dataset, sample, sampler dan batch untuk P6.
// Implementasi pada tugas berikutnya membangun sumber streaming, transform dan
// loader di atas batas ini tanpa mengubah semantiknya.
package data

import (
	"context"
	"fmt"
	"reflect"

	"github.com/adhitiad/gotorch/src/errs"
)

// Dataset menyediakan akses acak ke kumpulan sampel berurut yang berhingga.
//
// Len bernilai non-negatif dan stabil selama masa hidup dataset. At menerima
// indeks dalam [0, Len), memeriksa ctx sebelum membaca, dan mengembalikan Sample
// yang peta field-nya terpisah dari dataset. Implementasi harus aman untuk
// panggilan At dan Len bersamaan dari worker loader yang terbatas. Close bersifat
// idempoten; At setelah Close mengembalikan error yang membungkus
// errs.ErrInvalidState. Pembatalan dikembalikan sebagai ctx.Err() dan tidak boleh
// meninggalkan keadaan parsial.
type Dataset interface {
	Len() int64
	At(ctx context.Context, index int64) (Sample, error)
	Close() error
}

// Sampler memetakan posisi epoch ke indeks dataset tanpa cursor tersembunyi yang
// dapat berubah. Len adalah jumlah posisi non-negatif yang dihasilkan dan boleh
// berbeda dari Dataset.Len. Index menerima posisi dalam [0, Len) dan mengembalikan
// indeks dataset dalam [0, datasetLen). Dengan konfigurasi dan seed yang sama,
// Index deterministik serta aman dipanggil bersamaan. Posisi tidak valid
// dibungkus errs.ErrOutOfBounds; pembatalan dikembalikan sebagai ctx.Err().
type Sampler interface {
	Len() int64
	Index(ctx context.Context, position int64) (int64, error)
}

// isNil mengenali nil typed yang tersimpan di dalam interface.
func isNil(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}

func checkContext(ctx context.Context) error {
	if isNil(ctx) {
		return fmt.Errorf("%w: nil context", errs.ErrInvalidArgument)
	}
	return ctx.Err()
}

func checkPosition(count, position int64, label string) error {
	if count < 0 {
		return fmt.Errorf("%w: negative %s count %d", errs.ErrInvalidState, label, count)
	}
	if position < 0 || position >= count {
		return fmt.Errorf("%w: %s %d outside [0,%d)", errs.ErrOutOfBounds, label, position, count)
	}
	return nil
}
