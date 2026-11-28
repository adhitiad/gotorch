// Package data menyediakan kontrak tokenizer untuk P6. Tokenizer mengubah teks
// menjadi ID token dan kembali secara deterministik, dengan dukungan special
// token dan pembatalan konteks.
package data

// Nama special token yang dikenal. Implementasi Tokenizer menentukan ID yang
// diberikan ke nama-nama ini melalui SpecialTokens().
const (
	// BOSName menandai awal sequence.
	BOSName = "[BOS]"
	// EOSName menandai akhir sequence.
	EOSName = "[EOS]"
	// UNKName mengganti token yang tidak ada di kosakata.
	UNKName = "[UNK]"
	// PADName melakukan padding batch ke panjang seragam.
	PADName = "[PAD]"
	// CLSName menandai token klasifikasi.
	CLSName = "[CLS]"
	// SEPName memisahkan dua segmen.
	SEPName = "[SEP]"
	// MASKName menutup token untuk masked language modeling.
	MASKName = "[MASK]"
)

// DefaultSpecialIDs mengembalikan salinan baru dari pemetaan ID khusus yang
// stabil untuk special token yang dikenal. Implementasi boleh menggunakan ID
// berbeda asalkan didokumentasikan dan konsisten antara Encode dan Decode.
func DefaultSpecialIDs() map[string]int64 {
	return map[string]int64{
		BOSName:  0,
		EOSName:  1,
		UNKName:  2,
		PADName:  3,
		CLSName:  4,
		SEPName:  5,
		MASKName: 6,
	}
}
