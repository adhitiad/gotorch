// Package main adalah titik masuk executable GoTorch.
// Menyediakan antarmuka baris perintah untuk mencoba fitur data engine
// (tokenizer, dataset, sampler, batcher, dan DataLoader) yang diimplementasikan
// pada fase P6.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/adhitiad/gotorch/src/data"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(2)
	}

	switch os.Args[1] {
	case "version":
		cmdVersion()
	case "tokenize":
		cmdTokenize(os.Args[2:])
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "perintah tidak diketahui: %q\n", os.Args[1])
		printUsage()
		os.Exit(2)
	}
}

func printUsage() {
	fmt.Println("GoTorch CLI — titik masuk executable untuk mencoba fitur data engine.")
	fmt.Println("")
	fmt.Println("Penggunaan:")
	fmt.Println("  go run ./cmd/gotorch <perintah> [argumen]")
	fmt.Println("")
	fmt.Println("Perintah:")
	fmt.Println("  version        Tampilkan versi dan informasi paket data.")
	fmt.Println("  tokenize       Encode dan decode teks dengan tokenizer yang dipilih.")
	fmt.Println("  help           Tampilkan bantuan ini.")
	fmt.Println("")
	fmt.Println("Contoh:")
	fmt.Println("  go run ./cmd/gotorch version")
	fmt.Println("  go run ./cmd/gotorch tokenize --tokenizer=bpe --text=\"hello world\" --vocab=200")
}

func cmdVersion() {
	fmt.Println("GoTorch data engine (P6)")
	fmt.Println("  Paket: github.com/adhitiad/gotorch/src/data")
	fmt.Println("  Kontrak: Dataset, Sample, Sampler, Batch, Batcher, DataLoader, Cache, Source")
	fmt.Println("  Tokenizer: ByteTokenizer, BPETokenizer, WordPieceTokenizer, UnigramTokenizer")
	fmt.Println("  Transform: TransformChain, RequireFields, CollateWithTransform")
}

func cmdTokenize(args []string) {
	fs := flag.NewFlagSet("tokenize", flag.ExitOnError)
	tokenizerName := fs.String("tokenizer", "byte", "jenis tokenizer: byte, bpe, wordpiece, unigram")
	text := fs.String("text", "", "teks yang akan di-encode")
	vocab := fs.Int("vocab", 200, "ukuran kosakata untuk tokenizer yang dilatih")
	_ = fs.Parse(args)

	if *text == "" {
		fmt.Fprintln(os.Stderr, "flag --text diperlukan")
		os.Exit(2)
	}

	ctx := context.Background()

	var tok data.Tokenizer
	switch strings.ToLower(*tokenizerName) {
	case "byte":
		tok = data.NewByteTokenizer()
	case "bpe":
		bpe := data.NewByteLevelBPETokenizer(*vocab, data.DefaultSpecialIDs())
		if err := bpe.Train([]string{*text}, *vocab); err != nil {
			fmt.Fprintf(os.Stderr, "Gagal melatih BPE: %v\n", err)
			os.Exit(1)
		}
		tok = bpe
	case "wordpiece":
		wpt := data.NewWordPieceTokenizer(false, data.DefaultSpecialIDs())
		if err := wpt.Train([]string{*text}, *vocab); err != nil {
			fmt.Fprintf(os.Stderr, "Gagal melatih WordPiece: %v\n", err)
			os.Exit(1)
		}
		tok = wpt
	case "unigram":
		ug := data.NewUnigramTokenizer(false, data.DefaultSpecialIDs())
		if err := ug.Train([]string{*text}, *vocab); err != nil {
			fmt.Fprintf(os.Stderr, "Gagal melatih Unigram: %v\n", err)
			os.Exit(1)
		}
		tok = ug
	default:
		fmt.Fprintf(os.Stderr, "jenis tokenizer tidak diketahui: %q\n", *tokenizerName)
		os.Exit(2)
	}

	enc, err := tok.Encode(ctx, *text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Gagal encode: %v\n", err)
		os.Exit(1)
	}
	decoded, err := tok.Decode(ctx, enc.IDs())
	if err != nil {
		fmt.Fprintf(os.Stderr, "Gagal decode: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("tokenizer     : %s\n", *tokenizerName)
	fmt.Printf("teks asli     : %q\n", *text)
	fmt.Printf("vocab size    : %d\n", tok.VocabSize())
	fmt.Printf("jumlah token  : %d\n", enc.Len())
	fmt.Printf("IDs           : %v\n", enc.IDs())
	fmt.Printf("tokens        : %v\n", enc.Tokens())
	fmt.Printf("special mask  : %v\n", enc.SpecialTokensMask())
	fmt.Printf("decode ulang  : %q\n", decoded)
	if decoded == *text {
		fmt.Println("status        : round-trip OK")
	} else {
		fmt.Println("status        : round-trip BERBEDA")
	}
}
