// Package scheduler menyediakan eksekusi paralel terbatas dengan pembatalan,
// batas antrean, penutupan deterministik, dan penanganan panic.
package scheduler

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"

	"github.com/adhitiad/gotorch/src/errs"
)

// Batas atas konfigurasi mencegah pembentukan goroutine atau antrean tanpa batas.
const (
	MaxWorkers = 1024
	MaxQueue   = 1 << 20
)

// ErrSchedulerClosed dikembalikan saat scheduler tidak menerima pekerjaan baru.
var ErrSchedulerClosed = errors.New("gotorch: scheduler is closed")

// ErrQueueFull dikembalikan TrySubmit saat antrean sudah penuh.
var ErrQueueFull = errors.New("gotorch: scheduler queue is full")

// PanicError membungkus nilai panic dari sebuah task agar tidak membunuh worker.
type PanicError struct {
	Value any
	Stack []byte
}

func (e *PanicError) Error() string {
	if e == nil {
		return "gotorch: nil panic error"
	}
	return fmt.Sprintf("gotorch: task panicked with %v", e.Value)
}

// Unwrap meneruskan panic berupa error agar errors.Is tetap berguna.
func (e *PanicError) Unwrap() error {
	if e == nil {
		return nil
	}
	err, ok := e.Value.(error)
	if ok {
		return err
	}
	return nil
}

// Task adalah unit kerja yang menerima konteks gabungan pemanggil dan scheduler.
type Task func(context.Context) error

// Config mengatur jumlah worker, kapasitas antrean, ukuran chunk, dan kebijakan thread BLAS.
type Config struct {
	Workers     int
	Queue       int
	ChunkSize   int
	BLASThreads int
	OnError     func(error)
}

// DefaultConfig memakai GOMAXPROCS untuk worker dan empat kali lipatnya untuk antrean.
func DefaultConfig() Config {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > MaxWorkers {
		workers = MaxWorkers
	}
	blas := runtime.GOMAXPROCS(0)
	if blas < 1 {
		blas = 1
	}
	if blas > MaxWorkers {
		blas = MaxWorkers
	}
	return Config{Workers: workers, Queue: workers * 4, BLASThreads: blas}
}

// Validate menolak konfigurasi yang ambigu atau tidak terbatas.
func (c Config) Validate() error {
	if c.Workers < 1 || c.Workers > MaxWorkers {
		return fmt.Errorf("%w: workers must be in [1,%d]", errs.ErrInvalidArgument, MaxWorkers)
	}
	if c.Queue < 0 || c.Queue > MaxQueue {
		return fmt.Errorf("%w: queue must be in [0,%d]", errs.ErrInvalidArgument, MaxQueue)
	}
	if c.ChunkSize < 0 {
		return fmt.Errorf("%w: chunk size must be non-negative", errs.ErrInvalidArgument)
	}
	if c.BLASThreads < 0 {
		return fmt.Errorf("%w: BLAS threads must be non-negative", errs.ErrInvalidArgument)
	}
	return nil
}

// Future adalah hasil asynchronous dari satu task.
type Future struct {
	done chan struct{}
	err  error
}

// Wait menunggu task selesai atau konteks pembanggil dibatalkan.
func (f *Future) Wait(ctx context.Context) error {
	if f == nil || isNil(ctx) {
		return errs.ErrInvalidArgument
	}
	select {
	case <-f.done:
		return f.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

type workerKey struct{}

type job struct {
	ctx    context.Context
	fn     Task
	future *Future
}

// Scheduler memiliki sejumlah worker tetap dan satu antrean berkapasitas terbatas.
// Close menghentikan penerimaan kerja, membatalkan konteks pekerjaan, menguras
// antrean, lalu menunggu semua worker berhenti. Task wajib menghormati konteksnya.
type Scheduler struct {
	config    Config
	tasks     chan *job
	stop      chan struct{}
	runCtx    context.Context
	cancel    context.CancelFunc
	submitMu  sync.Mutex
	submitWG  sync.WaitGroup
	workerWG  sync.WaitGroup
	once      sync.Once
	done      chan struct{}
	closed    bool
	running   atomic.Int64
	submitted atomic.Uint64
	completed atomic.Uint64
	failed    atomic.Uint64
	panics    atomic.Uint64
	canceled  atomic.Uint64
}

// New membuat scheduler dan langsung memulai worker sebanyak config.Workers.
func New(config Config) (*Scheduler, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	runCtx, cancel := context.WithCancel(context.Background())
	s := &Scheduler{
		config: config,
		tasks:  make(chan *job, config.Queue),
		stop:   make(chan struct{}),
		runCtx: runCtx,
		cancel: cancel,
		done:   make(chan struct{}),
	}
	s.workerWG.Add(config.Workers)
	for i := 0; i < config.Workers; i++ {
		go s.worker()
	}
	return s, nil
}

// Submit mengantrekan task dan menunggu ruang antrean bila penuh.
func (s *Scheduler) Submit(ctx context.Context, fn Task) (*Future, error) {
	return s.submit(ctx, fn, true)
}

// TrySubmit mengantrekan task tanpa menunggu; ErrQueueFull bila antrean penuh.
func (s *Scheduler) TrySubmit(ctx context.Context, fn Task) (*Future, error) {
	return s.submit(ctx, fn, false)
}

// SubmitWait mengantrekan task lalu menunggu hasilnya.
func (s *Scheduler) SubmitWait(ctx context.Context, fn Task) error {
	future, err := s.Submit(ctx, fn)
	if err != nil {
		return err
	}
	return future.Wait(ctx)
}

func (s *Scheduler) submit(ctx context.Context, fn Task, block bool) (*Future, error) {
	if s == nil || isNil(ctx) || fn == nil {
		return nil, errs.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	future := &Future{done: make(chan struct{})}
	j := &job{ctx: ctx, fn: fn, future: future}

	s.submitMu.Lock()
	if s.closed {
		s.submitMu.Unlock()
		return nil, ErrSchedulerClosed
	}
	// Pekerjaan bersarang dijalankan sebaris agar tidak memakan worker lain.
	if isWorker(ctx) {
		s.submitMu.Unlock()
		s.submitted.Add(1)
		s.executeJob(j)
		return future, nil
	}

	// Hitungan submit ditambahkan sebelum melepas kunci agar shutdown tidak
	// pernah menutup antrean saat pengiriman masih berlangsung.
	s.submitWG.Add(1)
	s.submitMu.Unlock()
	defer s.submitWG.Done()
	if block {
		select {
		case s.tasks <- j:
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-s.stop:
			return nil, ErrSchedulerClosed
		}
	} else {
		select {
		case s.tasks <- j:
		default:
			return nil, ErrQueueFull
		}
	}
	s.submitted.Add(1)
	return future, nil
}

func (s *Scheduler) worker() {
	defer s.workerWG.Done()
	for j := range s.tasks {
		s.executeJob(j)
	}
}

func (s *Scheduler) executeJob(j *job) {
	s.running.Add(1)
	defer s.running.Add(-1)

	taskCtx, cancel := context.WithCancel(j.ctx)
	stopWatch := context.AfterFunc(s.runCtx, cancel)
	taskCtx = context.WithValue(taskCtx, workerKey{}, true)
	err, panicked := s.runTask(taskCtx, j.fn)
	stopWatch()
	cancel()

	switch {
	case err == nil:
		s.completed.Add(1)
	case panicked:
		s.panics.Add(1)
		s.failed.Add(1)
	default:
		s.failed.Add(1)
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			s.canceled.Add(1)
		}
	}
	if err != nil {
		s.notifyError(err)
	}
	j.future.err = err
	close(j.future.done)
}

func (s *Scheduler) runTask(ctx context.Context, fn Task) (err error, panicked bool) {
	defer func() {
		if value := recover(); value != nil {
			err = &PanicError{Value: value, Stack: debug.Stack()}
			panicked = true
		}
	}()
	return fn(ctx), false
}

func (s *Scheduler) notifyError(err error) {
	if s.config.OnError == nil {
		return
	}
	defer func() { _ = recover() }()
	s.config.OnError(err)
}

// Close menghentikan penerimaan kerja dan menunggu pengurasan seluruh antrean.
// Pemanggilan ulang aman dan mengembalikan hasil penutupan pertama.
func (s *Scheduler) Close(ctx context.Context) error {
	if s == nil || isNil(ctx) {
		return errs.ErrInvalidArgument
	}
	s.once.Do(func() { go s.shutdown() })
	select {
	case <-s.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Scheduler) shutdown() {
	s.submitMu.Lock()
	if !s.closed {
		s.closed = true
		close(s.stop)
	}
	s.submitMu.Unlock()
	s.cancel()
	s.submitWG.Wait()
	close(s.tasks)
	s.workerWG.Wait()
	close(s.done)
}

// Stats mengembalikan snapshot jumlah worker, antrean, dan hasil eksekusi.
func (s *Scheduler) Stats() Stats {
	if s == nil {
		return Stats{}
	}
	s.submitMu.Lock()
	defer s.submitMu.Unlock()
	return Stats{
		Workers:       s.config.Workers,
		QueueCapacity: s.config.Queue,
		Queued:        len(s.tasks),
		Running:       int(s.running.Load()),
		Submitted:     s.submitted.Load(),
		Completed:     s.completed.Load(),
		Failed:        s.failed.Load(),
		Panics:        s.panics.Load(),
		Canceled:      s.canceled.Load(),
		Closed:        s.closed,
	}
}

// Workers mengembalikan jumlah worker tetap yang dimiliki scheduler.
func (s *Scheduler) Workers() int {
	if s == nil {
		return 0
	}
	return s.config.Workers
}

// Stats adalah snapshot operasional scheduler.
type Stats struct {
	Workers       int
	QueueCapacity int
	Queued        int
	Running       int
	Submitted     uint64
	Completed     uint64
	Failed        uint64
	Panics        uint64
	Canceled      uint64
	Closed        bool
}

func isWorker(ctx context.Context) bool {
	value := ctx.Value(workerKey{})
	worker, ok := value.(bool)
	return ok && worker
}

func isNil(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}
