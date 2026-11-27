package runtime

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type testComponent struct {
	name string
	log  *[]string
	fail bool
}

func (c testComponent) Name() string { return c.name }
func (c testComponent) Start(context.Context) error {
	*c.log = append(*c.log, "start:"+c.name)
	if c.fail {
		return errors.New("start failed")
	}
	return nil
}
func (c testComponent) Stop(context.Context) error {
	*c.log = append(*c.log, "stop:"+c.name)
	return nil
}

func TestManagerLifecycleOrderAndIdempotence(t *testing.T) {
	var log []string
	m, err := NewManager(testComponent{name: "one", log: &log}, testComponent{name: "two", log: &log})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := m.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if m.State() != StateRunning {
		t.Fatalf("state = %s", m.State())
	}
	if err := m.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	want := []string{"start:one", "start:two", "stop:two", "stop:one"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("log = %v, want %v", log, want)
	}
	if m.State() != StateClosed {
		t.Fatalf("state = %s", m.State())
	}
}

func TestManagerRollsBackFailedStart(t *testing.T) {
	var log []string
	m, err := NewManager(testComponent{name: "one", log: &log}, testComponent{name: "two", log: &log, fail: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Start(context.Background()); err == nil {
		t.Fatal("expected startup failure")
	}
	want := []string{"start:one", "start:two", "stop:one"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("log = %v, want %v", log, want)
	}
	if m.State() != StateFailed {
		t.Fatalf("state = %s", m.State())
	}
	if err := m.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestManagerRejectsDuplicateNames(t *testing.T) {
	var log []string
	_, err := NewManager(testComponent{name: "same", log: &log}, testComponent{name: "same", log: &log})
	if err == nil {
		t.Fatal("duplicate component name accepted")
	}
}
