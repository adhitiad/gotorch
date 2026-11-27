// Package distributed defines stable run and worker identity contracts.
// Transport, collectives, and worker orchestration are implemented in P10.
package distributed

import (
	"fmt"
	"strings"

	"github.com/adhitiad/gotorch/errs"
)

// Identity uniquely locates one worker in a specific distributed run epoch.
// RunID and WorkerID are opaque, globally unique identifiers supplied by the caller.
type Identity struct {
	RunID     string
	WorkerID  string
	Rank      uint32
	WorldSize uint32
	Epoch     uint64
}

// Validate checks the local identity invariants.
func (id Identity) Validate() error {
	if strings.TrimSpace(id.RunID) == "" || strings.TrimSpace(id.WorkerID) == "" {
		return fmt.Errorf("%w: run ID and worker ID are required", errs.ErrInvalidDistributedID)
	}
	if id.WorldSize == 0 || id.Rank >= id.WorldSize {
		return fmt.Errorf("%w: rank %d must be less than non-zero world size %d", errs.ErrInvalidDistributedID, id.Rank, id.WorldSize)
	}
	return nil
}

// ValidateMembership verifies a complete, duplicate-free membership snapshot.
// Every member must belong to the same run, epoch, and world size, and every rank
// in [0, WorldSize) must appear exactly once.
func ValidateMembership(members []Identity) error {
	if len(members) == 0 {
		return fmt.Errorf("%w: membership is empty", errs.ErrInvalidDistributedID)
	}
	first := members[0]
	if err := first.Validate(); err != nil {
		return err
	}
	if uint64(len(members)) != uint64(first.WorldSize) {
		return fmt.Errorf("%w: got %d members for world size %d", errs.ErrInvalidDistributedID, len(members), first.WorldSize)
	}
	ranks := make(map[uint32]struct{}, len(members))
	workers := make(map[string]struct{}, len(members))
	for _, member := range members {
		if err := member.Validate(); err != nil {
			return err
		}
		if member.RunID != first.RunID || member.Epoch != first.Epoch || member.WorldSize != first.WorldSize {
			return fmt.Errorf("%w: membership contains inconsistent run metadata", errs.ErrInvalidDistributedID)
		}
		if _, exists := ranks[member.Rank]; exists {
			return fmt.Errorf("%w: duplicate rank %d", errs.ErrInvalidDistributedID, member.Rank)
		}
		if _, exists := workers[member.WorkerID]; exists {
			return fmt.Errorf("%w: duplicate worker %q", errs.ErrInvalidDistributedID, member.WorkerID)
		}
		ranks[member.Rank] = struct{}{}
		workers[member.WorkerID] = struct{}{}
	}
	return nil
}
