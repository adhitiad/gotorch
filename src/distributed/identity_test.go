package distributed

import (
	"errors"
	"testing"

	"github.com/adhitiad/gotorch/src/errs"
)

func TestIdentityValidation(t *testing.T) {
	valid := Identity{RunID: "run-a", WorkerID: "worker-a", Rank: 0, WorldSize: 1}
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	valid.Rank = 1
	if err := valid.Validate(); err == nil || !errors.Is(err, errs.ErrInvalidDistributedID) {
		t.Fatalf("got %v", err)
	}
}

func TestValidateMembership(t *testing.T) {
	members := []Identity{
		{RunID: "run-a", WorkerID: "worker-a", Rank: 0, WorldSize: 2, Epoch: 4},
		{RunID: "run-a", WorkerID: "worker-b", Rank: 1, WorldSize: 2, Epoch: 4},
	}
	if err := ValidateMembership(members); err != nil {
		t.Fatal(err)
	}
	members[1].Rank = 0
	if err := ValidateMembership(members); err == nil {
		t.Fatal("duplicate rank accepted")
	}
}

func TestValidateMembershipRejectsMixedEpoch(t *testing.T) {
	members := []Identity{
		{RunID: "run-a", WorkerID: "worker-a", Rank: 0, WorldSize: 2, Epoch: 4},
		{RunID: "run-a", WorkerID: "worker-b", Rank: 1, WorldSize: 2, Epoch: 5},
	}
	if err := ValidateMembership(members); err == nil {
		t.Fatal("mixed epochs accepted")
	}
}
