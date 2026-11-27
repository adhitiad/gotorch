package compute

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/adhitiad/gotorch/src/dtype"
	"github.com/adhitiad/gotorch/src/errs"
	"github.com/adhitiad/gotorch/src/shape"
	"github.com/adhitiad/gotorch/src/storage"
	"github.com/adhitiad/gotorch/src/tensor"
)

func floatArg(t *testing.T, values []float64, dims ...int64) Argument {
	t.Helper()
	sh,err:=shape.New(dims...);if err!=nil{t.Fatal(err)}
	meta,err:=tensor.NewMetadata(dtype.Float64,sh);if err!=nil{t.Fatal(err)}
	size,err:=meta.RequiredBytes();if err!=nil{t.Fatal(err)}
	s,err:=storage.NewHeapAllocator().Allocate(context.Background(),size);if err!=nil{t.Fatal(err)}
	t.Cleanup(func(){_ = s.Release()})
	raw:=make([]byte,len(values)*8)
	for i,v:=range values{binary.LittleEndian.PutUint64(raw[i*8:],math.Float64bits(v))}
	if len(raw)>0 { if n,e:=s.WriteAt(raw,0);e!=nil||n!=len(raw){t.Fatalf("write fixture: %d %v",n,e)} }
	return Argument{Metadata:meta,Storage:s}
}

func outputArg(t *testing.T, dims ...int64) Argument {
	t.Helper()
	sh,err:=shape.New(dims...);if err!=nil{t.Fatal(err)}
	meta,err:=tensor.NewMetadata(dtype.Float64,sh);if err!=nil{t.Fatal(err)}
	size,err:=meta.RequiredBytes();if err!=nil{t.Fatal(err)}
	s,err:=storage.NewHeapAllocator().Allocate(context.Background(),size);if err!=nil{t.Fatal(err)}
	t.Cleanup(func(){_ = s.Release()})
	return Argument{Metadata:meta,Storage:s}
}

func readFixture(t *testing.T,arg Argument,n int) []float64 {
	t.Helper()
	raw:=make([]byte,n*8)
	if n>0 { got,err:=arg.Storage.ReadAt(raw,0);if err!=nil||got!=len(raw){t.Fatalf("read fixture: %d %v",got,err)} }
	out:=make([]float64,n)
	for i:=range out{out[i]=math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))}
	return out
}

func TestReferenceElementwiseAndReduction(t *testing.T) {
	backend:=NewReferenceCPUBackend()
	a:=floatArg(t,[]float64{1,2,3},3)
	b:=floatArg(t,[]float64{4,5,6},3)
	out:=outputArg(t,3)
	spec:=KernelSpec{Operation:OpMul,DType:dtype.Float64,Layout:tensor.LayoutContiguous}
	if err:=Run(context.Background(),backend,spec,Invocation{Inputs:[]Argument{a,b},Outputs:[]Argument{out}});err!=nil{t.Fatal(err)}
	got:=readFixture(t,out,3)
	want:=[]float64{4,10,18}
	for i:=range want{if got[i]!=want[i]{t.Fatalf("mul[%d]=%v want %v",i,got[i],want[i])}}

	scalar:=outputArg(t)
	if err:=Run(context.Background(),backend,KernelSpec{Operation:OpSum,DType:dtype.Float64,Layout:tensor.LayoutContiguous},Invocation{Inputs:[]Argument{a},Outputs:[]Argument{scalar}});err!=nil{t.Fatal(err)}
	if got:=readFixture(t,scalar,1)[0];got!=6{t.Fatalf("sum=%v want 6",got)}
}

func TestReferenceMatMul(t *testing.T) {
	backend:=NewReferenceCPUBackend()
	a:=floatArg(t,[]float64{1,2,3,4,5,6},2,3)
	b:=floatArg(t,[]float64{7,8,9,10,11,12},3,2)
	out:=outputArg(t,2,2)
	err:=Run(context.Background(),backend,KernelSpec{Operation:OpMatMul,DType:dtype.Float64,Layout:tensor.LayoutContiguous},Invocation{Inputs:[]Argument{a,b},Outputs:[]Argument{out}})
	if err!=nil{t.Fatal(err)}
	got:=readFixture(t,out,4)
	want:=[]float64{58,64,139,154}
	for i:=range want{if got[i]!=want[i]{t.Fatalf("matmul[%d]=%v want %v",i,got[i],want[i])}}
}

func TestReferenceRejectsInvalidShapesAndEmptyMean(t *testing.T) {
	backend:=NewReferenceCPUBackend()
	a:=floatArg(t,[]float64{},0)
	out:=outputArg(t)
	err:=Run(context.Background(),backend,KernelSpec{Operation:OpMean,DType:dtype.Float64,Layout:tensor.LayoutContiguous},Invocation{Inputs:[]Argument{a},Outputs:[]Argument{out}})
	if !errors.Is(err,errs.ErrInvalidArgument){t.Fatalf("mean empty: %v",err)}
}

func TestRegistryDispatchAndDuplicateNames(t *testing.T) {
	registry:=NewRegistry()
	if err:=registry.Register(NewReferenceCPUBackend());err!=nil{t.Fatal(err)}
	if err:=registry.Register(NewReferenceCPUBackend());!errors.Is(err,errs.ErrInvalidState){t.Fatalf("duplicate: %v",err)}
	a:=floatArg(t,[]float64{2},1)
	out:=outputArg(t,1)
	call:=Invocation{Inputs:[]Argument{a},Outputs:[]Argument{out}}
	spec:=KernelSpec{Operation:OpNeg,DType:dtype.Float64,Layout:tensor.LayoutContiguous}
	selected,err:=registry.Select(spec,call);if err!=nil{t.Fatal(err)}
	if selected.Name()!="purego-reference"{t.Fatalf("selected %q",selected.Name())}
	if err:=registry.Execute(context.Background(),spec,call);err!=nil{t.Fatal(err)}
	if got:=readFixture(t,out,1)[0];got!=-2{t.Fatalf("neg=%v",got)}
}

func TestReferenceRejectsUnsupportedDType(t *testing.T) {
	backend:=NewReferenceCPUBackend()
	sh,_:=shape.New(1)
	meta,_:=tensor.NewMetadata(dtype.Int32,sh)
	arg:=Argument{Metadata:meta,Storage:&testStorage{size:4}}
	if backend.Supports(KernelSpec{Operation:OpNeg,DType:dtype.Int32,Layout:tensor.LayoutContiguous},Invocation{Inputs:[]Argument{arg},Outputs:[]Argument{arg}}){t.Fatal("int32 unexpectedly supported")}
}
