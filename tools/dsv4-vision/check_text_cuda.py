"""Independent BF16-boundary and FP64 checks for the native TP2 text graph glue."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_exl3_cuda import Driver
from check_ops_cuda import bf16, floats, u, f


def check(d):
    rng = np.random.default_rng(24101)
    cases = []
    def record(name, actual, expected, atol=.005, rtol=.005):
        err = np.abs(actual.astype(np.float64) - expected)
        matched = bool(np.all(np.isclose(actual, expected, atol=atol, rtol=rtol)))
        cases.append(dict(name=name, passed=matched, max_abs=float(err.max()),
                          max_abs_by_column=err.reshape(-1, err.shape[-1]).max(axis=0).tolist()))
    for qnorm in (0, 1):
        raw = bf16(rng.normal(0, .3, (3, 512)))
        x = floats(raw)
        weight = rng.normal(1, .01, 512).astype(np.float32)
        w = d.upload(weight) if not qnorm else u(0)
        y = d.empty(raw.nbytes)
        d.launch('tf_dsv4_text_norm', (3, 1, 1), (256, 1, 1), d.upload(raw), w, y, u(512), u(0), f(1e-6), u(qnorm))
        if qnorm:
            mean = floats(bf16(floats(bf16(x*x)).mean(axis=1, keepdims=True)))
            inv = floats(bf16(1 / np.sqrt(floats(bf16(mean + np.float32(1e-6))))))
            expected = floats(bf16(x * inv))
        else:
            expected = floats(bf16(x / np.sqrt(np.mean(x.astype(np.float64)**2, axis=1, keepdims=True)+1e-6) * weight))
        record('qnorm' if qnorm else 'rmsnorm', floats(d.download(y, raw.shape, np.uint16)), expected)
    raw = bf16(rng.normal(size=(3, 128)))
    x = floats(raw).astype(np.float64)
    for step in (1, 2, 4, 8, 16, 32, 64):
        v = x.reshape(3, -1, step*2)
        a, b = v[:, :, :step].copy(), v[:, :, step:].copy()
        v[:, :, :step], v[:, :, step:] = a+b, a-b
    y = d.upload(raw)
    d.launch('tf_dsv4_text_rotate', (3,1,1),(128,1,1), y, u(128))
    record('indexer_hadamard',floats(d.download(y,raw.shape,np.uint16)),x/np.sqrt(128))
    for inverse in (0,1):
        raw = bf16(rng.normal(size=(3,512)))
        freq = (1 / 160000 ** (np.arange(32,dtype=np.float32)*2/64)).astype(np.float32)
        y = d.upload(raw)
        d.launch('tf_dsv4_text_rope',(3,1,1),(128,1,1),y,d.upload(freq),u(262143),u(512),u(64),u(inverse))
        expected = floats(raw).copy()
        angles = (np.float32(262143)*freq).astype(np.float64)
        a,b=expected[:,-64::2].copy(),expected[:,-63::2].copy()
        s=np.sin(angles)*(-1 if inverse else 1);c=np.cos(angles)
        expected[:,-64::2],expected[:,-63::2]=a*c-b*s,b*c+a*s
        record('inverse_rope' if inverse else 'rope',floats(d.download(y,raw.shape,np.uint16)),expected,atol=.008)
    for ratio in (4,128):
        dim=128;channels=dim*(2 if ratio==4 else 1);slots=ratio*(2 if ratio==4 else 1)
        state=d.empty(slots*channels*4);scores=d.empty(slots*channels*4);y=d.empty(dim*2)
        ape=rng.normal(0,.1,(ratio,channels)).astype(np.float32);ape_ptr=d.upload(ape)
        for request in range(2):
            kv=rng.normal(0,.2,(ratio*3,channels)).astype(np.float32)
            logits=rng.normal(0,.4,kv.shape).astype(np.float32)
            for pos in range(ratio*3):
                d.launch('tf_dsv4_text_store',((channels+127)//128,1,1),(128,1,1),d.upload(kv[pos]),d.upload(logits[pos]),ape_ptr,state,scores,u(pos),u(ratio),u(dim))
                if (pos+1)%ratio:continue
                d.launch('tf_dsv4_text_pool',((dim+127)//128,1,1),(128,1,1),state,scores,y,u(pos),u(ratio),u(dim))
                start=max(0,pos+1-slots);end=pos+1
                values=[];weights=[]
                for i in range(start,end):
                    part=dim if ratio==4 and i>=end-ratio else 0
                    values.append(kv[i,part:part+dim]);weights.append((logits[i]+ape[i%ratio])[part:part+dim])
                values=np.array(values,dtype=np.float64);weights=np.array(weights,dtype=np.float64)
                probability=np.exp(weights-weights.max(axis=0));probability/=probability.sum(axis=0)
                expected=(values*probability).sum(axis=0)
                record(f'pool_{ratio}_request{request}_pos{pos}',floats(d.download(y,(dim,),np.uint16)),expected,atol=.001)
    for cast in (0,1,2):
        raw=rng.normal(size=(2,257)).astype(np.float32);y=d.empty(257*4)
        d.launch('tf_dsv4_text_merge',(3,1,1),(128,1,1),d.upload(raw),y,u(257),u(cast))
        expected=raw[0]+raw[1]
        actual=d.download(y,(257,),np.uint16 if cast==1 else np.float32)
        record(f'tp_merge_{cast}',floats(actual) if cast==1 else actual,floats(bf16(expected)) if cast else expected,atol=0,rtol=0)
    for hashed in (0,1):
        logits=rng.normal(size=256).astype(np.float32);bias=rng.normal(0,.01,256).astype(np.float32)
        ids=np.array([17],np.uint32);table=np.tile(np.arange(6,dtype=np.int64),(32,1))
        picks=d.empty(24);weights=d.empty(24)
        d.launch('tf_dsv4_text_route',(1,1,1),(1,1,1),d.upload(logits),d.upload(ids),d.upload(table),d.upload(bias),picks,weights,u(hashed),f(2.5))
        expected=np.sqrt(np.logaddexp(0,logits.astype(np.float64)))
        top=table[17] if hashed else np.argsort(-(expected+bias))[:6]
        np.testing.assert_array_equal(d.download(picks,(6,),np.uint32),top)
        record(f'route_{hashed}',d.download(weights,(6,),np.float32),expected[top]/expected[top].sum()*2.5,atol=1e-6,rtol=1e-6)
    q, kv = [bf16(rng.normal(0,.2,shape)) for shape in [(2,128),(17,128)]]
    weights=bf16(rng.normal(0,.1,2));out=d.empty(17*4)
    d.launch('tf_dsv4_text_index',(1,1,1),(128,1,1),d.upload(q),d.upload(kv),d.upload(weights),out,u(17),u(2),u(128))
    scores=floats(bf16(floats(q).astype(np.float64)@floats(kv).astype(np.float64).T))
    ww=floats(bf16(floats(weights)/np.sqrt(128*4)))
    expected=floats(bf16(floats(bf16(np.maximum(scores,0)*ww[:,None])).sum(axis=0)))
    record('index_scores',d.download(out,(17,),np.float32),expected,atol=.0001,rtol=.004)
    for kind in (0,1,2):
        x=rng.normal(size=257).astype(np.float32)
        raw=x if kind==0 else x.astype(np.float16) if kind==1 else bf16(x)
        out=d.empty(257*4)
        d.launch('tf_dsv4_text_cast',(3,1,1),(128,1,1),d.upload(raw),out,u(257),u(kind))
        expected=floats(raw) if kind==2 else raw.astype(np.float32)
        record(f'cast_{kind}',d.download(out,(257,),np.float32),expected,atol=0,rtol=0)
    x=np.array([-1,0,129279,2147483647],np.int32);out=d.empty(x.size*8)
    d.launch('tf_dsv4_text_ids',(1,1,1),(128,1,1),d.upload(x),out,u(x.size))
    np.testing.assert_array_equal(d.download(out,x.shape,np.int64),x.astype(np.int64))
    record('ids_widen',d.download(out,x.shape,np.int64),x,atol=0,rtol=0)
    x=bf16(rng.normal(size=257));y=rng.normal(size=257).astype(np.float32);out=d.upload(y)
    d.launch('tf_dsv4_text_accumulate',(3,1,1),(128,1,1),out,d.upload(x),u(257))
    record('expert_accumulate',d.download(out,y.shape,np.float32),y+floats(x),atol=0,rtol=0)
    x=rng.normal(size=257).astype(np.float32);out=d.empty(x.size*2)
    d.launch('tf_dsv4_text_round',(3,1,1),(128,1,1),d.upload(x),out,u(257))
    record('moe_round',floats(d.download(out,x.shape,np.uint16)),floats(bf16(x)),atol=0,rtol=0)
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fatbin');p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    d=Driver(args.fatbin,0)
    try:cases=check(d)
    finally:d.close()
    result=dict(passed=sum(c['passed'] for c in cases),failed=sum(not c['passed'] for c in cases),cases=cases,nccl_executed=False,full_model_executed=False)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print('text glue:',result['passed'],'passed,',result['failed'],'failed')
    if result['failed']:raise SystemExit(1)
if __name__=='__main__':main()
