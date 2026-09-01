#!/usr/bin/env python3
"""selftest.py -- verify every component before any experiment is run.

Runs on any platform. Anything requiring Linux cgroups is skipped and reported
as skipped rather than silently passing.
"""
import subprocess, sys, tempfile, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "harness")); sys.path.insert(0, str(HERE / "analysis"))

P=F=SK=0
def check(name, cond, detail=""):
    global P,F
    ok=bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    P+=ok; F+=(not ok); return ok
def skip(name, why):
    global SK; SK+=1; print(f"  [SKIP] {name}  ({why})")

print("=" * 70); print("RHX SELF-TEST"); print("=" * 70)

print("\n[1] workload binary")
bin_path = HERE/"workload"/"rategen"
if not check("rategen built", bin_path.exists(), str(bin_path)):
    print("  run: cd workload && make"); sys.exit(1)

print("\n[2] Poisson generator statistical validity")
from workload import Workload, RateSpec, parse_counts, parse_residency
with tempfile.TemporaryDirectory() as td:
    wl=Workload(bin_path, Path(td)/"w", 4000, 999, RateSpec("gamma",0.8,0.5))
    wl.start(duration_s=30, report_s=0); time.sleep(0.5); wl.reset_counts()
    time.sleep(14)
    cp=wl.dump_counts("st"); wl.stop()
    cd=parse_counts(cp); lam=cd["lambda_true"]; c=cd["counts"]; W=cd["elapsed_s"]
    exp=lam.sum()*W; z=(c.sum()-exp)/np.sqrt(exp)
    check("aggregate rate matches sum of lambdas", abs(z)<4, f"z={z:+.3f}")
    # A wide rate band is a MIXTURE of Poissons and is overdispersed by
    # construction, so var/mean != 1 there. The correct invariant is the law of
    # total variance: Var(count) = E[lam]*W + W^2*Var(lam), which holds for any
    # band. Checked on a wide band, plus var/mean~1 on a narrow band.
    band=(lam>0.5)&(lam<1.5)
    if band.sum()>50:
        lb, cb = lam[band], c[band]
        pred_var = lb.mean()*W + (W**2)*lb.var()
        obs_var  = cb.var(ddof=1)
        rel = abs(obs_var-pred_var)/pred_var
        check("counts obey the law of total variance", rel<0.25,
              f"obs={obs_var:.1f} pred={pred_var:.1f} rel={rel:.3f}")
    narrow=(lam>0.9)&(lam<1.1)
    if narrow.sum()>40:
        cn=c[narrow]; ratio=cn.var(ddof=1)/max(cn.mean(),1e-9)
        check("narrow band is Poisson (var/mean~1)", 0.55<ratio<1.8,
              f"var/mean={ratio:.3f}")
    lh=c/W; err=abs(lh[lam>0.5].mean()-lam[lam>0.5].mean())/lam[lam>0.5].mean()
    check("rate recovery for lambda>0.5", err<0.10, f"rel err={err:.4f}")

print("\n[3] estimator")
from estimator import *
rng=np.random.default_rng(0); lt=rng.gamma(0.8,0.5,20000)
oe=estimate_oracle(lt); w=victim_weights(lt,"coldest_first",frac=0.3)
check("coldest_first selects the requested fraction", abs(w.mean()-0.3)<0.01, f"{w.mean():.4f}")
check("victims are colder than the population", lt[w>0].mean()<lt.mean(),
      f"{lt[w>0].mean():.4f} < {lt.mean():.4f}")
p=predict_recovery(oe,w,60.0,n_grid=25,n_boot=150,seed=1)
check("S(0)==1 exactly", p.S_pred[0]==1.0 and p.S_lo[0]==1.0 and p.S_hi[0]==1.0)
check("S(t) is non-increasing", np.all(np.diff(p.S_pred)<=1e-12))
vic=lt[w>0]; St=np.array([np.mean(np.exp(-vic*t)) for t in p.t_grid])
check("bracket contains the true S(t) everywhere",
      np.all((p.S_lo-1e-9<=St)&(St<=p.S_hi+1e-9)))
c0=compare_densities(lt,lt,60.0)
check("identical densities give zero error", c0["S_max_abs_err"]<1e-12)

print("\n[4] kernel fitting and scoring")
from kernels import *
t=np.linspace(0,60,40)
f_=fit_family(t,k_power(t,10.0,0.8),"power_law")
check("power-law params recovered", abs(f_.params["tau"]-10)<1e-3 and abs(f_.params["zeta"]-0.8)<1e-4)
check("family selection picks power_law", compare_families(t,k_power(t,10.,.8))["best_by_aicc"]=="power_law")
check("family selection picks exponential", compare_families(t,k_exponential(t,12.))["best_by_aicc"]=="exponential")
Sv=k_power(t,10.,.8)
pr={"t_grid":list(t),"S_pred":list(Sv),"S_pred_lo":list(Sv*.95),"S_pred_hi":list(Sv*1.05),
    "window_T_s":60.,"lambda_min":1/60,"frozen_fraction":.1,"estimator":{}}
check("perfect prediction scores coverage 1.0", score_prediction(pr,t,Sv)["coverage"]==1.0)
off=score_prediction(pr,t,Sv+0.15)
check("offset error diagnosed as offset-like", "offset-like" in off["diagnosis"])
sh=score_prediction(pr,t,k_exponential(t,10.))
check("shape error diagnosed as shape-like", "shape-like" in sh["diagnosis"])
lamx=rng.gamma(0.8,0.5,100000)
th=frontier_from_rate(lamx,0.15)
check("frontier inverts the cold flux", abs(cold_flux(lamx,th)-0.15)<1e-4, f"theta={th:.5f}")
check("infeasible rho returns nan (Thm 9)", np.isnan(frontier_from_rate(lamx,1e6)))
u=np.full(100000,1.0)
check("homogeneous amplification == 1 (Thm 6)",
      abs(amplification_from_rates(u,0.3,0.1)["amplification"]-1.0)<1e-9)
check("heterogeneous amplification > 1 (Thm 5)",
      amplification_from_rates(lamx,0.3,0.1)["amplification"]>1.2)

print("\n[5] pre-registration ledger")
import prereg
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    good={"window_T_s":60.,"lambda_min":1/60.,"frozen_fraction":.2,
          "t_grid":[0,30,60],"S_pred":[1,.5,.3],"S_pred_lo":[1,.4,.2],
          "S_pred_hi":[1,.6,.4],"estimator":{"src":"test"}}
    r=prereg.register_prediction(root,"a",good,"envhash")
    check("prediction registers", len(r["digest"])==64)
    try: prereg.register_prediction(root,"a",good,"envhash"); ok=False
    except FileExistsError: ok=True
    check("re-registration is refused", ok)
    bad=dict(good); bad["lambda_min"]=0.5
    try: prereg.register_prediction(root,"b",bad,"e"); ok=False
    except ValueError: ok=True
    check("lambda_min != 1/T is rejected", ok)
    check("ledger verifies", prereg.Ledger(root).verify()["ok"])
    pp=root/"predictions"/"a.json"; d=json.loads(pp.read_text()); d["S_pred"][1]=0.99
    pp.write_text(prereg.canonical_json(d))
    try: prereg.load_prediction(root,"a"); ok=False
    except ValueError: ok=True
    check("tampering with a prediction is detected", ok)
    led=root/"prereg_ledger.jsonl"; lines=led.read_text().splitlines()
    e=json.loads(lines[0]); e["payload"]["label"]="zzz"
    lines[0]=json.dumps(e,sort_keys=True); led.write_text("\n".join(lines)+"\n")
    check("tampering with the ledger is detected", not prereg.Ledger(root).verify()["ok"])

print("\n[6] randomization inference")
from randomization import *
d=confounding_demo(3000,seed=3)
check("observational data is confounded (nonzero corr, zero true effect)",
      abs(d["observational_partial_corr"])>0.0)
check("randomized test is calibrated", 0.01<d["randomized_p_value"]<0.99,
      f"p={d['randomized_p_value']:.4f}")
rng2=np.random.default_rng(11); ps=[]
for _ in range(120):
    n=600; X=rng2.normal(0,1,n); H=(rng2.random(n)<0.5).astype(int)
    D=1.2*X+rng2.normal(0,1,n)
    ps.append(randomization_test(H,np.full(n,.5),make_partial_corr_statistic(D,X),
              n_resample=200,seed=int(rng2.integers(1<<30)))["p_value"])
ps=np.array(ps)
check("Type-I error at nominal 5%", (ps<0.05).mean()<0.12, f"observed={(ps<0.05).mean():.3f}")
try: SequentialRandomizer(1, propensity=0.02); ok=False
except ValueError: ok=True
check("propensity bounded away from 0/1 is enforced", ok)

print("\n[7] cgroup / telemetry")
from cgroupv2 import cgroup_v2_available, parse_psi, parse_keyed
psi=parse_psi("some avg10=8.45 avg60=5.20 avg300=2.10 total=982341\nfull avg10=3.21 avg60=1.87 avg300=0.94 total=341209")
check("PSI parser reads some/full", psi["some_avg10"]==8.45 and psi["full_total"]==341209.0)
check("PSI parser tolerates empty input", parse_psi(None)["some_avg10"] is None)
ks=parse_keyed("anon 100\nfile 200\n",["anon","file","missing"],"stat_")
check("keyed parser distinguishes absent from zero",
      ks["stat_anon"]==100 and ks["stat_missing"] is None)
if cgroup_v2_available(): check("cgroup v2 available", True)
else: skip("cgroup v2 operations", "not a Linux host with unified hierarchy")

print("\n[8] environment gate")
import envcapture
env=envcapture.capture(); v=envcapture.validate_for_claims(env)
invalid_host = not env["is_linux"] or env["virtualization"]["is_virtualized"]
check("environment capture handles absent fields and refuses invalid hosts",
      "publishable" in v and ((not v["publishable"]) if invalid_host else True))
original_validator = envcapture.validate_for_claims
try:
    def broken_validator(_env):
        raise RuntimeError("deliberate validator failure")
    envcapture.validate_for_claims = broken_validator
    safe_v = envcapture.validate_for_claims_safe(env)
finally:
    envcapture.validate_for_claims = original_validator
check("validator failures become publishability blockers",
      safe_v["publishable"] is False and bool(safe_v["blockers"]) and
      safe_v.get("validation_error", {}).get("type") == "RuntimeError")

print("\n" + "=" * 70)
print(f"PASS {P}   FAIL {F}   SKIP {SK}")
print("=" * 70)
sys.exit(1 if F else 0)
