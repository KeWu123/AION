"""Actual-file, frozen-workload experiment with streaming query admission.

K definitions are frozen from the common initial last-4096 context. Generic
arms evaluate a definition only on admission; this controlled workload does
not claim adaptive selection of future functions. See protocol.json for the
stronger information available to the two all-functions-known oracle arms.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import time
import numpy as np

import exp_file_controlled as base
from anq.file_store import ForwardFileStore, IndexedFileStore, prepare_prefix
from anq.position_scan import SequentialPositionScan, AtomicPositionScan
from anq.coordinated_scan import CoordinatedPositionScan
from anq.ratio import Prefix, RatioTrace
import file_workload as workload
import file_streaming_diagnostics as diagnostics

ROOT = Path(__file__).resolve().parents[1]
ARMS = (*base.ARMS, 'coordinated_wor', 'cohort_wor', 'known_wr_oracle')
DEFAULT_ARMS = ARMS
ORACLES = ('exact_fused', 'known_wr_oracle')
BASE_EXPECTED_SHA256 = 'df3a41b9ec06ed95ba50908334183fe15f3ddc701cd677a08926755e0544421e'
LEGACY_V1_SHA256 = '2757a8fb980dc75b8b53e0294ea83def9097589f6127c0e6de12664585a6a3e3'
PRECISION_FACTORS = (.5,1.,1.5,2.)
MAP_PEAK_ACCOUNTING = 'exact_trace_replay_v1'
SOURCE_NAMES = tuple(dict.fromkeys((
    'code/exp_file_streaming.py', 'code/file_streaming_diagnostics.py', *base.SOURCE_NAMES,
    *(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'code/anq').glob('*.py'))))))


def write_csv(path, rows):
    """Keep every field, including rows with oracle-specific diagnostics."""
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def query_contract(args, K):
    profile = getattr(args, 'precision_profile', 'heterogeneous')
    if profile not in ('uniform', 'heterogeneous'):
        raise ValueError('precision_profile must be uniform or heterogeneous')
    targets = [args.target * (PRECISION_FACTORS[k % len(PRECISION_FACTORS)]
               if profile == 'heterogeneous' else 1.) for k in range(K)]
    supplied = getattr(args, 'query_caps', None)
    caps = [args.cap] * K if supplied is None else list(supplied)
    if len(caps) != K or any(isinstance(c, bool) or int(c) != c or c <= 0 for c in caps):
        raise ValueError('query_caps must contain K positive integer suffix caps')
    if any(not math.isfinite(t) or t <= 0 for t in targets):
        raise ValueError('positive finite per-query targets required')
    return targets, list(map(int, caps))


def _empty():
    return np.empty(0, dtype=np.float64)


def _cleanup(store):
    """A secondary close failure is diagnostic and cannot replace the primary."""
    if store is None:
        return []
    try:
        store.close()
    except BaseException as error:
        return [dict(type=type(error).__name__,message=str(error))]
    return []


def _save_failure(trace_path,arm,error,started,states,done,reports,events,cleanup_errors,phase):
    """Best-effort independent JSON/NPZ writes while preserving the primary."""
    partial = dict(status='FAILED',error_type=type(error).__name__,error=str(error),
        method=arm,error_phase=phase,cleanup_errors=cleanup_errors,
        interrupted_continuation_seconds=time.perf_counter()-started,
        reports=reports,finished_queries=done,events=events,
        states=[dict(query=k,n=s['n'],prefix=vars(s['prefix']),
                     registered=s['registered'],reports=s['reports'],
                     fused_su_parts=s.get('fused_su_parts',[]),
                     fused_sv_parts=s.get('fused_sv_parts',[]))
                for k,s in enumerate(states)])
    arrays = {}
    for k,s in enumerate(states):
        for name,dtype in (('positions',np.int64),('fresh',np.bool_),('us',np.float64),('vs',np.float64)):
            key = {'us':'u','vs':'v'}.get(name,name)
            arrays[f'q{k}_{key}'] = (np.concatenate(s[name]) if s[name]
                else np.empty(0,dtype=dtype))
    for suffix,write in (
        ('.partial.json',lambda p:p.write_text(json.dumps(partial,indent=2)+'\n',encoding='utf8')),
        ('.partial.npz',lambda p:np.savez_compressed(p,**arrays))):
        try:
            write(Path(trace_path).with_suffix(suffix))
        except BaseException as secondary:
            error.add_note(f'Failure archive {suffix} also failed: {type(secondary).__name__}: {secondary}')


def _oracle_exact(path, N, h, prefix, specs, seed, args, trace_path):
    """Same audited fused kernel/chunks/fsum and actual-file path, safe teardown.

    The orchestration lives here so even a terminal close exception retains
    all reports and completed fused-chunk moments without changing the base.
    """
    K = len(specs)
    targets,caps = query_contract(args,K)
    started = time.perf_counter()
    store = None
    states = [dict(us=[],vs=[],positions=[],fresh=[],n=0,reports=0,evals=0,
                   prefix=Prefix(),registered=True,fused_su_parts=[],fused_sv_parts=[])
              for _ in specs]
    done,reports = [],[]
    events = [dict(kind='oracle_all_known', epoch=-1, queries=list(range(len(specs))),
                   source_h=h, information='all-functions-known oracle')]
    sampling_seconds = function_seconds = report_seconds = 0.
    phase = 'source_setup'
    try:
        store = ForwardFileStore(path,N,h,prefix=prefix)
        init_seconds = time.perf_counter()-started
        phase = 'fused_continuation'
        t = time.perf_counter()
        store.expose_to(N,return_positions=False)
        centers = np.asarray([s['centers'] for s in specs])
        scales = np.asarray([1/(2*s['bandwidth']**2) for s in specs])
        sampling_seconds += time.perf_counter()-t
        for first in range(0,N,args.exact_chunk):
            last = min(N,first+args.exact_chunk)
            t = time.perf_counter()
            records = store.read_range(first,last)
            sampling_seconds += time.perf_counter()-t
            t = time.perf_counter()
            su,sv = workload.fused_first_moments(records['p'],records['r'],records['q'],
                centers,scales,specs[0]['function']=='weighted_price')
            for k in range(K):
                states[k]['fused_su_parts'].append(float(su[k]))
                states[k]['fused_sv_parts'].append(float(sv[k]))
                states[k]['n'] += last-first
                states[k]['evals'] += last-first
            function_seconds += time.perf_counter()-t
            events.append(dict(kind='fused_chunk',first=first,last=last,queries=list(range(K))))
        t = time.perf_counter()
        for k,s in enumerate(states):
            lo,hi = base.exact_enclosure(math.fsum(s['fused_su_parts']),math.fsum(s['fused_sv_parts']),N)
            target = base.is_target(lo,hi,targets[k])
            s['reports'] = 1
            common = dict(query=k,n=N,lower=lo,upper=hi,target=target,target_relative=targets[k],
                query_cap=caps[k],alpha=args.alpha/K,registration_epoch=-1,admission_wave=0,
                information='all-functions-known oracle',suffix_cap_applies=False)
            reports.append(dict(**common,report=1,physical=N,stage='exact_fused',census=True))
            done.append(dict(**common,contribution_evaluations=N,censored=not target,reports=1,
                finish_reason='exact_census',registration_physical=h,registration_seen=h,
                prefix_rows=0,prefix_moments_json='',draw_positions_sha256='',pairs_sha256='',
                fresh_count=0,source_seen_at_finish=N,suffix_remaining=0,unique_positions=N,
                peak_map_entries=0,map_peak_accounting=MAP_PEAK_ACCOUNTING,trace_sha256=''))
        report_seconds += time.perf_counter()-t
        stats = store.stats
        assert not stats['failed']
        for counter in (stats['setup'],stats['continuation'],stats['prefix_setup']):
            assert counter['read_accounting_complete']
            assert counter['read_exception_calls']==counter['read_error_position_unknown']==0
        phase = 'close'
        store.close()
    except BaseException as error:
        cleanup_errors = _cleanup(store)
        _save_failure(trace_path,'exact_fused',error,started,states,done,reports,events,cleanup_errors,phase)
        raise
    continuation = time.perf_counter()-started
    event_path = Path(trace_path).with_suffix('.events.json')
    event_path.write_text(json.dumps(events, indent=2)+'\n', encoding='utf8')
    event_digest = base.sha(event_path)
    for row in done:
        row['events_sha256'] = event_digest
    result = dict(method='exact_fused',queries=K,targets=sum(r['target'] for r in done),
        censored=sum(r['censored'] for r in done),continuation_seconds=continuation,
        init_seconds=init_seconds,sampling_and_cache_seconds=sampling_seconds,function_seconds=function_seconds,
        prefix_prepare_seconds=0.,report_seconds=report_seconds,contribution_evaluations=N*K,
        scalar_exp_evaluations=N*K*len(specs[0]['centers']),peak_query_map_entries=0,
        peak_map_replay_seconds=0.,diagnostic_seconds=0.,map_peak_accounting=MAP_PEAK_ACCOUNTING,
        peak_map_replay_deletions=0,atomic_pool_proposals=0,
        actual_read_calls=stats['continuation']['read_calls'],
        actual_requested_bytes=stats['continuation']['read_requested_bytes'],
        actual_returned_bytes=stats['continuation']['read_returned_bytes'],
        actual_read_seconds=stats['continuation']['read_seconds'],initial_h=h,
        final_source_prefix=stats['physical_count'],visible_records=stats['seen_count'],
        retained_cache_payload_bytes=stats['retained_cache_payload_bytes'],trace_sha256='',
        backend_stats_json=json.dumps(stats,sort_keys=True),events_sha256=event_digest,
        information='all-functions-known oracle',active_window=K,
        precision_profile=getattr(args,'precision_profile','heterogeneous'),
        target_vector_json=json.dumps(targets),cap_vector_json=json.dumps(caps),
        epoch_count=0,admission_waves=1,cohort_groups=0,cohort_pool_draw_calls=0,
        all_specs_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest())
    return result, done, reports


def run_session(path, N, h, prefix, specs, arm, seed, args, trace_path, *, selection_policy=None):
    """Run one serial arm, then serialize positions/fresh/pairs outside timer.

    Optional API-only participation_schedule maps an epoch index to the query
    IDs participating in that epoch. Unspecified epochs include all active
    queries. It permits paused/unequal histories for atomic/coordinated tests.
    Rebase always processes every active query on every source batch; a cohort
    must include all its living members when its one shared pool is drawn.
    """
    if arm not in ARMS:
        raise ValueError('unknown arm')
    if not 0 <= h <= N or not specs or args.block <= 0 or not 0 < args.alpha < 1:
        raise ValueError('invalid session parameters')
    K = len(specs)
    targets, caps = query_contract(args, K)
    window = K if arm in ORACLES else min(K, getattr(args, 'active_window', 4))
    if window <= 0:
        raise ValueError('positive active_window required')
    if selection_policy is not None and arm in ORACLES:
        raise ValueError('own-history selection is incompatible with future-function oracles')
    if arm == 'exact_fused':
        return _oracle_exact(path, N, h, prefix, specs, seed, args, trace_path)
    information = ('own-history policy' if selection_policy is not None else
                   'all-functions-known oracle' if arm in ORACLES else 'streaming frozen-workload')
    replacement = arm.endswith('_wr') or arm == 'known_wr_oracle'
    indexed = arm.startswith('indexed')
    cohort = arm == 'cohort_wor'
    schedule = getattr(args, 'participation_schedule', None) or {}
    arrivals = getattr(args, 'admission_epochs', None)
    if arrivals is not None:
        arrivals = list(arrivals)
        if (len(arrivals) != K or any(isinstance(x, bool) or int(x) != x or x < 0 for x in arrivals)
                or arrivals != sorted(arrivals) or arrivals[0] != 0):
            raise ValueError('admission_epochs must be K sorted nonnegative integers starting at zero')
        if arm in ORACLES:
            raise ValueError('scripted arrivals require admission-only function access')
    measure_latency = getattr(args, 'measure_query_latency', False)
    started = time.perf_counter()
    states = [dict(us=[], vs=[], positions=[], fresh=[], n=0, reports=0, evals=0,
                   prefix=Prefix(), qid=None, group=None, registered=False,
                   sampling_seconds=0., function_seconds=0., report_seconds=0.,
                   prefix_prepare_seconds=0., peak_map_entries=0)
              for _ in specs]
    store = scan = None
    try:
        store = (IndexedFileStore if indexed else ForwardFileStore)(path,N,h,prefix=prefix)
        history = (store.read_range(max(0,h-4096),h) if selection_policy is not None else None)
        if arm != 'rebase' and not cohort:
            cls = (CoordinatedPositionScan if arm == 'coordinated_wor' else
                   AtomicPositionScan if arm.startswith('atomic') or arm == 'known_wr_oracle'
                   else SequentialPositionScan)
            scan = cls(N,seed=seed+700,replacement=replacement)
            scan.expose(N if indexed else h,return_positions=False)
    except BaseException as error:
        cleanup_errors = _cleanup(store)
        _save_failure(trace_path,arm,error,started,states,[],[],[],cleanup_errors,'source_setup')
        raise
    groups = {}
    active, done, reports, events = [], [], [], []
    next_query = wave = epoch = 0
    sampling_seconds = function_seconds = report_seconds = prefix_seconds = 0.
    selection_seconds = 0.
    proposals = pool_draw_calls = 0
    init_seconds = time.perf_counter()-started

    def current_pairs(s):
        return ((np.concatenate(s['us']), np.concatenate(s['vs'])) if s['us']
                else (_empty(), _empty()))

    def report(k, stage):
        nonlocal report_seconds
        s = states[k]
        t = time.perf_counter()
        u, v = current_pairs(s)
        tr = RatioTrace(u, v, None if replacement else N, s['prefix'], args.alpha/K)
        lo, hi = tr.interval(iterations=24)
        s['reports'] += 1
        target = base.is_target(lo, hi, targets[k])
        census = not replacement and s['prefix'].n+s['n'] == N
        row = dict(query=k, report=s['reports'], n=s['n'], physical=store.physical_rows,
                   seen=store.stats['seen_count'], lower=lo, upper=hi, target=target,
                   target_relative=targets[k], query_cap=caps[k], alpha=args.alpha/K,
                   stage=stage, epoch=epoch, registration_epoch=s['registration_epoch'],
                   admission_wave=s['wave'], prefix_rows=s['prefix'].n,
                   census=census, information=information)
        reports.append(row)
        elapsed = time.perf_counter()-t
        s['report_seconds'] += elapsed
        report_seconds += elapsed
        if target or s['n'] >= caps[k] or census:
            reason = 'target' if target else 'census' if census else 'cap'
            completed = time.perf_counter()-started if measure_latency else None
            done.append(dict(query=k, n=s['n'], contribution_evaluations=s['evals'],
                lower=lo, upper=hi, target=target, censored=not target,
                target_relative=targets[k], query_cap=caps[k], alpha=args.alpha/K,
                reports=s['reports'], finish_reason=reason,
                registration_physical=s['registration_h'],
                registration_seen=s['registration_seen'],
                registration_epoch=s['registration_epoch'], admission_wave=s['wave'],
                prefix_rows=s['prefix'].n, prefix_moments_json='',
                source_seen_at_finish=store.stats['seen_count'],
                suffix_remaining='' if replacement else N-s['prefix'].n-s['n'],
                draw_positions_sha256='', pairs_sha256='', fresh_count=0,
                unique_positions=0, information=information, suffix_cap_applies=True))
            if measure_latency:
                done[-1].update(registration_elapsed_seconds=s['registration_elapsed_seconds'],
                    completion_elapsed_seconds=completed,
                    response_seconds=completed-s['registration_elapsed_seconds'],
                    scheduled_arrival_epoch=arrivals[k] if arrivals is not None else '')
            events.append(dict(kind='finish', epoch=epoch, query=k, n=s['n'],
                               source_h=store.physical_rows, reason=reason))
            if scan is not None:
                scan.stop(s['qid'])
            if cohort:
                g = groups[s['group']]
                g['members'].remove(k)
                if not g['members']:
                    g['scan'].stop(g['qid'])
                    g['stopped'] = True
                    events.append(dict(kind='cohort_stop', epoch=epoch,
                                       group=s['group'], draws=g['draws']))
            return True
        return False

    def admit():
        nonlocal next_query, wave, sampling_seconds, function_seconds, prefix_seconds, selection_seconds
        # A wave is fixed before any registration report/retirement is released.
        incoming = list(range(next_query, min(K, next_query+window-len(active))))
        if arrivals is not None:
            incoming = [k for k in incoming if arrivals[k] <= epoch]
        if not incoming:
            return
        arrival_time = time.perf_counter()-started if measure_latency else None
        this_wave = wave
        wave += 1
        next_query += len(incoming)
        if cohort:
            pool = SequentialPositionScan(N, seed=seed+1700+this_wave, replacement=False)
            pool.expose(store.physical_rows, return_positions=False)
            groups[this_wave] = dict(scan=pool, qid=pool.register(), members=set(incoming),
                                     draws=0, stopped=False)
            events.append(dict(kind='cohort_create', epoch=epoch, group=this_wave,
                               members=incoming.copy(), source_h=store.physical_rows))
        events.append(dict(kind='admit_wave', epoch=epoch, wave=this_wave,
                           queries=incoming.copy(), source_h=store.physical_rows))
        for k in incoming:
            s = states[k]
            if selection_policy is not None:
                t = time.perf_counter()
                specs[k] = selection_policy(history, k, reports)
                events.append(dict(kind='history_selection',epoch=epoch,query=k,
                    source_h=store.physical_rows,context_ids=history['id'].tolist(),
                    policy_report_count=len(reports),
                    spec=specs[k]))
                selection_seconds += time.perf_counter()-t
            s.update(registered=True, registration_h=store.physical_rows,
                     registration_seen=store.stats['seen_count'], registration_epoch=epoch,
                     wave=this_wave, group=this_wave if cohort else None)
            if measure_latency:
                s['registration_elapsed_seconds'] = arrival_time
            if scan is not None:
                s['qid'] = scan.register()
            if arm == 'rebase':
                t = time.perf_counter()
                records = store.read_range(0, s['registration_h'])
                elapsed = time.perf_counter()-t
                sampling_seconds += elapsed
                s['sampling_seconds'] += elapsed
                t = time.perf_counter()
                u, v = workload.evaluate(records, specs[k], validate=False)
                elapsed = time.perf_counter()-t
                function_seconds += elapsed
                s['function_seconds'] += elapsed
                t = time.perf_counter()
                s['prefix'] = workload.prefix_from_pairs(u, v)
                elapsed = time.perf_counter()-t
                prefix_seconds += elapsed
                s['prefix_prepare_seconds'] += elapsed
                s['evals'] += len(records)
            active.append(k)
        # Finish ALL members of this registration wave before refilling.
        retired = [k for k in incoming if report(k, 'registration')]
        active[:] = [k for k in active if k not in retired]

    try:
        while next_query < K or active:
            if next_query < K and len(active) < window:
                admit()
                if not active:
                    if next_query < K and arrivals is not None:
                        events.append(dict(kind='idle_round_skip', epoch=epoch,
                                           next_epoch=arrivals[next_query]))
                        epoch = max(epoch, arrivals[next_query])
                    continue
            participants = list(schedule.get(epoch, active))
            if (not participants or len(set(participants)) != len(participants)
                    or any(k not in active for k in participants)):
                raise ValueError('participation schedule must select nonempty distinct active queries')
            if arm == 'rebase' and set(participants) != set(active):
                raise ValueError('Rebase must process all active queries on each common source batch')
            if cohort:
                for g in groups.values():
                    present = set(participants) & g['members']
                    if present and present != g['members']:
                        raise ValueError('a living cohort shares one stream; cannot pause only one member')
            count = min(args.block, *(caps[k]-states[k]['n'] for k in participants))
            if not replacement:
                count = min(count, *(N-states[k]['prefix'].n-states[k]['n'] for k in participants))
            if arm == 'rebase':
                count = min(count, N-store.physical_rows)
            if count <= 0:
                raise AssertionError('zero-count batch: admission/census/cap retirement failed')
            # This log is recorded before any random draw or payload release.
            events.append(dict(kind='epoch_commit', epoch=epoch, participants=participants.copy(),
                               count=count, source_h=store.physical_rows,
                               progress=[states[k]['n'] for k in participants]))
            t = time.perf_counter()
            if arm == 'rebase':
                before = store.physical_rows
                store.expose_to(before+count, return_positions=False)
                addresses = {k: np.arange(before, before+count, dtype=np.int64) for k in participants}
                flags = {k: np.ones(count, dtype=np.bool_) for k in participants}
                for k in participants:
                    states[k]['positions'].append(addresses[k].copy())
                    states[k]['fresh'].append(flags[k].copy())
                records = store.read_range(before, before+count)
                batches = {k: records for k in participants}
            elif cohort:
                addresses, flags = {}, {}
                for gid in dict.fromkeys(states[k]['group'] for k in participants):
                    g = groups[gid]
                    pool = g['scan']
                    pool.expose(store.physical_rows-pool.physical, return_positions=False)
                    pos, fresh = pool.draw(g['qid'], count)
                    pool_draw_calls += 1
                    g['draws'] += count
                    events.append(dict(kind='cohort_draw',epoch=epoch,group=gid,count=count,
                                       cumulative_draws=g['draws'],members=sorted(g['members']),
                                       source_h=pool.physical))
                    for k in participants:
                        if states[k]['group'] == gid:
                            addresses[k], flags[k] = pos, fresh
                            states[k]['positions'].append(pos.copy())
                            states[k]['fresh'].append(fresh.copy())
                    store.expose_to(pool.physical, return_positions=False)
                pos = np.stack([addresses[k] for k in participants])
                records = store.gather(pos.ravel())
                batches = {k: records[i*count:(i+1)*count] for i,k in enumerate(participants)}
            else:
                qs = [states[k]['qid'] for k in participants]
                if isinstance(scan, AtomicPositionScan):
                    pos, fresh, trials = scan.draw_epochs(qs, count)
                    proposals += int(trials.sum())
                    for i,k in enumerate(participants):
                        states[k]['positions'].append(pos[i].copy())
                        states[k]['fresh'].append(fresh[i].copy())
                else:
                    draws = []
                    for k,q in zip(participants,qs):
                        draw = scan.draw(q,count)
                        draws.append(draw)
                        states[k]['positions'].append(draw[0].copy())
                        states[k]['fresh'].append(draw[1].copy())
                    pos, fresh = np.stack([d[0] for d in draws]), np.stack([d[1] for d in draws])
                if not indexed:
                    store.expose_to(scan.physical, return_positions=False)
                records = store.gather(pos.ravel())
                batches = {k: records[i*count:(i+1)*count] for i,k in enumerate(participants)}
                addresses = {k: pos[i] for i,k in enumerate(participants)}
                flags = {k: fresh[i] for i,k in enumerate(participants)}
            sampling_seconds += time.perf_counter()-t
            events.append(dict(kind='epoch_joint_release', epoch=epoch, participants=participants.copy(),
                               count=count, source_h=store.physical_rows))
            # Every committed query receives and processes its batch before
            # any new query is admitted. Reports may retire unequal histories.
            for k in participants:
                s = states[k]
                t = time.perf_counter()
                u, v = workload.evaluate(batches[k], specs[k], validate=False)
                s['us'].append(u); s['vs'].append(v)
                s['n'] += count; s['evals'] += count
                elapsed = time.perf_counter()-t
                function_seconds += elapsed
                s['function_seconds'] += elapsed
            if selection_policy is not None:
                # Deterministic tag order after the entire committed API call.
                # Only already released payloads enter the next policy context.
                t = time.perf_counter()
                for k in participants:
                    history = np.concatenate((history,batches[k]))[-4096:].copy()
                selection_seconds += time.perf_counter()-t
            retired = [k for k in participants if report(k, 'ordinary')]
            active[:] = [k for k in active if k not in retired]
            if arm == 'rebase':
                assert all(states[k]['prefix'].n+states[k]['n'] == store.physical_rows for k in active)
            epoch += 1
        stats = store.stats
        assert not stats['failed']
        for counter in (stats['setup'], stats['continuation'], stats['prefix_setup']):
            assert counter['read_accounting_complete']
            assert counter['read_exception_calls'] == counter['read_error_position_unknown'] == 0
        assert len(done) == K and len({r['query'] for r in done}) == K
        if cohort:
            assert all(g['stopped'] and not g['members'] for g in groups.values())
        # Teardown is part of the successful continuation and is caught by the
        # same failure boundary as source mutations and report generation.
        store.close()
    except BaseException as error:
        # Preserve even an unsuccessful arm's reports and committed schedule.
        # Its diagnostic serialization is outside its interrupted timer.
        cleanup_errors = _cleanup(store)
        _save_failure(trace_path,arm,error,started,states,done,reports,events,cleanup_errors,'continuation_or_close')
        raise
    continuation = time.perf_counter()-started
    # Hashing, serialization, uniqueness diagnostics, and truth are deferred.
    saved = {}
    for k,s in enumerate(states):
        u,v = current_pairs(s)
        saved[f'q{k}_positions'] = np.concatenate(s['positions']) if s['positions'] else np.empty(0,dtype=np.int64)
        saved[f'q{k}_fresh'] = np.concatenate(s['fresh']) if s['fresh'] else np.empty(0,dtype=np.bool_)
        saved[f'q{k}_u'], saved[f'q{k}_v'] = u,v
    peaks = diagnostics.replay_peaks(N,arm,[saved[f'q{k}_positions'] for k in range(K)],events)
    np.savez_compressed(trace_path, **saved)
    digest = base.sha(trace_path)
    event_path = Path(trace_path).with_suffix('.events.json')
    event_path.write_text(json.dumps(events, indent=2)+'\n', encoding='utf8')
    event_digest = base.sha(event_path)
    for row in done:
        k,s = row['query'], states[row['query']]
        positions = saved[f'q{k}_positions']
        row.update(trace_sha256=digest, events_sha256=event_digest,
                   draw_positions_sha256=base.array_sha(positions),
                   pairs_sha256=hashlib.sha256(saved[f'q{k}_u'].tobytes()+saved[f'q{k}_v'].tobytes()).hexdigest(),
                   fresh_count=int(saved[f'q{k}_fresh'].sum()),
                   unique_positions=len(np.unique(positions)),
                   prefix_moments_json=json.dumps(vars(s['prefix']), sort_keys=True),
                   function_seconds=s['function_seconds'], report_seconds=s['report_seconds'],
                   prefix_prepare_seconds=s['prefix_prepare_seconds'],
                   peak_map_entries=peaks['personal'][k],map_peak_accounting=MAP_PEAK_ACCOUNTING)
        if not replacement:
            assert row['unique_positions'] == row['n']
    result = dict(method=arm, queries=K, targets=sum(r['target'] for r in done),
        censored=sum(r['censored'] for r in done), continuation_seconds=continuation,
        init_seconds=init_seconds, sampling_and_cache_seconds=sampling_seconds,
        function_seconds=function_seconds, prefix_prepare_seconds=prefix_seconds,
        report_seconds=report_seconds,
        selection_seconds=selection_seconds,
        contribution_evaluations=sum(r['contribution_evaluations'] for r in done),
        scalar_exp_evaluations=sum(r['contribution_evaluations'] for r in done)*len(specs[0]['centers']),
        peak_query_map_entries=peaks['peak'],peak_map_replay_seconds=peaks['seconds'],
        diagnostic_seconds=peaks['seconds'],map_peak_accounting=MAP_PEAK_ACCOUNTING,
        peak_map_replay_deletions=peaks['deletions'],atomic_pool_proposals=proposals,
        cohort_pool_draw_calls=pool_draw_calls, cohort_groups=len(groups),
        actual_read_calls=stats['continuation']['read_calls'],
        actual_requested_bytes=stats['continuation']['read_requested_bytes'],
        actual_returned_bytes=stats['continuation']['read_returned_bytes'],
        actual_read_seconds=stats['continuation']['read_seconds'],
        initial_h=h, final_source_prefix=stats['physical_count'], visible_records=stats['seen_count'],
        retained_cache_payload_bytes=stats['retained_cache_payload_bytes'],
        trace_sha256=digest, events_sha256=event_digest, backend_stats_json=json.dumps(stats,sort_keys=True),
        information=information, active_window=window,
        precision_profile=getattr(args, 'precision_profile', 'heterogeneous'),
        target_vector_json=json.dumps(targets), cap_vector_json=json.dumps(caps),
        epoch_count=epoch, admission_waves=wave,
        all_specs_sha256=hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest())
    return result, done, reports


def protocol(args):
    return dict(version='streaming-v2-exact-map-replay-safe-teardown',legacy_v1_driver_sha256=LEGACY_V1_SHA256,
        map_peak_accounting=MAP_PEAK_ACCOUNTING,heterogeneous_precision_factors=list(PRECISION_FACTORS),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        workload='All K function definitions frozen identically from the common initial last4096 '
                 'physical-prefix rows. A controlled workload, not adaptive future-function selection. '
                 'Generic arms only evaluate a function after its streaming registration.',
        admission='Default active-window=4. Fill vacancies only after every committed participant '
                  'has received its batch and all reports/retirements for that epoch are processed. '
                  'Each refill is a new admission wave; per-query histories can differ.',
        precision='Default relative target .10 times cyclic [.5,1,1.5,2]; uniform is available. '
                  'This moderate heterogeneous profile is an explicit postpilot choice following '
                  'the tighter v1 [.25,.5,1,2] profile and .025-target censoring; not preregistration. '
                  'Each query keeps the same target, alpha/K and suffix cap in every arm. '
                  'Exact fused performs N evaluations regardless of suffix cap.',
        epoch='Participant IDs and fixed batch count are committed before sampling; atomic arms '
              'jointly release only after draw_epochs returns. No data-dependent within-epoch admission. '
              'Count=min(block, participating query cap remaining, participating WoR own remaining); '
              'Rebase additionally limits by global source remaining and processes every active query.',
        cohort='One independent SequentialPositionScan WoR pool/stream per admission wave. One draw '
               'per living group, copied to that group active members. Retired members stop receiving '
               'draws; the group pool stops only when its final member exits. Distinct waves have '
               'independent streams and share the actual source exposure ledger.',
        coordinated='CoordinatedPositionScan uses separate shared thinning-gate and cache tapes, '
                    'with one independent WoR pool per query and support for unequal query progress. '
                    'atomic_wor is the old independent-cache comparator.',
        rebase='At each registration evaluate that query on the actual entire current contiguous '
               'physical source prefix and compile compensated moments. All subsequent common source '
               'batches go to all active queries. Registrations after EOF get complete-prefix n=0 '
               'census reports, with no zero-count sampling loop.',
        oracles='exact_fused and known_wr_oracle know all K functions and activate them initially. '
                'exact_fused uses the same strong fused one-row-pass actual-file kernel/chunks/'
                'compensated chunk moments as the audited base, with driver-local safe teardown. known_wr_oracle '
                'uses a shared WR stream. Their stronger information excludes them from any '
                'unknown-future-condition theoretical minimum comparison.',
        io='Headerless 32-byte records in a seeded uniform materialized permutation. Counters are '
           'actual unbuffered FileIO read API bytes; warm/uncontrolled OS cache, no device-I/O claim. '
           'Indexed scan fresh flags refer to a full-N virtual cursor (therefore false), while the '
           'file-store ledger separately records actual visible IDs and contiguous physical prefix.',
        costs='All arm jobs serial. Full continuation timer includes initialization, policy, all '
              'reports and store teardown. Position/pair hashes, serialization, uniqueness and '
              'offline truth outside that timer. Sparse-map peaks use a separate exact replay of '
              'retained positions/events after continuation, with its own seconds column: sequential '
              'full-query batches, atomic/coordinated microepochs and each delete, cohort single-pool '
              'draw order, sum of every living pool including paused queries. Common costs deduplicated: source validation/'
              'permutation/materialization once per dataset/replicate; prefix/context once per '
              'dataset/replicate/fraction; selection once per function and that prefix.',
        retention='Every intermediate and final report including failures/censors is retained; '
                  'all generic query positions, private fresh flags, float64 pairs, actual prefix '
                  'moments, admission/commit/release/finish events, physical sources and code snapshots '
                  'retained. Exceptions leave INCOMPLETE and failure.json.',
        inference='Ideal uniform-word marginal reasoning; deterministic finite PCG64 implementation. '
                  'Own-query CS with fixed alpha/K, no independence or product-capital claim.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', type=Path, default=ROOT/'data')
    ap.add_argument('--datasets', nargs='+', default=['tpch','tpcds'])
    ap.add_argument('--fractions', nargs='+', type=float, default=[.1,.9])
    ap.add_argument('--functions', nargs='+', choices=['affinity','weighted_price'], default=['affinity','weighted_price'])
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=list(DEFAULT_ARMS))
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=202610173000)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--active-window', type=int, default=4)
    ap.add_argument('--centers', type=int, default=8)
    ap.add_argument('--block', type=int, default=4096)
    ap.add_argument('--cap', type=int, default=65536)
    ap.add_argument('--query-caps', nargs='+', type=int)
    ap.add_argument('--target', type=float, default=.10)
    ap.add_argument('--precision-profile', choices=['uniform','heterogeneous'], default='heterogeneous')
    ap.add_argument('--alpha', type=float, default=.05)
    ap.add_argument('--exact-chunk', type=int, default=65536)
    ap.add_argument('--synthetic-n', type=int, default=0)
    args = ap.parse_args()
    if (not all(0 < f < 1 for f in args.fractions) or
            any(x <= 0 for x in (args.reps,args.K,args.active_window,args.centers,args.block,args.cap,args.exact_chunk))
            or not 0 < args.alpha < 1 or args.synthetic_n < 0):
        raise ValueError('invalid positive experiment arguments')
    if len(set(args.arms)) != len(args.arms) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError('duplicate arms/datasets would overwrite traces')
    if len(set(args.functions)) != len(args.functions) or len(set(args.fractions)) != len(args.fractions):
        raise ValueError('duplicate functions/fractions would overwrite traces')
    query_contract(args,args.K)
    out = (ROOT/args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out/'INCOMPLETE').write_text('Study is not complete until manifest.json.\n', encoding='utf8')
    for name in ('source','physical_sources','traces'):
        (out/name).mkdir()
    sources = {name:base.sha(ROOT/name) for name in SOURCE_NAMES}
    if sources['code/exp_file_controlled.py'] != BASE_EXPECTED_SHA256:
        raise RuntimeError('audited base driver changed; review before using this driver')
    for name in SOURCE_NAMES:
        target = out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    (out/'sources_before.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf8')
    (out/'protocol.json').write_text(json.dumps(protocol(args),indent=2)+'\n',encoding='utf8')
    try:
        _main_run(args,out,sources)
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error)),indent=2)+'\n',encoding='utf8')
        (out/'sources_after.json').write_text(json.dumps(
            {name:base.sha(ROOT/name) for name in sources},indent=2)+'\n',encoding='utf8')
        raise


def _main_run(args,out,sources):
    t = time.perf_counter()
    workload.warmup()
    diagnostics.warmup()
    for cls in (SequentialPositionScan,AtomicPositionScan,CoordinatedPositionScan):
        for wr in (False,True):
            scan = cls(16,seed=33,replacement=wr)
            scan.expose(4,return_positions=False)
            qs = [scan.register() for _ in range(2)]
            if isinstance(scan,AtomicPositionScan):
                scan.draw_epochs(qs,2)
            else:
                scan.draw(qs[0],2)
    RatioTrace(np.full(8,.5),np.ones(8),None,alpha=args.alpha/args.K).interval(iterations=24)
    jit_seconds = time.perf_counter()-t
    sessions, queries, reports, source_preps, prefix_preps, selection_preps, spec_rows = [],[],[],[],[],[],[]
    loads,audits = {},{}
    base.original_loader.DATA_ROOT = args.data_root
    tables = dict(sessions=sessions,queries=queries,paths=reports,source_preparation=source_preps,
                  prefix_preparation=prefix_preps,selection_preparation=selection_preps,specs=spec_rows)

    def save_tables():
        for name,rows in tables.items():
            write_csv(out/f'{name}.csv',rows)

    for dataset in args.datasets:
        if args.synthetic_n:
            N = args.synthetic_n
            if N < 2:
                raise ValueError('synthetic N must exceed one')
            rng = np.random.default_rng(831)
            data = tuple(rng.random(N) for _ in range(3))
            loads[dataset] = dict(population=N,synthetic=True,seed=831,
                normalized_arrays_sha256=[base.array_sha(a) for a in data])
        else:
            data,loads[dataset] = base.original_loader.load(dataset)
            N = len(data[0])
        all_specs = {}
        for rep in range(args.reps):
            seed = args.seed+rep
            source_path = out/'physical_sources'/f'{dataset}_{rep}.bin'
            prep = base.physical_file(source_path,data,seed)
            source_preps.append(dict(dataset=dataset,replicate=rep,**prep))
            for fraction in args.fractions:
                h = int(N*fraction)
                if h <= 0:
                    raise ValueError('initial prefix must contain a row')
                t = time.perf_counter()
                prefix = prepare_prefix(source_path,N,h)
                prefix_seconds = time.perf_counter()-t
                t = time.perf_counter()
                with ForwardFileStore(source_path,N,h,prefix=prefix) as store:
                    context = store.read_range(max(0,h-4096),h)
                    context_stats = store.stats
                prefix_preps.append(dict(dataset=dataset,replicate=rep,fraction=fraction,initial_h=h,
                    common_prefix_seconds=prefix_seconds,common_prefix_stats_json=json.dumps(prefix.stats,sort_keys=True),
                    common_context_seconds=time.perf_counter()-t,common_context_stats_json=json.dumps(context_stats,sort_keys=True)))
                for function in args.functions:
                    t = time.perf_counter()
                    specs = workload.select_specs(context,args.K,args.centers,function)
                    selection_preps.append(dict(dataset=dataset,replicate=rep,fraction=fraction,
                        function=function,common_selection_seconds=time.perf_counter()-t))
                    targets,caps = query_contract(args,args.K)
                    for spec in specs:
                        k = spec['query']
                        all_specs[(function,fraction,rep,k)] = spec
                        spec_rows.append(dict(dataset=dataset,function=function,fraction=fraction,replicate=rep,
                            query=k,target_relative=targets[k],query_cap=caps[k],alpha=args.alpha/args.K,
                            spec_json=json.dumps(spec,sort_keys=True)))
                    for arm in np.roll(np.asarray(args.arms),(rep+int(fraction*10))%len(args.arms)):
                        trace_path = out/'traces'/f'{dataset}_{function}_{fraction}_{rep}_{arm}.npz'
                        try:
                            result,qq,pp = run_session(source_path,N,h,prefix,specs,str(arm),seed,args,trace_path)
                        except BaseException:
                            save_tables()
                            raise
                        for row in [result,*qq,*pp]:
                            row.update(dataset=dataset,function=function,fraction=fraction,replicate=rep,source_seed=seed,method=str(arm))
                        sessions.append(result);queries.extend(qq);reports.extend(pp)
                        save_tables()
                    print(dataset,rep,fraction,function,'streaming timed arms complete',flush=True)
                del prefix,context
        audits[dataset] = base.truth_audit(data,all_specs,
            [r for r in queries if r['dataset']==dataset],
            [r for r in reports if r['dataset']==dataset],out/'physical_sources',dataset)
        save_tables()
        del data
    after = {name:base.sha(ROOT/name) for name in sources}
    physical_after = {p.name:base.sha(p) for p in (out/'physical_sources').glob('*.bin')}
    (out/'sources_after.json').write_text(json.dumps(after,indent=2)+'\n',encoding='utf8')
    assert sources == after
    assert all(physical_after[f"{r['dataset']}_{r['replicate']}.bin"] == r['source_sha256'] for r in source_preps)
    meta = dict(status='COMPLETE',protocol_version='streaming-v2-exact-map-replay-safe-teardown',
        map_peak_accounting=MAP_PEAK_ACCOUNTING,
        legacy_v1_driver_sha256=LEGACY_V1_SHA256,protocol_sha256=base.sha(out/'protocol.json'),source_sha256=sources,
        source_after_sha256=after,physical_source_after_sha256=physical_after,input=loads,audit=audits,
        jit_preparation_seconds=jit_seconds,sessions=len(sessions),queries=len(queries),all_reports=len(reports),
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        output_sha256={p.relative_to(out).as_posix():base.sha(p) for p in sorted(out.rglob('*'))
                       if p.is_file() and p.name!='INCOMPLETE'},
        note='Serial actual-file synthetic/integration or real study as arguments declare; no cold-cache or device-I/O claim.')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    (out/'INCOMPLETE').unlink()
    print('COMPLETE',len(sessions),len(queries),len(reports),flush=True)


if __name__ == '__main__':
    main()
