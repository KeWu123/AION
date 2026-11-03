"""Post-timing exact sparse-map replay; consumes positions, never RNG words.

Replay follows the driver's actual mutation order. It does not simulate a new
sample or estimate a peak from batch endpoints. Entry counts mean the sum of
the forward and inverse sparse dictionaries, as in query_entries().
"""
import time
import numpy as np
from numba import njit, types
from numba.typed import Dict, List
from anq.position_scan import _delete, _DICT_TYPE


@njit(cache=False)
def _replay(size, pool_for_query, codes, identities, values):
    K = len(pool_for_query)
    forwards = List.empty_list(_DICT_TYPE)
    inverses = List.empty_list(_DICT_TYPE)
    for _ in range(K):
        forwards.append(Dict.empty(types.int64, types.int64))
        inverses.append(Dict.empty(types.int64, types.int64))
    remaining = np.full(K, size, dtype=np.int64)
    entries = np.zeros(K, dtype=np.int64)
    owners = np.zeros(K, dtype=np.int64)
    alive = np.zeros(K, dtype=np.bool_)
    personal = np.zeros(K, dtype=np.int64)
    total = peak = deleted = 0
    for i in range(len(codes)):
        identity = identities[i]
        if codes[i] == 1:  # Admit a query to its own or its cohort's pool.
            if alive[identity]:
                raise ValueError('duplicate replay admission')
            alive[identity] = True
            pool = pool_for_query[identity]
            owners[pool] += 1
            personal[identity] = max(personal[identity], entries[pool])
        elif codes[i] == 2:  # Retire one query; only the final owner drops pool.
            if not alive[identity]:
                raise ValueError('replay finish before admission')
            alive[identity] = False
            pool = pool_for_query[identity]
            owners[pool] -= 1
            if owners[pool] == 0:
                total -= entries[pool]
                entries[pool] = 0
                forwards[pool].clear()
                inverses[pool].clear()
        else:  # One actual swap-delete, including census's shrinking tail.
            pool = identity
            if owners[pool] == 0 or remaining[pool] <= 0:
                raise ValueError('delete from stopped/exhausted replay pool')
            _delete(forwards[pool], inverses[pool], remaining[pool], values[i])
            remaining[pool] -= 1
            now = len(forwards[pool])+len(inverses[pool])
            total += now-entries[pool]
            entries[pool] = now
            peak = max(peak, total)
            for query in range(K):
                if alive[query] and pool_for_query[query] == pool:
                    personal[query] = max(personal[query], now)
            deleted += 1
    if np.any(alive) or total:
        raise ValueError('successful replay has unfinished living pools')
    return peak, personal, deleted


def replay_peaks(size, arm, positions, events):
    """Exact completed-run peaks with registration/finish and draw ordering.

    Sequential/indexed: one complete query batch before the next query.
    Atomic/coordinated: microepoch order, each participating pool's delete.
    Cohort: one pool per admission wave in the actual cohort_draw event order.
    Every pool stays in the global sum while living, even when paused.
    """
    started = time.perf_counter()
    K = len(positions)
    if arm.endswith('_wr') or arm in ('known_wr_oracle', 'rebase', 'exact_fused'):
        return dict(peak=0, personal=[0]*K, deletions=0,
                    seconds=time.perf_counter()-started)
    capacity = sum(len(x) for x in positions)+2*K
    codes = np.empty(capacity,dtype=np.int8)
    identities = np.empty(capacity,dtype=np.int64)
    values = np.empty(capacity,dtype=np.int64)
    pool_for_query = np.arange(K,dtype=np.int64)
    offsets = np.zeros(K,dtype=np.int64)
    used = 0
    atomic = arm in ('atomic_wor','coordinated_wor')
    cohort = arm == 'cohort_wor'

    def operation(code,identity,value=0):
        nonlocal used
        codes[used],identities[used],values[used] = code,identity,value
        used += 1

    for event in events:
        kind = event['kind']
        if kind == 'admit_wave':
            for query in event['queries']:
                if cohort:
                    pool_for_query[query] = event['wave']
                operation(1,query)
        elif kind == 'finish':
            operation(2,event['query'])
        elif kind == 'cohort_draw' and cohort:
            members,count = event['members'],event['count']
            representative = members[0]
            start = int(offsets[representative])
            batch = positions[representative][start:start+count]
            if len(batch) != count:
                raise ValueError('cohort replay exceeds retained positions')
            for query in members:
                first = int(offsets[query])
                if not np.array_equal(batch,positions[query][first:first+count]):
                    raise ValueError('cohort members do not share retained positions')
                offsets[query] += count
            for value in batch:
                operation(3,event['group'],int(value))
        elif kind == 'epoch_joint_release' and not cohort:
            participants,count = event['participants'],event['count']
            batches = []
            for query in participants:
                start = int(offsets[query])
                batch = positions[query][start:start+count]
                if len(batch) != count:
                    raise ValueError('replay exceeds retained query positions')
                batches.append(batch)
                offsets[query] += count
            if atomic:
                for step in range(count):
                    for query,batch in zip(participants,batches):
                        operation(3,query,int(batch[step]))
            else:
                for query,batch in zip(participants,batches):
                    for value in batch:
                        operation(3,query,int(value))
    if any(offsets[k] != len(positions[k]) for k in range(K)):
        raise ValueError('events do not consume every retained query position')
    peak,personal,deletions = _replay(size,pool_for_query,codes[:used],identities[:used],values[:used])
    return dict(peak=int(peak),personal=personal.tolist(),deletions=int(deletions),
                seconds=time.perf_counter()-started)


def warmup():
    _replay(2,np.asarray([0],dtype=np.int64),np.asarray([1,3,2],dtype=np.int8),
            np.asarray([0,0,0],dtype=np.int64),np.asarray([0,0,0],dtype=np.int64))
