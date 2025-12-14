from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Building:
    name: str
    build_time: int
    earning_per_unit: int

# buildings per the PDF
BUILDINGS = [
    Building("T", 5, 1500),  # Theatre
    Building("P", 4, 1000),  # Pub
    Building("C", 10, 2000)  # Commercial Park
]

def max_profit_schedule(total_time: int) -> dict:
    """
    Returns dict { 'profit': int, 'schedule': [(time_complete, Building)], 'counts': {'T':n,...} }
    DP on time t: dp[t] = max profit achievable in t units (building sequentially).
    Recurrence: dp[t] = max(dp[t-1], max_{b: build_time<=t}(dp[t-b.build_time] + b.earning_per_unit*(total_time - t)))
    We compute dp for t from 0..total_time.
    """
    if total_time <= 0:
        return {"profit": 0, "schedule": [], "counts": {"T":0,"P":0,"C":0}}

    T = total_time
    dp = [0] * (T + 1)
    choice = [None] * (T + 1)  # store chosen Building for finishing at t

    for t in range(1, T + 1):
        # option: do nothing at t (carry from t-1)
        dp[t] = dp[t-1]
        choice[t] = None
        for b in BUILDINGS:
            d = b.build_time
            if d <= t:
                # finishing at time t gives this contribution:
                contribution = b.earning_per_unit * (T - t)
                candidate = dp[t - d] + contribution
                if candidate > dp[t]:
                    dp[t] = candidate
                    choice[t] = b

    # Reconstruct ALL optimal schedules using recursion
    solutions = []
    
    def reconstruct(current_t, current_schedule):
        if current_t == 0:
            solutions.append(list(current_schedule))
            return

        # Check all possible previous states that could lead to dp[current_t]
        
        # Option 1: Do nothing at current_t (carried from current_t-1)
        # Verify if dp[current_t] == dp[current_t-1]
        if dp[current_t] == dp[current_t-1]:
            reconstruct(current_t - 1, current_schedule)
            
        # Option 2: Build a building ending at current_t
        for b in BUILDINGS:
            d = b.build_time
            if d <= current_t:
                contribution = b.earning_per_unit * (T - current_t)
                prev_val = dp[current_t - d]
                if prev_val + contribution == dp[current_t]:
                    # This building is a valid part of an optimal path
                    # Append (start, finish, b)
                    new_sched = current_schedule + [(current_t - d, current_t, b)]
                    reconstruct(current_t - d, new_sched)

    reconstruct(T, [])
    
    # Post-process solutions
    final_results = []
    # Use set of tuples to dedup if multiple paths lead to same schedule (order insensitive? No, specific times matter)
    # Actually, the recursion separates paths by time steps, so we might get duplicate logical sets but unique time-sequences.
    # User likely wants distinct sets of buildings or specific schedules.
    # Let's return distinct schedules.
    
    unique_schedules = []
    seen_hashes = set()

    for sched in solutions:
        sched.sort(key=lambda x: x[0]) # Enforce chronological order
        # Create a hashable representation to dedup
        sched_tuple = tuple((s, f, b.name) for s, f, b in sched)
        if sched_tuple not in seen_hashes:
            seen_hashes.add(sched_tuple)
            unique_schedules.append(sched)

    results_formatted = []
    for sched in unique_schedules:
        counts = {"T":0,"P":0,"C":0}
        for _, _, b in sched:
            counts[b.name] += 1
        results_formatted.append({"schedule": sched, "counts": counts})
        
    return {"profit": dp[T], "solutions": results_formatted}

if __name__ == "__main__":
    print(max_profit_schedule(7))   # profit 3000, T:1
    print(max_profit_schedule(8))   # profit 4500, T:1
    print(max_profit_schedule(13))  # profit 16500, T:2
