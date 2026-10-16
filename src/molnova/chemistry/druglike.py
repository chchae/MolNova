"""User-specified, input-order-preserving druglikeness preselection."""


def filter_druglike(rows):
    """Keep RO5<=1 and the best ceil(N/2) by QED descending, SA ascending."""
    target_count = (len(rows) + 1) // 2
    candidates = [i for i, row in enumerate(rows) if row['ro5_violations'] <= 1]
    candidates.sort(key=lambda i: (-rows[i]['qed'], rows[i]['sa_score']))
    selected = set(candidates[:target_count])
    return [
        {**row, 'druglike_selected': i in selected,
         'druglike_reject_reason': ('ro5_violations' if row['ro5_violations'] > 1
                                   else 'qed_rank' if i not in selected else None)}
        for i, row in enumerate(rows)
    ]
