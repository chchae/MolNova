"""Initial-attachment ancestry, growth budgets and balanced site selection."""


def attachment_labels(query):
    """Legacy labels, also used to explain historical run records."""
    dummies=[a for a in query.GetAtoms() if a.GetAtomicNum()==0]
    labels=[str(a.GetAtomMapNum() if a.HasProp('molAtomMapNumber') else i)
            for i,a in enumerate(dummies)]
    if len(labels)!=len(set(labels)):
        raise ValueError('Initial attachment labels must be distinct')
    return labels


def initial_attachment_spec(query, legacy_limits=None):
    """Decode [*:N] growth budgets with separate stable ordinal root IDs.

    An explicit legacy map or a :0 dummy keeps the old label interpretation.
    This avoids silently changing established :0/:1 artificial projects.
    """
    dummies=[a for a in query.GetAtoms() if a.GetAtomicNum()==0]
    legacy=bool(legacy_limits) or any(a.HasProp('molAtomMapNumber') and a.GetAtomMapNum()==0 for a in dummies)
    if legacy:
        labels=attachment_labels(query)
        limits=dict(legacy_limits or {})
        unknown=set(limits)-set(labels)
        if unknown:
            raise ValueError('Cycle limits reference absent initial attachment labels: '+', '.join(sorted(unknown)))
        mode='legacy_labels'
    else:
        labels=[str(i) for i in range(len(dummies))]
        limits={label:a.GetAtomMapNum() if a.HasProp('molAtomMapNumber') else 999
                for label,a in zip(labels,dummies)}
        if any(n<1 for n in limits.values()):
            raise ValueError('Initial attachment cycle limits must be positive; 999 means unlimited')
        mode='map_cycle_limits'
    return labels,limits,mode


def root_for_site(history, uid):
    for label,anchor in history.get('initial_attachment_labels',{}).items():
        if uid==anchor:
            return label
    atom=next((a for a in history['atoms'] if a['uid']==uid),None)
    if atom is None:
        raise ValueError(f'Unknown attachment atom UID: {uid}')
    root=history.get('branch_roots',{}).get(atom['branch'])
    if root is None:
        raise ValueError(f'Missing initial-attachment ancestry for {uid}')
    return root


def growth_available(history, root):
    limit=history.get('initial_attachment_max_cycles',{}).get(root,999)
    return limit==999 or history.get('growth_counts',{}).get(root,0)<limit


def balanced_sites(passing, history, count):
    """Cover active roots first; rotate spare slots by cumulative assignments."""
    groups={}
    for uid,distance in passing:
        root=root_for_site(history,uid)
        if growth_available(history,root):
            groups.setdefault(root,[]).append((uid,distance))
    for sites in groups.values():
        sites.sort(key=lambda item:(-item[1],item[0]))
    used={root:0 for root in groups}
    lifetime=history.get('attachment_selection_counts',{})
    selected=[]
    while len(selected)<count and groups:
        root=min(groups,key=lambda r:(used[r],lifetime.get(r,0),-groups[r][0][1],int(r)))
        selected.append(groups[root].pop(0))
        used[root]+=1
        if not groups[root]:
            del groups[root]
    return selected
