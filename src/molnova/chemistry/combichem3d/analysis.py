"""Human-readable appendices from saved evidence; never changes scientific state."""
import copy
import re
from rdkit import Chem
from molnova import database
from molnova.chemistry.combichem3d.attachment_policy import attachment_labels, root_for_site
from molnova.chemistry.combichem3d.logging_utils import logger
from molnova.chemistry.scaffold import mcs_readable_smarts


def attachment_display(mol, history, sites):
    """User-requested CH*/c* display notation, not computational SMILES."""
    marked=set(sites)
    copy_mol=Chem.Mol(mol)
    by_index={a['index']:a['uid'] for a in history['atoms']}
    hydrogens={}
    for atom in copy_mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if by_index.get(atom.GetIdx()) in marked:
            number=atom.GetIdx()+1
            atom.SetAtomMapNum(number)
            hydrogens[number]=atom.GetTotalNumHs(includeNeighbors=True)
    text=Chem.MolToSmiles(Chem.RemoveHs(copy_mol),isomericSmiles=True)
    def replace(match):
        token,number=match.group(1),int(match.group(2))
        count=hydrogens[number]
        if count<1:
            return match.group(0)
        # Mark replacing exactly one H, preserving isotope/charge/stereo text.
        token=re.sub(r'H\d*','',token,count=1)
        remaining=count-1
        suffix=('H'+(str(remaining) if remaining>1 else '')) if remaining else ''
        # The accepted display convention places * immediately before :number.
        return f'[{token}{suffix}*:{number}]'
    return re.sub(r'\[([^\]]*):(\d+)\]',replace,text)


def final_analysis_lines(db):
    snapshot=database.growth_analysis_snapshot(db)
    run=snapshot['run'];cfg=run['settings']
    frames={r['id']:r for r in snapshot['frames']}
    counts={}
    def numeric(value):
        return f'{value:.3f}' if value is not None else 'n/a'
    for row in snapshot['counts']:
        counts.setdefault(row['iteration'],{})[row['status']]=row['n']
    lines=['===== 실행 결과 분석 BEGIN =====',
           f"종료 사유={run['stop_reason']} 계산시간={run['elapsed']:.2f}s "
           f"시도={run['attempted']} 통과_고유분자={run['unique_count']}",
           f"설정: K={cfg['sidechains-per-site']} fragment_heavy_atoms={cfg['sidechain-min-heavy-atoms']}..{cfg['sidechain-max-heavy-atoms']} "
           f"optimization_iterations={cfg['optimization-iterations']} max_seconds={cfg['max-seconds']}",
           '번호는 각 프레임의 1-based 원자 번호. CH*/c* 표기는 부착점 표시용이며 계산 입력과 별개.',
           'PASS는 유한 탐색에서 구조/충돌/설정 필터 통과를 뜻하며 수렴·결합친화도를 보증하지 않음.']
    if not cfg.get('filters'):
        lines.append('전체 분자 물성 필터 없음: 개별 fragment 원자수 제한은 최종 분자의 크기 제한이 아님.')
    initial=snapshot['frames'][0]['history'] if snapshot['frames'] else {}
    labels=initial.get('initial_attachment_labels')
    if labels is None and initial.get('attachments'):
        query=Chem.MolFromSmarts(cfg.get('reference-mcs') or '')
        names=attachment_labels(query) if query is not None else []
        if len(names)==len(initial['attachments']):
            labels=dict(zip(names,initial['attachments']))
            lines.append('구버전 실행: 초기 부착점 라벨/branch 대응은 저장 reference-mcs와 초기 attachment 순서에서 복원한 표시용 정보.')
    for iteration in sorted(counts):
        c=counts[iteration]
        lines.append(f"CYCLE {iteration} 결과: 시도={sum(c.values())} PASS={c.get('PASS',0)} "
                     f"CLASH_REJECT={c.get('CLASH_REJECT',0)} UNRESOLVED={c.get('UNRESOLVED',0)}")
        for row in snapshot['optimization']:
            if row['iteration']==iteration:
                lines.append((f"  최적화: 수렴={row['converged']} 반복한도={row['iteration_limit']} "
                              f"통과했으나_미수렴={row['passed_unconverged']}") if row['recorded'] else
                             '  최적화: 구버전 수렴 상태 집계 기록 없음')
        for row in snapshot['properties']:
            if row['iteration']==iteration:
                lines.append(f"  출력 통과: 고유분자={row['unique_count']} 포즈={row['n']} "
                             f"MW={numeric(row['mw_min'])}..{numeric(row['mw_max'])} "
                             f"cLogP={numeric(row['clogp_min'])}..{numeric(row['clogp_max'])} "
                             f"QED={numeric(row['qed_min'])}..{numeric(row['qed_max'])}")
        for row in snapshot['errors']:
            if row['iteration']==iteration:
                lines.append(f"  미해결 오류(분자수, 중복가능): {row['error']} n={row['n']}")
    for cycle in snapshot['cycles']:
        report=cycle['report'];number=cycle['iteration']
        row=frames.get(cycle['frame_pose'])
        if row is None:
            prepared=report.get('prepared_frame')
            if prepared is None:
                lines.append(f"CYCLE {number} FRAME 준비상태={report.get('status',cycle['state'])}")
                continue
            row=dict(prepared,history=prepared['history'],id=None,compound_id=None)
        h=copy.deepcopy(row['history'])
        if 'initial_attachment_labels' not in h and labels:
            h['initial_attachment_labels']=labels
            h['branch_roots']={f'branch_{i+1}':label for i,label in enumerate(labels)}
        mol=Chem.MolFromMolBlock(row['molblock'],removeHs=False)
        if mol is None:
            raise ValueError('Cannot reconstruct frame for final analysis')
        lines.extend([f"CYCLE {number} FRAME CMPID_{row['compound_id']} pose={row['id']} "
                      f"status={report.get('status',cycle['state'])}",
                      '  FRAME_SMILES='+Chem.MolToSmiles(Chem.RemoveHs(mol),isomericSmiles=True),
                      '  FRAME_ATTACH_DISPLAY='+attachment_display(mol,h,h.get('attachments',[]))])
        smarts=h.get('frame_smarts',h.get('core_smarts',''))
        lines.extend(['  FRAME MCS SMARTS: '+smarts,'  FRAME MCS SMARTS readable: '+mcs_readable_smarts(smarts)])
        if report.get('attempts'):
            lines.append('  MCS 축소='+ ' -> '.join(f"{a['elite_count']}:{a['status']}" for a in report['attempts']))
            final=report['attempts'][-1]
            for pose,smiles in zip(final.get('elite_pose_ids',[]),final.get('elite_smiles',[])):
                lines.append(f'  ELITE pose={pose} SMILES={smiles}')
        if h.get('initial_attachment_labels'):
            lines.append('  초기 부착점 표기='+h.get('initial_attachment_notation','legacy_labels'))
            for label,uid in h['initial_attachment_labels'].items():
                used=h.get('growth_counts',{}).get(label)
                limit=h.get('initial_attachment_max_cycles',{}).get(label,999)
                lines.append(f"  ROOT {label}: initial_uid={uid} growth_cycles_used={used if used is not None else '이전 기록 없음'} "
                             f"max_cycles={'unlimited' if limit==999 else limit} "
                             f"input_map_value={h.get('initial_attachment_map_values',{}).get(label,'이전 기록 없음')} "
                             f"cumulative_selected_sites={h.get('attachment_selection_counts',{}).get(label,'이전 기록 없음')}")
        probes=report.get('probes',{})
        if number==1:
            probes={r['site']:r['report'] for r in snapshot['probes'] if r['parent_pose']==cycle['frame_pose']}
        selected={a['uid'] for a in report.get('selected_attachments',[])} if number>1 else set(h.get('attachments',[]))
        lines.append('  선택 규칙='+report.get('selection_policy','initial_input' if number==1 else 'legacy_global_distance'))
        atoms={a['uid']:a for a in h['atoms']}
        lines.append(f"  부착점 검사: 후보={len(probes)} PASS={sum(p['status']=='PASS' for p in probes.values())} "
                     f"CLASH_REJECT={sum(p['status']=='CLASH_REJECT' for p in probes.values())} "
                     f"UNRESOLVED={sum(p['status']=='UNRESOLVED' for p in probes.values())} "
                     f"선택={len(selected)} 한도로_검사제외={len(report.get('blocked_attachments',[]))}")
        for uid,p in sorted(probes.items(),key=lambda item:(-item[1].get('bond_distance',0),item[0])):
            atom=atoms[uid]
            try:root=root_for_site(h,uid)
            except ValueError:root='unknown'
            clash=p.get('clashes') or {}
            lines.append(f"  ATTACH atom={atom['index']+1} root={root} branch={atom['branch']} "
                         f"bond_distance={p.get('bond_distance','n/a')} status={p['status']} "
                         f"selected={uid in selected} phase={p.get('phase')} "
                         f"reported_pose_receptor_clashes={clash.get('receptor_clashes')} "
                         f"reported_pose_internal_clashes={clash.get('internal_clashes')}")
        for status in ('PASS','CLASH_REJECT','UNRESOLVED'):
            sites=[uid for uid,p in probes.items() if p['status']==status]
            lines.append(f'  {status}_ATTACH_DISPLAY='+attachment_display(mol,h,sites))
        lines.append('  SELECTED_ATTACH_DISPLAY='+attachment_display(mol,h,selected))
        for blocked in report.get('blocked_attachments',[]):
            lines.append(f"  CYCLE_LIMIT_BLOCKED atom={atoms[blocked['uid']]['index']+1} "
                         f"root={blocked['root_attachment']} used={blocked['growth_cycles_used']} max={blocked['max_cycles']}")
        if report.get('blocked_attachments'):
            lines.append('  CYCLE_LIMIT_BLOCKED_ATTACH_DISPLAY='+attachment_display(
                mol,h,[r['uid'] for r in report['blocked_attachments']]))
        lines.append('  충돌 수는 저장된 마지막/통과 포즈의 기록이며 CLASH_REJECT는 설정된 유한 탐색의 결과.')
    lines.append('===== 실행 결과 분석 END =====')
    return lines


def log_final_analysis(db):
    lines=final_analysis_lines(db)
    logger.info('\n%s','\n'.join(lines))
    return lines
