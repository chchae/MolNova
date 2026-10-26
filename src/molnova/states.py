from enum import StrEnum


class CompoundState(StrEnum):
    REFERENCE = "reference"
    GENERATED = "generated"
    DRUGLIKE_REJECTED = "druglike_rejected"
    SA_REJECTED = "sa_rejected"
    SYNTHETIC_RUNNING = "synthetic_running"
    LIGPREP_RUNNING = "ligprep_running"
    LIGPREPPED = "ligprepped"
    GLIDE_RUNNING = "glide_running"
    DOCKED = "docked"
    AMBER_RUNNING = "amber_running"
    PRIME_GBSA_RUNNING = "prime_gbsa_running"
    PRIME_GBSA_DONE = "prime_gbsa_done"
    MD_GBSA_RUNNING = "md_gbsa_running"
    MD_GBSA_DONE = "md_gbsa_done"
    FEP_RUNNING = "fep_running"
    FEP_DONE = "fep_done"
    FAILED = "failed"


class PoseVerdict(StrEnum):
    PASS = 'PASS'
    CLASH_REJECT = 'CLASH_REJECT'
    UNRESOLVED = 'UNRESOLVED'


class GrowthTaskState(StrEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    READY = 'ready'
    DONE = 'done'
