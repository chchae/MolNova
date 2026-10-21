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
    PRIMG_GBSA_RUNNING = "primg_gbsa_running"
    PRIMG_GBSA_DONE = "primg_gbsa_done"
    FEP_RUNNING = "fep_running"
    FEP_DONE = "fep_done"
    FAILED = "failed"
