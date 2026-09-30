from enum import StrEnum


class CompoundState(StrEnum):
    REFERENCE = "reference"
    GENERATED = "generated"
    SYNTHETIC_RUNNING = "synthetic_running"
    LIGPREP_RUNNING = "ligprep_running"
    LIGPREPPED = "ligprepped"
    GLIDE_RUNNING = "glide_running"
    DOCKED = "docked"
    GBSA_RUNNING = "gbsa_running"
    GBSA_DONE = "gbsa_done"
    FEP_RUNNING = "fep_running"
    FEP_DONE = "fep_done"
    FAILED = "failed"
