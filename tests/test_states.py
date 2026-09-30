from molnova.states import CompoundState


def test_state_values():
    assert CompoundState.GENERATED == "generated"
    assert CompoundState.DOCKED == "docked"
    assert CompoundState.FEP_DONE == "fep_done"
