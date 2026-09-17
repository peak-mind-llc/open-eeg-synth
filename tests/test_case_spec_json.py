import json

from open_eeg_synth.brain.plants import FocalSlow, RhythmicBursts
from open_eeg_synth.case import ArtifactSpec, CaseSpec, case_id_for
from open_eeg_synth.recipes import resting_case


def test_full_spec_survives_json_and_ids_are_stable():
    spec = resting_case(
        99,
        duration_s=30.0,
        drowsy_from_s=15.0,
        plants=(FocalSlow("F7"), RhythmicBursts(("Fz",), amp_uv=25.0)),
        artifacts=(
            ArtifactSpec("blink", {"median_uv": 90.0}),
            ArtifactSpec("emg", {"side": "both"}),
        ),
    )
    text = json.dumps(spec.to_dict(), sort_keys=True)
    back = CaseSpec.from_dict(json.loads(text))
    assert back == spec
    assert case_id_for(back) == case_id_for(spec)
    assert case_id_for(resting_case(100, duration_s=30.0)) != case_id_for(spec)
    assert spec.to_dict()["conditions"][0]["timeline"]["segments"][1]["state"] == "drowsy"
