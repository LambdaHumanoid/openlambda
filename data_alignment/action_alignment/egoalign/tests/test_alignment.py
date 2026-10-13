import json
import numpy as np
import pandas as pd
import pytest
from egoalign.alignment import parse_alignment
from scripts.human_data.build_smpl_endpoint_retarget_candidate import aligned_raw_dataframe

@pytest.mark.parametrize('key', ['camera_duration_s', 'gopro_duration_s'])
def test_alias_and_selection(tmp_path, key):
    for cam in ['gopro', 'level-cam']:
        (tmp_path/f'episode_000000.{cam}_alignment.json').write_text(json.dumps({'offset_s':1, key:2}))
    selected, indices = aligned_raw_dataframe(pd.DataFrame({'timestamp':np.arange(5)}), sync_dir=tmp_path, episode_index=0, expected_rows=3)
    assert indices.tolist() == [1,2,3]
    assert selected.timestamp.tolist() == [1,2,3]

@pytest.mark.parametrize('value', [None, 0, -1, float('nan'), float('inf'), 'bad'])
def test_bad_duration(value):
    with pytest.raises(ValueError): parse_alignment({'offset_s':0, 'gopro_duration_s':value})

def test_missing_and_conflicting():
    assert parse_alignment({'offset_s':0}) == (0,None)
    with pytest.raises(ValueError): parse_alignment({'offset_s':0},require_duration=True)
    with pytest.raises(ValueError): parse_alignment({'offset_s':0,'camera_duration_s':1,'gopro_duration_s':2})
    assert parse_alignment({'offset_s':0,'camera_duration_s':1,'gopro_duration_s':1}) == (0,1)

def test_full_preserves_rows_without_recropping(tmp_path):
    raw=pd.DataFrame({'timestamp':np.arange(5)})
    selected, indices=aligned_raw_dataframe(raw,sync_dir=tmp_path,episode_index=0,expected_rows=5,raw_selection_mode='full')
    assert indices.tolist()==list(range(5))
    pd.testing.assert_frame_equal(raw,selected)
