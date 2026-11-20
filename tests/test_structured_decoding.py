import pytest
from chest_xpert.structured_decoding import allowed_status_tokens

def test_closed_decoder_only_forces_schema_and_label_spelling_not_a_class():
 labels={'present':[1],'absent':[2,3],'uncertain':[4,5]}
 assert allowed_status_tokens(labels,[],99)==[1,2,4]
 assert allowed_status_tokens(labels,[1],99)==[99]
 assert allowed_status_tokens(labels,[2],99)==[3]
 assert allowed_status_tokens(labels,[4,5],99)==[99]
 with pytest.raises(ValueError):allowed_status_tokens(labels,[8],99)
