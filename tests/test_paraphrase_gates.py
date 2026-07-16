from src.paraphrase_gates import deterministic_gates, normalized_numbers


def test_preserved_numbers_units_and_rewording_pass():
    original = "Mia buys 3 boxes with 12 pencils each. How many pencils does she buy?"
    paraphrase = (
        "How many pencils are purchased by Mia when each of the 3 boxes contains 12 pencils?"
    )
    gates = deterministic_gates(original, paraphrase, 0.10)
    assert gates["passed"] is True


def test_changed_number_is_rejected():
    gates = deterministic_gates("A runner covers 5 miles.", "A runner covers 6 miles.", 0.0)
    assert gates["passed"] is False
    assert gates["number_multiset_match"] is False


def test_number_normalization_preserves_multiplicity():
    assert normalized_numbers("1,000 plus 5 and 5") == normalized_numbers("1000 with 5 then 5")
