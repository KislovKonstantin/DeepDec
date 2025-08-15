from deepdec import ResponseParser

def test_parser():
    response = """
    [ANSWER] Correct answer
    [MARK] YES
    [REVISION NOTES]"""
    assert ResponseParser.extract_answer(response) == """Correct answer
    [MARK] YES
    [REVISION NOTES]"""
    assert ResponseParser.extract_mark(response) == "YES"
    assert ResponseParser.extract_notes(response, "YES") == ""