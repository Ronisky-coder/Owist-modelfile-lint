from owist_modelfile_lint.parser import parse


def test_parses_basic_modelfile():
    text = "FROM llama3.2\nPARAMETER temperature 0.7\n"
    result = parse(text)
    assert len(result.instructions) == 2
    assert result.instructions[0].keyword == "FROM"
    assert result.instructions[0].args == "llama3.2"
    assert result.instructions[1].keyword == "PARAMETER"
    assert result.instructions[1].args == "temperature 0.7"
    assert not result.parse_errors


def test_case_insensitive_keywords():
    result = parse("from llama3.2\nParameter temperature 0.7\n")
    assert result.instructions[0].keyword == "FROM"
    assert result.instructions[1].keyword == "PARAMETER"


def test_skips_comments_and_blank_lines():
    text = "# this is a comment\n\nFROM llama3.2\n\n# another comment\n"
    result = parse(text)
    assert len(result.instructions) == 1
    assert result.instructions[0].keyword == "FROM"


def test_multiline_triple_quoted_value():
    text = 'SYSTEM """You are an assistant.\nBe helpful.\n"""\n'
    result = parse(text)
    assert len(result.instructions) == 1
    instr = result.instructions[0]
    assert instr.keyword == "SYSTEM"
    assert "You are an assistant." in instr.args
    assert "Be helpful." in instr.args
    assert instr.line == 1
    assert instr.end_line == 3


def test_single_line_triple_quoted_value():
    text = 'SYSTEM """You are an assistant."""\n'
    result = parse(text)
    assert result.instructions[0].args == "You are an assistant."


def test_unterminated_triple_quote_reports_parse_error():
    text = 'TEMPLATE """this never closes\n'
    result = parse(text)
    assert len(result.parse_errors) == 1
    assert "unterminated" in result.parse_errors[0][1]


def test_unrecognized_instruction_reports_parse_error():
    text = "MADEUP something\n"
    result = parse(text)
    assert len(result.instructions) == 0
    assert len(result.parse_errors) == 1
    assert "unrecognized instruction" in result.parse_errors[0][1]


def test_multiple_message_instructions():
    text = "MESSAGE user hi\nMESSAGE assistant hello\n"
    result = parse(text)
    assert len(result.instructions) == 2
    assert all(i.keyword == "MESSAGE" for i in result.instructions)
