"""Tests for ffs/output.py."""
import json

from ffs.output import print_json, print_kv, print_list_table


class TestPrintJson:
    def test_prints_dict(self, capsys):
        print_json({"key": "value"})
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed == {"key": "value"}

    def test_prints_list(self, capsys):
        print_json([1, 2, 3])
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed == [1, 2, 3]


class TestPrintKv:
    def test_renders_without_error(self, capsys):
        print_kv({"Name": "test", "Status": "done"})
        output = capsys.readouterr().out
        assert "test" in output
        assert "done" in output

    def test_renders_with_title(self, capsys):
        print_kv({"Key": "Val"}, title="My Title")
        output = capsys.readouterr().out
        assert "My Title" in output


class TestPrintListTable:
    def test_renders_rows(self, capsys):
        rows = [{"ID": "abc", "Name": "test"}]
        print_list_table(rows, ["ID", "Name"])
        output = capsys.readouterr().out
        assert "abc" in output
        assert "test" in output

    def test_renders_with_title(self, capsys):
        rows = [{"Col": "val"}]
        print_list_table(rows, ["Col"], title="Results")
        output = capsys.readouterr().out
        assert "Results" in output
