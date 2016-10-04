"""
Unit test for settings module.
"""

import argparse
import io

import pytest

from thrifty import settings

DEFAULT_FOO = '2e6'
DEFAULT_BAZ = '1e6'

DEFINITIONS = {
    'foo': settings.Definition(
        args=['--foo', '-f'],
        parser=float,
        default=DEFAULT_FOO,
        description=None,
    ),

    'bar.baz': settings.Definition(
        args=['--baz', '-b'],
        parser=float,
        default=DEFAULT_BAZ,
        description=None,
    ),

    'xyzzy': settings.Definition(
        args=['--xyzzy', '-x'],
        parser=str,
        default=None,
        description=None,
    ),
}


def test_argparse_simple():
    """Can generate argparse arguments."""
    parser = argparse.ArgumentParser()
    settings.add_argparse_arguments(parser, ['foo', 'bar.baz'],
                                    definitions=DEFINITIONS)
    args = vars(parser.parse_args(['-f', '12.34', '--baz=56.78']))
    assert args['foo'] == '12.34'
    assert args['bar.baz'] == '56.78'


def test_load_default_values():
    """Can load default values."""
    values = settings.load(None, None, DEFINITIONS)
    assert len(values) == 2
    assert values['foo'] == float(DEFAULT_FOO)
    assert values['bar.baz'] == float(DEFAULT_BAZ)


def test_load_config():
    """Can load settings from config file."""
    config = io.BytesIO("bar.baz:   1234.56")
    values = settings.load(None, config, DEFINITIONS)
    assert values['foo'] == float(DEFAULT_FOO)
    assert values['bar.baz'] == 1234.56


def test_load_syntax_error():
    """Throw ConfigSyntaxError if config's syntax is invalid."""
    config = io.BytesIO("foobar")
    with pytest.raises(settings.ConfigSyntaxError):
        settings.load(None, config, DEFINITIONS)


def test_load_key_error_config():
    """Throw SettingKeyError if a setting in the config file is not defined."""
    config = io.BytesIO("foobar: 1")
    with pytest.raises(settings.SettingKeyError):
        settings.load(None, config, DEFINITIONS)


def test_load_key_error_arg():
    """Throw SettingKeyError if arg contains a key without a definition."""
    args = {'foobar': '1'}
    with pytest.raises(settings.SettingKeyError):
        settings.load(args, None, DEFINITIONS)


def test_load_args():
    """Can args override config and default."""
    config = io.BytesIO("bar.baz:   12.34")
    args = {'bar.baz': "7.8", 'foo': '9.0'}
    values = settings.load(args, config, DEFINITIONS)
    assert values['foo'] == 9.0
    assert values['bar.baz'] == 7.8


def test_loadargs(tmpdir):
    """End-to-end test for load_args function."""
    tmp = tmpdir.join("thrift.cfg")
    tmp.write("xyzzy: xyz\nfoo: 1.2\nbar.baz: 3.6")

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', dest='a')
    argv = ['-a', 'extra', '--foo=2.3', '-c', tmp.strpath]

    config, args = settings.load_args(parser, ['xyzzy', 'foo'],
                                      argv=argv, definitions=DEFINITIONS)
    args.pop('verbose')
    assert len(config) == 2
    assert len(args) == 1
    assert config['xyzzy'] == 'xyz'
    assert config['foo'] == 2.3
    assert args['a'] == 'extra'
