import sys

import click


def log(message) -> None:
    # stderr only: Bash captures stdout as the command's answer.
    print(message, file=sys.stderr)


class MaybeNone(click.ParamType):
    # Bash passes the literal "none" for an empty value.
    name = "text_or_none"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return value


MAYBE_NONE = MaybeNone()
