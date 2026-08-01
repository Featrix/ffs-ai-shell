"""Click extensions shared across ffs command groups."""
import difflib

import click


class DYMGroup(click.Group):
    """A click.Group that suggests near-miss matches for unknown subcommands.

    e.g. `ffs networks` -> "No such command 'networks'. Did you mean 'network'?"
    """

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            cmd_name = args[0]
            matches = difflib.get_close_matches(
                cmd_name, self.list_commands(ctx), n=3, cutoff=0.5
            )
            if not matches:
                raise
            if len(matches) == 1:
                suggestion = f"Did you mean '{matches[0]}'?"
            else:
                options = "', '".join(matches)
                suggestion = f"Did you mean one of these?\n    '{options}'"
            raise click.UsageError(f"{e.message}\n\n{suggestion}", ctx=e.ctx) from None
