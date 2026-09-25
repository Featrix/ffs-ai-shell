"""Click extensions shared across ffs command groups."""
import difflib

import click

# What people actually type when they want help. click only accepts --help, so
# `ffs help` and `ffs ?` used to come back as "No such command 'help'." — and,
# worse, "Did you mean 'agent-help'?", which sends you to the wrong place.
HELP_ALIASES = frozenset({"help", "?", "h", "commands"})

# Dash-prefixed spellings can't live in HELP_ALIASES: click's parser consumes
# them as options before a command name is ever resolved. They're registered as
# help options on the root context instead (see cli.py).
HELP_OPTION_NAMES = ["-h", "-?", "--help"]


class DYMGroup(click.Group):
    """A click.Group that accepts `help`/`?` and suggests near-miss commands.

    e.g. `ffs networks` -> "No such command 'networks'. Did you mean 'network'?"
    """

    def resolve_command(self, ctx, args):
        if args and args[0].lower() in HELP_ALIASES:
            self._show_help(ctx, args[1:])

        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as e:
            raise click.UsageError(self._unknown_command(ctx, args[0], e), ctx=e.ctx) from None

    def _show_help(self, ctx, rest):
        """Print help for this group, or for the subcommand path in `rest`.

        So `ffs help`, `ffs help foundation` and `ffs help foundation predict`
        all do what the matching --help does. Unrecognised trailing words are
        ignored rather than made into an error — someone typing
        `ffs help me predict` still gets something useful.
        """
        target, target_ctx = self, ctx
        for name in rest:
            if not isinstance(target, click.Group):
                break
            sub = target.get_command(target_ctx, name)
            if sub is None:
                break
            target_ctx = click.Context(sub, info_name=name, parent=target_ctx)
            target = sub
        click.echo(target.get_help(target_ctx))
        ctx.exit()

    def _unknown_command(self, ctx, name, error):
        """The message for a command that doesn't exist.

        Always ends with a way forward: a close match if there is one, and the
        command that lists everything either way.
        """
        matches = difflib.get_close_matches(name, self.list_commands(ctx), n=3, cutoff=0.5)
        path = ctx.command_path or "ffs"

        lines = [error.message]
        if len(matches) == 1:
            lines.append(f"Did you mean '{matches[0]}'?")
        elif matches:
            options = "', '".join(matches)
            lines.append(f"Did you mean one of these?\n    '{options}'")
        lines.append(f"Run '{path} help' to see the available commands.")
        return "\n\n".join(lines)
