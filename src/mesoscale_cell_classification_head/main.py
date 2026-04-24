"""Placeholder entry point for the mesoscale-cell-classification-head package."""


def beep_boop(msg: str) -> str:
    """Replace every 'o' with '0' in ``msg``.

    Parameters
    ----------
    msg : str
        Input string.

    Returns
    -------
    str
        ``msg`` with all 'o' characters replaced by '0'.
    """
    return msg.replace("o", "0")


def main() -> None:
    """Print a greeting via :func:`beep_boop`."""
    print(beep_boop("Howdy"))


if __name__ == "__main__":
    main()
