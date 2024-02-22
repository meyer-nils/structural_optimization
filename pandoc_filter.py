#!/usr/bin/env python3
import pandocfilters as pf


def objectives_to_bold(key, value, format, meta):
    if key == "Para":
        if len(value) > 1:
            # Examples
            if value[0]["t"] == "Span" and value[1]["t"] == "Span":
                return pf.Para(
                    [pf.Strong([pf.Str("Example: "), value[0]]), pf.LineBreak()]
                    + value[2:]
                )
            # Leanring Objectives
            if value[0]["t"] == "Span" and value[1]["t"] != "Span":
                return pf.Para(
                    [pf.Strong([pf.Str("Learning Objectives")]), pf.LineBreak()]
                    + value[2:]
                )


if __name__ == "__main__":
    pf.toJSONFilter(objectives_to_bold)
