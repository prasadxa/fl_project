def escape_latex(s: str) -> str:
    s = str(s)
    latex_special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    trans_table = str.maketrans(latex_special_chars)
    return s.translate(trans_table)

print(escape_latex(r"test \ {} _ ^ % $ # & ~"))
